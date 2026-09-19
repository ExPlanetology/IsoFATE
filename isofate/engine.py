# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX/diffrax integration core for `isofate_coupler.isocalc_jax` - split out so it can be
`eqx.filter_jit`'d independently of `isocalc_jax`'s plain-Python/NumPy setup (atmodeller
scaffolding, etc.), which isn't worth tracing.
"""

import functools

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from isofate.escape import EscapeState
from isofate.escape_core import EscapeNumberFlux
from isofate.isofunks import R_atm, R_env
from isofate.parameters import Parameters
from isofate.utils import gravitational_acceleration


def _algebraic(
    t: ArrayLike,
    y: Array,
    thermal: bool,
    t_total: ArrayLike,
    parameters: Parameters,
) -> dict[str, Array]:
    """Everything derivable from (t, y) alone, given the run's fixed quantities - shared by
    `_vector_field` (during integration) and the post-solve vmapped diagnostics in
    `_integrate_isocalc_jax`, so the physics chain (radius_env -> ... -> Phi) is defined once.

    A top-level function rather than a closure nested inside `_integrate_isocalc_jax`: its fixed
    arguments are bound via `functools.partial` at the call site instead. That's safe here (as
    opposed to threading them through diffrax's own `args` parameter) because a partial's bound
    values are baked into the callable object itself - JAX never re-flattens them as pytree state
    the way it would if they were carried through diffrax's internal `lax.while_loop` via `args`.
    So no `eqx.field(static=True)` bookkeeping is needed to keep e.g. `thermal` (branched on with
    a raw Python `if` inside `R_env`) or `escape`'s own bool/str config fields from being
    force-converted into traced arrays.

    `M_atm` is not separate integrated state: with no atmodeller coupling in this simplified case,
    dM_atm/dt is exactly dot(dy/dt, atomic_masses), so M_atm(t) is always exactly
    dot(y(t), atomic_masses) - it's computed fresh here from y rather than tracked by
    `_integrate_isocalc_jax`'s diffrax solve. This also means `planet.f_atm` is never read for the
    initial condition: the initial atmosphere mass is `dot(isofate_species_abund, atomic_masses)`.

    For the same reason, the Bondi radius R_B is recomputed here from the current (evolving) mu at
    every call, rather than fixed once from a bootstrap `options.mu` guess before the trajectory
    exists - there's no "before mu exists" moment to bootstrap, since mu is always computable
    straight from y.

    Clips y to >= 0 before use: the adaptive step-size controller can propose trial steps that
    briefly overshoot into slightly negative territory near exhaustion (the original discrete
    loop clipped `y` after every step for the same reason), and a negative M_atm/f_atm would
    otherwise feed a fractional power (R_env's c2**0.59 term) with a negative base.

    Args:
        thermal: A plain Python bool, not a traced array - see the note above about why binding
            this via `functools.partial` (rather than diffrax's `args`) keeps it safely static.
    """
    system = parameters.system
    escape = parameters.escape_mechanism
    escape_number_flux = parameters.escape_number_flux

    atomic_masses = escape_number_flux.species.atomic_masses
    Mp = system.planet.mass
    T = system.equilibrium_temperature
    d = system.semi_major_axis
    Fp = system.insolation

    y = jnp.maximum(y, 0.0)
    N_tot = jnp.sum(y)
    safe_N_tot = jnp.where(N_tot > 0, N_tot, 1.0)
    mu = escape_number_flux.species.atmosphere_mean_mu(y)
    x = jnp.where(N_tot > 0, y / safe_N_tot, jnp.zeros_like(y))

    M_atm = jnp.dot(y, atomic_masses)  # y already clipped >= 0 above, so M_atm is too
    f_atm = M_atm / Mp
    radius_env = R_env(Mp, f_atm, Fp, t, thermal)
    radius_atm = R_atm(T, Mp, system.planet.rocky_radius, radius_env, mu)
    R_B = system.bondi_radius(mu, T)  # recomputed from the current mu, not a fixed bootstrap
    # was `min(R_B, R_H, radius_p)` in isocalc's plain-Python loop - Python's builtin min() on
    # a traced value, same class of fix as Fxuv/Phi_1_2/Phi_minor_species earlier this session.
    radius_p = jnp.minimum(
        R_B, jnp.minimum(system.hill_radius, system.planet.rocky_radius + radius_atm + radius_env)
    )

    Vpot = system.gravitational_potential(radius_p)
    A = 4 * jnp.pi * radius_p**2
    g = gravitational_acceleration(Mp, radius_p)

    state = EscapeState(
        radius_p=radius_p,
        Mp=Mp,
        T=T,
        Vpot=Vpot,
        d=d,
        A=A,
        mu=mu,
        radius_env=radius_env,
        f_atm=f_atm,
        t_now=t,
        t_total=t_total,
    )
    phi = escape.compute_mass_flux(state)
    Phi, phi_c = escape_number_flux.get_number_flux(y, T, g, phi)

    return dict(
        mu=mu,
        x=x,
        M_atm=M_atm,
        f_atm=f_atm,
        radius_env=radius_env,
        radius_p=radius_p,
        Vpot=Vpot,
        A=A,
        phi=phi,
        phi_c=phi_c,
        Phi=Phi,
    )


def _vector_field(
    t: ArrayLike,
    y: Array,
    _args,
    *,
    thermal: bool,
    t_total: ArrayLike,
    parameters: Parameters,
) -> Array:
    """RHS of the isocalc ODE system - the only genuine integration state's derivative (dy/dt).
    Everything else, including M_atm, is recomputed from the solution afterward (see
    `_integrate_isocalc_jax`).

    `diffrax.ODETerm` calls its vector field as exactly `vf(t, y, args)`; the keyword-only
    parameters here are bound via `functools.partial` before this is handed to `ODETerm` (see
    `_algebraic`'s docstring for why binding this way, rather than via `args`, is safe) - `args`
    itself is unused (`_integrate_isocalc_jax` passes `args=None` to `diffeqsolve`).
    """
    alg = _algebraic(t, y, thermal, t_total, parameters)
    return -alg["Phi"] * alg["A"]


def _exhausted(
    t: ArrayLike,
    y: Array,
    _args,
    *,
    escape_number_flux: EscapeNumberFlux,
    M_atm0: ArrayLike,
    sum_y0: ArrayLike,
    exhaustion_fraction: ArrayLike,
    **kwargs,
) -> Array:
    """Event condition: entire atmosphere lost - replaces isocalc's `if M_atm <= 0 or sum(y) <=
    0: break`.

    Exact `M_atm <= 0`/`sum(y) <= 0` makes the ODE's right-hand side effectively singular as the
    dominant species' abundance -> 0 (Phi_minor_species' N_2/N_1 term blows up), which stalls
    Tsit5's adaptive step-size controller (step size underflows before the exact zero is ever
    reached, rather than converging). Firing once M_atm/sum(y) drop below a small-but-nonzero
    fraction of their initial values avoids this while still meaning "the atmosphere is gone" to
    any reasonable tolerance.

    `diffrax.Event` calls its condition as exactly `cond_fn(t, y, args, **kwargs)`; the
    keyword-only parameters here are bound via `functools.partial` before this is handed to
    `Event` (see `_algebraic`'s docstring for why binding this way, rather than via `args`, is
    safe) - `args` itself is unused.
    """
    M_atm = jnp.dot(y, escape_number_flux.species.atomic_masses)
    return (M_atm <= exhaustion_fraction * M_atm0) | (jnp.sum(y) <= exhaustion_fraction * sum_y0)


@eqx.filter_jit
def _integrate_isocalc_jax(
    t0_seconds: ArrayLike,
    t_a: Array,
    y0: Array,
    thermal: bool,
    t_total: ArrayLike,
    parameters: Parameters,
) -> tuple[Array, dict[str, Array]]:
    """The diffrax-solvable core of `isocalc_jax`: an ODE integration over `y` (species
    abundances) plus the diagnostics back-computed from its saved trajectory. Split out from
    `isocalc_jax` itself (which also does plain-Python/NumPy setup - atmodeller scaffolding, etc.
    - not worth tracing) so this can be `eqx.filter_jit`'d: repeated calls (e.g. across an
    MCMC/optimization loop) then reuse one compiled program instead of dispatching every jnp op
    individually.

    `_algebraic`/`_vector_field`/`_exhausted` are top-level functions (not closures nested here);
    see `_algebraic`'s docstring for the M_atm/R_B design and why `functools.partial` (used below
    to bind them to this call's fixed values) is the safe way to give them that data.

    `parameters` is itself an `eqx.Module` (pytree) - `filter_jit` already partitions its array
    leaves (traced) from any non-array config fields (held static) without needing explicit
    annotations, so it's passed through as-is.

    Args:
        thermal: A plain Python bool, not a traced array - `eqx.filter_jit` holds non-array
            arguments static automatically (required here, since `R_env` branches on it with a
            raw Python `if`). Every other argument here is expected to be an actual array (see
            `isocalc_jax`'s call site, which wraps its locals in `jnp.asarray` before calling) so
            that varying them across calls reuses this same compiled program rather than
            retracing.

    Returns:
        (y_a, alg_a): the saved trajectory (already inf-filled with the held terminal state past
        the exhaustion event) and the vmapped `_algebraic` diagnostics for every t_a entry
        (including the derived `M_atm`/`f_atm`).
    """
    n_tot = t_a.shape[0]
    escape_number_flux = parameters.escape_number_flux
    atomic_masses = escape_number_flux.species.atomic_masses

    sum_y0 = jnp.sum(y0)
    M_atm0 = jnp.dot(y0, atomic_masses)
    _exhaustion_fraction = 1e-6

    vector_field = functools.partial(
        _vector_field,
        thermal=thermal,
        t_total=t_total,
        parameters=parameters,
    )
    exhausted = functools.partial(
        _exhausted,
        escape_number_flux=escape_number_flux,
        M_atm0=M_atm0,
        sum_y0=sum_y0,
        exhaustion_fraction=_exhaustion_fraction,
    )

    term = diffrax.ODETerm(vector_field)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0_seconds,
        # Not `t0_seconds + t_total`: t_a's last entry runs one delta_t past that (a pre-existing
        # quirk of t_a's own formula, shared with isocalc, harmless there since t_a is only a
        # diagnostic label in the discrete loop) - diffrax requires saveat.ts to lie within
        # [t0, t1], so t1 is taken directly from t_a instead of re-derived independently.
        t1=t_a[-1],
        dt0=None,
        y0=y0,
        args=None,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e3),
        saveat=diffrax.SaveAt(ts=t_a),
        event=diffrax.Event(cond_fn=exhausted),
        max_steps=100_000,
    )
    y_a = sol.ys

    # Entries of t_a at/after the point where the event fired come back as inf (diffrax's
    # SaveAt(ts=...)+Event interaction - empirically confirmed, not documented behavior). Replace
    # them by holding the last finite (i.e. last actually-integrated) trajectory value constant,
    # falling back to the initial condition when no saved entry is finite at all (e.g. a short
    # exhaustion time relative to the requested output cadence, so every t_a entry postdates it).
    finite_mask = jnp.all(jnp.isfinite(y_a), axis=1)
    any_finite = jnp.any(finite_mask)
    last_finite_idx = jnp.max(jnp.where(finite_mask, jnp.arange(n_tot), -1))
    last_finite_idx = jnp.maximum(last_finite_idx, 0)
    y_fallback = jnp.where(any_finite, y_a[last_finite_idx], y0)
    y_a = jnp.where(finite_mask[:, None], y_a, y_fallback)

    # Back-compute every diagnostic (including the derived M_atm) from the saved trajectory in one
    # vmapped pass, rather than a per-timestep Python loop. Only t/y vary per output point - the
    # rest are shared/broadcast (in_axes=None), matching what closing over them would have done.
    alg_a = jax.vmap(_algebraic, in_axes=(0, 0, None, None, None))(
        t_a, y_a, thermal, t_total, parameters
    )

    return y_a, alg_a
