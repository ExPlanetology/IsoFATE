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

from isofate.constants import const
from isofate.escape.fractionation import EscapeNumberFluxBase
from isofate.escape.mechanisms import EscapeState
from isofate.parameters import Parameters
from isofate.system import Planet, System
from isofate.utils import gravitational_acceleration


def convective_envelope_thickness(
    parameters: Parameters, y, Fp, age, thermal: bool = True
) -> ArrayLike:
    """Radial thickness contributed by the convective portion of the H/He envelope (down to the
    radiative-convective boundary), one of the three additive terms making up the total planet
    radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez & Fortney 2014.

    Args:
        parameters: Simulation parameters - `parameters.system.planet.mass` [kg] and the envelope
            mass fraction (`parameters.atmosphere_mass_fraction(y)`) are read from it.
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species` (see
            `Parameters.atmosphere_mass_fraction`).
        Fp: incident bolometric flux [W/m2]
        age: age [s]
        thermal: toggles radius dependence on thermal evolution [True/False]

    Returns:
        Thickness contribution of the convective envelope [m]
    """
    planet = parameters.system.planet
    f_env = parameters.atmosphere_mass_fraction(y)

    c1 = planet.mass / const.Me  # Me = Earth mass [kg]
    # FIXME: Collin to clarify the magic number below
    c2 = f_env / 0.05
    c3 = Fp / const.Fe  # Fe = Earth incident bolometric flux [W/m2]
    if thermal:
        # FIXME: Collin to clarify the magic number below
        c4 = age * const.s2yr / 5e9
    else:
        c4 = 1

    return 2.06 * const.Re * c1 ** (-0.21) * c2 ** (0.59) * c3 ** (0.044) * c4 ** (-0.18)


def radiative_atmosphere_thickness(Teq, planet: Planet, envelope_thickness, mu) -> ArrayLike:
    """Radial thickness contributed by the radiative outer atmosphere above the
    radiative-convective boundary - the third of the three additive terms making up the total
    planet radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez & Fortney 2014

    Args:
        Teq: planet equilibrium temperature [K]
        planet: Planet parameters - `planet.mass` [kg] and `planet.rocky_radius` [m] are read
            from it.
        envelope_thickness: convective-envelope thickness contribution [m] (see
            `convective_envelope_thickness`)
        mu: mean molecular mass [kg/particle]

    Returns:
        Thickness contribution of the radiative atmosphere [m]
    """
    # field strength at base of atm
    g = const.G * planet.mass / ((planet.rocky_radius + envelope_thickness) ** 2)
    H = const.kb * Teq / (g * mu)  # scale height

    return 9 * H


def total_radius(parameters: Parameters, y, Fp, age, Teq, mu, thermal: bool = True) -> Array:
    """Total planet radius [m]: the rocky-core radius plus the two additive envelope/atmosphere
    thickness terms, computed internally.

    R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Args:
        parameters: Simulation parameters - `parameters.system.planet` is read from it (and
            passed through to `convective_envelope_thickness`/`radiative_atmosphere_thickness`).
        y: Per-species abundances [atoms] (see `convective_envelope_thickness`)
        Fp: incident bolometric flux [W/m2] (see `convective_envelope_thickness`)
        age: age [s] (see `convective_envelope_thickness`)
        Teq: planet equilibrium temperature [K] (see `radiative_atmosphere_thickness`)
        mu: mean molecular mass [kg/particle] (see `radiative_atmosphere_thickness`)
        thermal: toggles radius dependence on thermal evolution [True/False]

    Returns:
        Total planet radius [m]
    """
    planet = parameters.system.planet
    envelope_thickness = convective_envelope_thickness(parameters, y, Fp, age, thermal)
    atmosphere_thickness = radiative_atmosphere_thickness(Teq, planet, envelope_thickness, mu)

    return planet.rocky_radius + envelope_thickness + atmosphere_thickness


def bondi_radius(Mp, mu, Teq, gamma=7 / 5):
    """Bondi radius calculation.

    Copy of `isofate.isofunks.R_Bondi`, kept unchanged there for `isoplot.py`.

    Args:
        Mp: planetary mass [kg]
        mu: average particle mass [kg]
        Teq: planetary equilibrium temperature [K]
        gamma: adiabatic index (heat capacity ratio) [ndim]

    Returns:
        Bondi radius [m]
    """
    return (gamma - 1) * const.G * Mp * mu / (gamma * const.kb * Teq)


def tidal_reduction_factor(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
    """Gravitational potential reduction factor due to stellar tidal forces (Erkaev et al. 2007).

    Args:
        system: System parameters - `system.planet.mass`, `system.star.mass`, and
            `system.semi_major_axis` are read from it.
        radius: Planet radius [m] - pass the current total (rocky + envelope) radius when it's
            time-evolving.
        floor: Minimum value returned. Defaults to `0.01`.

    Returns:
        Gravitational potential reduction factor [ndim]
    """
    delta = system.planet.mass / system.star.mass
    lam = system.semi_major_axis / radius
    zeta = lam * (delta / 3) ** (1 / 3)
    V_reduction = 1 - 3 / 2 / zeta + 1 / 2 / jnp.power(zeta, 3)

    return jnp.maximum(V_reduction, floor)


def gravitational_potential(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
    """Gravitational potential at the outer layer [J/kg], reduced by `tidal_reduction_factor`.

    Args:
        system: System parameters - `system.planet.mass` is read from it (and passed through to
            `tidal_reduction_factor`).
        radius: Planet radius [m] - pass the current total (rocky + envelope) radius when it's
            time-evolving.
        floor: Minimum `tidal_reduction_factor` value used. Defaults to `0.01`.

    Returns:
        Gravitational potential [J/kg]
    """
    K: Array = tidal_reduction_factor(system, radius, floor)

    return K * const.G * system.planet.mass / radius


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
    a raw Python `if` inside `convective_envelope_thickness`) or `escape`'s own bool/str config
    fields from being force-converted into traced arrays.

    `M_atm` is not separate integrated state: with no atmodeller coupling in this simplified case,
    dM_atm/dt is exactly dot(dy/dt, atomic_masses), so M_atm(t) is always exactly
    dot(y(t), atomic_masses) - it's computed fresh here from y rather than tracked by
    `_integrate_isocalc_jax`'s diffrax solve. This also means `planet.f_atm` is never read for the
    initial condition: the initial atmosphere mass is `dot(isofate_species_abund, atomic_masses)`.

    For the same reason, the Bondi radius R_B is recomputed here from the current (evolving) mu at
    every call, rather than fixed once from a bootstrap guess before the trajectory exists -
    there's no "before mu exists" moment to bootstrap, since mu is always computable straight from
    y (there is no `IsocalcOptions.mu` field to bootstrap from).

    Clips y to >= 0 before use: the adaptive step-size controller can propose trial steps that
    briefly overshoot into slightly negative territory near exhaustion (the original discrete
    loop clipped `y` after every step for the same reason), and a negative M_atm/f_atm would
    otherwise feed a fractional power (convective_envelope_thickness's c2**0.59 term) with a
    negative base.

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
    mu = parameters.atmosphere_mean_mu(y)
    x = jnp.where(N_tot > 0, y / safe_N_tot, jnp.zeros_like(y))

    M_atm = jnp.dot(y, atomic_masses)  # y already clipped >= 0 above, so M_atm is too
    f_atm = parameters.atmosphere_mass_fraction(y)
    radius_env = convective_envelope_thickness(parameters, y, Fp, t, thermal)
    R_B = bondi_radius(Mp, mu, T)  # recomputed from the current mu, not a fixed bootstrap
    # was `min(R_B, R_H, radius_p)` in isocalc's plain-Python loop - Python's builtin min() on
    # a traced value, same class of fix as Fxuv/Phi_1_2/Phi_minor_species earlier this session.
    radius_p = jnp.minimum(
        R_B,
        jnp.minimum(system.hill_radius, total_radius(parameters, y, Fp, t, T, mu, thermal)),
    )

    Vpot = gravitational_potential(system, radius_p)
    A = 4 * jnp.pi * radius_p**2
    g = gravitational_acceleration(Mp, radius_p)

    state = EscapeState(
        radius_p=radius_p,
        Mp=Mp,
        T=T,
        Vpot=Vpot,
        d=d,
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
    escape_number_flux: EscapeNumberFluxBase,
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
            arguments static automatically (required here, since `convective_envelope_thickness` branches
            on it with a raw Python `if`). Every other argument here is expected to be an actual
            array (see `isocalc_jax`'s call site, which wraps its locals in `jnp.asarray` before
            calling) so that varying them across calls reuses this same compiled program rather
            than retracing.

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

    # TODO: Should be done outside the time loop
    # Back-compute every diagnostic (including the derived M_atm) from the saved trajectory in one
    # vmapped pass, rather than a per-timestep Python loop. Only t/y vary per output point - the
    # rest are shared/broadcast (in_axes=None), matching what closing over them would have done.
    alg_a = jax.vmap(_algebraic, in_axes=(0, 0, None, None, None))(
        t_a, y_a, thermal, t_total, parameters
    )

    return y_a, alg_a
