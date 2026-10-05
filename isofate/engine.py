# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX/diffrax integration core for `isofate_coupler.isocalc_jax`: the planet-radius/escape physics
functions and `IsocalcIntegrator`, which owns the time stepping - split out so it can be
`eqx.filter_jit`'d independently of `isocalc_jax`'s plain-Python/NumPy setup (atmodeller
scaffolding, etc.), which isn't worth tracing.
"""

from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from atmodeller.jax_utils import as_j64
from jax import Array
from jaxtyping import ArrayLike

from isofate.constants import const
from isofate.escape.fractionation import EscapeNumberFluxBase
from isofate.escape.mechanisms import EscapeMechanism, EscapeState
from isofate.parameters import Parameters
from isofate.system import Planet, System
from isofate.utils import (
    gravitational_acceleration,
    gravitational_potential,
    scale_height,
    sphere_area,
)

EXHAUSTION_FRACTION: float = 1e-6
"""Fraction of the initial atmospheric mass below which the atmosphere counts as exhausted.

Not exactly zero: as the dominant species' abundance -> 0, the ODE's right-hand side becomes
effectively singular (Phi_minor_species' N_2/N_1 term blows up), which stalls Tsit5's adaptive
step-size controller (the step size underflows before the exact zero is ever reached). A small but
nonzero floor still means "the atmosphere is gone" to any reasonable tolerance.
"""

_PLACEHOLDER_REDUCTION: float = 0.01
"""Fractional reduction of every abundance applied by `placeholder_reequilibration`."""

_RESTART_DT0_FRACTION: float = 1e-6
"""Initial step on restarts in `integrate_segments`, as a fraction of the remaining time span."""


def convective_envelope_thickness(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Radial thickness contributed by the convective portion of the H/He envelope (down to the
    radiative-convective boundary), one of the three additive terms making up the total planet
    radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez and Fortney, 2014.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Thickness contribution of the convective envelope [m]
    """
    planet: Planet = parameters.system.planet
    envelope_mass_fraction: Array = parameters.envelope_mass_fraction(y)
    thermal: bool = parameters.isocalc_options.thermal
    Fp: Array = parameters.system.insolation

    c1: Array = planet.mass / const.Me  # Me = Earth mass [kg]
    # FIXME: Collin to clarify the magic number below
    c2: Array = envelope_mass_fraction / 0.05
    c3: Array = Fp / const.Fe  # Fe = Earth incident bolometric flux [W/m2]
    if thermal:
        # FIXME: Collin to clarify the magic number below
        c4: ArrayLike = age * const.s2yr / 5e9
    else:
        c4 = 1

    return 2.06 * const.Re * c1 ** (-0.21) * c2 ** (0.59) * c3 ** (0.044) * c4 ** (-0.18)


def radiative_atmosphere_thickness(parameters: Parameters, y: Array, age: ArrayLike) -> ArrayLike:
    """Radial thickness contributed by the radiative outer atmosphere above the
    radiative-convective boundary - the third of the three additive terms making up the total
    planet radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez and Fortney, 2014.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Thickness contribution of the radiative atmosphere [m]
    """
    planet: Planet = parameters.system.planet
    envelope_thickness: ArrayLike = convective_envelope_thickness(parameters, y, age)
    mu: Array = parameters.atmosphere_mean_mu(y)
    equilibrium_temperature: Array = parameters.system.equilibrium_temperature

    g: ArrayLike = gravitational_acceleration(
        planet.mass, planet.rocky_radius + envelope_thickness
    )

    return 9 * scale_height(equilibrium_temperature, mu, g)


def total_radius(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Total planet radius [m]: the rocky-core radius plus the two additive envelope/atmosphere
    thickness terms, computed internally.

    R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Total planet radius [m]
    """
    planet: Planet = parameters.system.planet
    envelope_thickness: Array = convective_envelope_thickness(parameters, y, age)
    atmosphere_thickness: ArrayLike = radiative_atmosphere_thickness(parameters, y, age)

    return planet.rocky_radius + envelope_thickness + atmosphere_thickness


def bondi_radius(parameters: Parameters, y: Array):
    """Bondi radius calculation.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`

    Returns:
        Bondi radius [m]
    """
    planet_mass: Array = parameters.system.planet.mass
    mu: Array = parameters.atmosphere_mean_mu(y)
    equilibrium_temperature: Array = parameters.system.equilibrium_temperature
    gamma: ArrayLike = parameters.atmosphere.adiabatic_index

    return (gamma - 1) * const.G * planet_mass * mu / (gamma * const.kb * equilibrium_temperature)


def escape_radius(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Effective outer radius of the planet, from which the atmosphere escapes.

    The structural radius (`total_radius`) capped by the Bondi radius, beyond which gas is not
    thermally bound to the planet, and by the Hill radius, beyond which the star's gravity
    dominates.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Escape radius [m]
    """
    return jnp.minimum(
        bondi_radius(parameters, y),
        jnp.minimum(parameters.system.hill_radius, total_radius(parameters, y, age)),
    )


def tidal_reduction_factor(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
    """Gravitational potential reduction factor due to stellar tidal forces (Erkaev et al. 2007).

    Args:
        system: System parameters
        radius: Planet radius [m] - pass the current total (rocky + envelope) radius when it's
            time-evolving.
        floor: Minimum value returned. Defaults to `0.01`.

    Returns:
        Gravitational potential reduction factor [ndim]
    """
    delta: Array = system.planet.mass / system.star.mass
    lam: Array = system.semi_major_axis / radius
    zeta: Array = lam * (delta / 3) ** (1 / 3)
    V_reduction: Array = 1 - 3 / 2 / zeta + 1 / 2 / jnp.power(zeta, 3)

    return jnp.maximum(V_reduction, floor)


def tidal_gravitational_potential(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
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
    tidal_factor: Array = tidal_reduction_factor(system, radius, floor)

    return tidal_factor * gravitational_potential(system.planet.mass, radius)


def _hold_last_finite(y_a: Array, y0: Array) -> Array:
    """Replaces the inf-filled entries of a saved trajectory with its last finite entry.

    Entries of `t_a` at/after the point where a diffrax event fired come back as inf (diffrax's
    SaveAt(ts=...)+Event interaction - empirically confirmed, not documented behavior). These are
    replaced by holding the last finite (i.e. last actually-integrated) trajectory value constant,
    falling back to the initial condition when no saved entry is finite at all (e.g. a short
    exhaustion time relative to the requested output cadence, so every `t_a` entry postdates it).

    Args:
        y_a: Saved trajectory [atoms], one row per output time
        y0: Initial condition [atoms]

    Returns:
        Trajectory with every non-finite row replaced
    """
    n_tot = y_a.shape[0]
    finite_mask = jnp.all(jnp.isfinite(y_a), axis=1)
    any_finite = jnp.any(finite_mask)
    last_finite_idx = jnp.max(jnp.where(finite_mask, jnp.arange(n_tot), -1))
    last_finite_idx = jnp.maximum(last_finite_idx, 0)
    y_fallback = jnp.where(any_finite, y_a[last_finite_idx], y0)

    return jnp.where(finite_mask[:, None], y_a, y_fallback)


class IntegrationStop(eqx.Module):
    """Where and why an `IsocalcIntegrator.integrate` call stopped.

    Args:
        t: Stop time [s] - the end of the requested output times if the event did not fire
        y: Per-species abundances at the stop time [atoms]
        mass_lost: True if the mass-loss event fired (see `IsocalcIntegrator.mass_lost`)
    """

    t: Array
    y: Array
    mass_lost: Array


class IsocalcIntegrator(eqx.Module):
    """Time integration of the atmospheric-escape ODE for one isocalc run.

    The only integrated state is `y`, the per-species atmospheric abundances [atoms]. Everything
    else (`M_atm`, `f_atm`, `mu`, radii, fluxes) is derived from `(t, y)` by `algebraic`, which is
    shared by `vector_field` during integration and the post-solve diagnostics in `integrate`, so
    the physics chain (radius_env -> ... -> Phi) is defined once.

    `parameters` is an `eqx.Module`, so `eqx.filter_jit` (on `integrate`) and diffrax's own
    filtering of `ODETerm(self.vector_field)` trace its array leaves while holding its non-array
    config fields static - e.g. `parameters.isocalc_options.thermal`, which
    `convective_envelope_thickness` branches on with a raw Python `if`.

    Args:
        parameters: Simulation parameters
        t_end: End time of the simulation, i.e. system age at the end of the run [s]
        rtol: Relative tolerance of the step-size controller. Defaults to ``1e-6``.
        atol: Absolute tolerance of the step-size controller [atoms]. Defaults to ``1e3``.
        max_steps: Maximum number of solver steps. Defaults to ``100_000``.
    """

    # TODO: Might not be best to have parameters live on the integrator, but rather be passed in?
    parameters: Parameters
    # An array rather than a Python float, so that a different `time` doesn't force a retrace
    t_end: Array = eqx.field(converter=as_j64)
    rtol: float = 1e-6
    atol: float = 1e3
    max_steps: int = 100_000

    def algebraic(self, t: ArrayLike, y: Array) -> dict[str, ArrayLike]:
        """Everything derivable from `(t, y)` alone, given the run's fixed quantities.

        `M_atm` is not separate integrated state: dM_atm/dt is exactly dot(dy/dt, atomic_masses),
        so M_atm(t) is always exactly dot(y(t), atomic_masses) - it's computed fresh here from y.

        Clips y to >= 0 before use: the adaptive step-size controller can propose trial steps that
        briefly overshoot into slightly negative territory near exhaustion, and a negative
        M_atm/f_atm would otherwise feed a fractional power (convective_envelope_thickness's
        c2**0.59 term) with a negative base.

        Args:
            t: Time [s]
            y: Per-species abundances [atoms], ordered per `parameters.isofate_species`

        Returns:
            Diagnostics keyed by name (`mu`, `x`, `M_atm`, `f_atm`, `radius_env`, `escape_radius`,
            `Vpot`, `A`, `phi`, `phi_c`, `Phi`)
        """
        parameters: Parameters = self.parameters
        system: System = parameters.system
        escape: EscapeMechanism = parameters.escape_mechanism
        escape_number_flux: EscapeNumberFluxBase = parameters.escape_number_flux

        planet_mass: Array = system.planet.mass
        equilibrium_temperature: Array = system.equilibrium_temperature

        y = jnp.maximum(y, 0.0)
        mu: Array = parameters.atmosphere_mean_mu(y)
        x: Array = parameters.atmosphere_atom_fractions(y)

        # y already clipped >= 0 above, so M_atm is too
        atmosphere_mass: Array = parameters.atmosphere_mass(y)
        atmosphere_mass_fraction: Array = parameters.atmosphere_mass_fraction(y)
        _convective_envelope_thickness: Array = convective_envelope_thickness(parameters, y, t)
        _escape_radius: Array = escape_radius(parameters, y, t)

        gravitational_potential: Array = tidal_gravitational_potential(system, _escape_radius)
        area: ArrayLike = sphere_area(_escape_radius)
        grav_acc: ArrayLike = gravitational_acceleration(planet_mass, _escape_radius)

        state: EscapeState = EscapeState(
            system=system,
            escape_radius=_escape_radius,
            gravitational_potential=gravitational_potential,
            mean_mu=mu,
            convective_envelope_thickness=_convective_envelope_thickness,
            atmosphere_mass_fraction=atmosphere_mass_fraction,
            t_current=t,
            t_end=self.t_end,
        )
        phi: ArrayLike = escape.compute_mass_flux(state)
        Phi, phi_c = escape_number_flux.get_number_flux(y, equilibrium_temperature, grav_acc, phi)

        return dict(
            mu=mu,
            x=x,
            M_atm=atmosphere_mass,
            f_atm=atmosphere_mass_fraction,
            radius_env=_convective_envelope_thickness,
            escape_radius=_escape_radius,
            Vpot=gravitational_potential,
            A=area,
            phi=phi,
            phi_c=phi_c,
            Phi=Phi,
        )

    def vector_field(self, t: ArrayLike, y: Array, args) -> Array:
        """RHS of the isocalc ODE system (dy/dt), in the `vf(t, y, args)` form `diffrax.ODETerm`
        expects - `args` is unused.

        Args:
            t: Time [s]
            y: Per-species abundances [atoms]
            args: Unused

        Returns:
            dy/dt [atoms/s]
        """
        del args

        alg: dict = self.algebraic(t, y)

        return -alg["Phi"] * alg["A"]

    def mass_lost(self, t: ArrayLike, y: Array, args: Array, **kwargs) -> Array:
        """Event condition: the atmospheric mass has dropped by
        `parameters.isocalc_options.mass_loss_fraction` of its value at the start of the
        integration.

        With the default fraction, ``1 - EXHAUSTION_FRACTION``, this fires once the atmosphere is
        effectively exhausted (see `EXHAUSTION_FRACTION` for why the threshold is not exactly
        zero). A smaller fraction (e.g. ``0.05``) stops the integration earlier, so that it can be
        restarted (see `integrate_segments`).

        Args:
            t: Time [s]
            y: Per-species abundances [atoms]
            args: Atmospheric mass at the start of the integration [kg]
            **kwargs: Passed by `diffrax.Event`; unused

        Returns:
            True once the given fraction of the atmospheric mass has been lost
        """
        del kwargs

        mass_loss_fraction: float = self.parameters.isocalc_options.mass_loss_fraction

        return self.parameters.atmosphere_mass(y) <= (1 - mass_loss_fraction) * args

    @eqx.filter_jit
    def integrate(
        self, t0_seconds: ArrayLike, t_a: Array, y0: Array, dt0: ArrayLike | None = None
    ) -> tuple[Array, dict[str, Array], IntegrationStop]:
        """Integrates `y` from `t0_seconds` and back-computes the diagnostics on `t_a`.

        The integration stops early if the `mass_lost` event fires: loss of
        `parameters.isocalc_options.mass_loss_fraction` of the atmospheric mass at `t0_seconds`
        (by default, effective exhaustion).

        Jitted so that repeated calls (e.g. across an MCMC/optimization loop, or the segments of
        `isocalc_jax2`) reuse one compiled program instead of dispatching every jnp op
        individually. All three arguments are expected to be actual arrays (callers wrap them in
        `jnp.asarray`) so that varying them across calls reuses the same compiled program rather
        than retracing.

        Args:
            t0_seconds: Integration start time [s]
            t_a: Output times [s]
            y0: Initial per-species abundances [atoms]
            dt0: Initial step size [s]. Defaults to ``None``, meaning diffrax chooses it.

        Returns:
            `(y_a, alg_a, stop)`: the saved trajectory (with the held terminal state past an
            event), the `algebraic` diagnostics for every `t_a` entry (including the derived
            `M_atm`/`f_atm`), and where and why the integration stopped.
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            diffrax.Tsit5(),
            t0=t0_seconds,
            # Not `self.t_end`: t_a's last entry runs one delta_t past that (a
            # pre-existing quirk of t_a's own formula, shared with isocalc, harmless there since
            # t_a is only a diagnostic label in the discrete loop) - diffrax requires saveat.ts to
            # lie within [t0, t1], so t1 is taken directly from t_a instead of re-derived.
            t1=t_a[-1],
            dt0=dt0,
            y0=y0,
            args=self.parameters.atmosphere_mass(y0),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            # Separate sub-saves: the t_a outputs, and the state at the stop time (the event time,
            # if one fired). A single SaveAt(ts=..., t1=True) would not do, because after an event
            # diffrax writes the t1 state into the first unused ts slot rather than the last.
            saveat=diffrax.SaveAt(subs=(diffrax.SubSaveAt(ts=t_a), diffrax.SubSaveAt(t1=True))),
            event=diffrax.Event(cond_fn=self.mass_lost),
            max_steps=self.max_steps,
        )
        ys_out, ys_stop = sol.ys  # pyright: ignore[reportGeneralTypeIssues]
        _, ts_stop = sol.ts  # pyright: ignore[reportGeneralTypeIssues]
        mass_lost: Array = sol.event_mask  # pyright: ignore[reportAssignmentType]
        stop = IntegrationStop(t=ts_stop[0], y=ys_stop[0], mass_lost=mass_lost)

        y_a = _hold_last_finite(ys_out, y0)

        # Back-compute every diagnostic (including the derived M_atm) from the saved trajectory in
        # one vmapped pass, rather than a per-timestep Python loop.
        alg_a = jax.vmap(self.algebraic)(t_a, y_a)

        return y_a, alg_a, stop


class Restart(eqx.Module):
    """A restart of the segmented integration (see `integrate_segments`).

    Args:
        t: Restart time [s]
        y_before: Per-species abundances when the mass-loss event fired [atoms]
        y_after: Per-species abundances after the `on_mass_lost` hook, from which the integration
            restarted [atoms]
    """

    t: Array
    y_before: Array
    y_after: Array


def placeholder_reequilibration(carry: Any, t: Array, y: Array) -> tuple[Any, Array]:
    """Placeholder `on_mass_lost` hook for `integrate_segments`: reduces every abundance by
    `_PLACEHOLDER_REDUCTION`.

    Stands in for the Atmodeller re-equilibration, which will update the atmospheric abundances at
    the fixed time `t` and carry interior state (reservoirs, warm start, ...) between restarts in
    `carry`. The reduction only exists so that the restart is visible (e.g. in tests).

    Args:
        carry: State carried between restarts; returned unchanged
        t: Restart time [s]; unused
        y: Per-species abundances when the mass-loss event fired [atoms]

    Returns:
        `(carry, y_new)`
    """
    # TODO: Replace with the Atmodeller re-equilibration
    del t

    return carry, (1 - _PLACEHOLDER_REDUCTION) * y


def integrate_segments(
    integrator: IsocalcIntegrator,
    t0_seconds: ArrayLike,
    t_a: Array,
    y0: Array,
    on_mass_lost: Callable[[Any, Array, Array], tuple[Any, Array]] = placeholder_reequilibration,
    carry: Any = None,
    max_segments: int = 1000,
) -> tuple[Array, dict[str, ArrayLike], list[Restart], Any]:
    """Integrates in segments, restarting after each mass-loss event.

    diffrax events can only end a solve, so each restart is a new `integrator.integrate` call. With
    `parameters.isocalc_options.mass_loss_fraction` set to e.g. ``0.05``, every segment ends once
    5% of the atmospheric mass at its start has been lost. `on_mass_lost(carry, t, y)` then updates
    `y` at the fixed time `t` (e.g. by re-equilibrating with Atmodeller) and the integration
    restarts from `(t, y_new)`. Only the time and the updated state are passed on: no solver state,
    so the derivative is recomputed after the jump in `y`, and no step-size controller state (it
    only holds error ratios, not the step size).

    Restarts use an explicit initial step, `_RESTART_DT0_FRACTION` of the remaining time span,
    rather than diffrax's automatic choice: at late times (e.g. t ~ 1e16 s) that choice can be
    effectively zero, and since the PID controller scales each new step from the previous one, the
    solve then never advances. The controller adapts the explicit step from there.

    The integration finishes at the last output time, or once the atmospheric mass drops below
    `EXHAUSTION_FRACTION` of its value at `t0_seconds` (checked both when the event fires and after
    the hook).

    Every segment saves on the full output grid, clipped to its own start time, so the shape is
    fixed and `integrate` compiles once. Each segment overwrites the rows from its start time
    onwards, so an output time exactly at a restart holds the post-hook state. The diagnostics are
    computed once over the assembled trajectory; the per-segment diagnostics computed inside
    `integrate` are discarded.

    A Python loop for now, because the hook will call Atmodeller; each `integrate` call is jitted.

    Args:
        integrator: Integrator, reused for every segment
        t0_seconds: Integration start time [s]
        t_a: Output times [s]
        y0: Initial per-species abundances [atoms]
        on_mass_lost: Hook `(carry, t, y) -> (carry, y_new)` called at each restart. Defaults to
            `placeholder_reequilibration`.
        carry: Initial state carried between calls of `on_mass_lost`. Defaults to ``None``.
        max_segments: Maximum number of segments. Defaults to ``1000``.

    Returns:
        `(y_a, alg_a, restarts, carry)`: the trajectory on `t_a`, the `algebraic` diagnostics for
        every `t_a` entry, one `Restart` per restart, and the final carried state

    Raises:
        RuntimeError: If the integration has not finished after `max_segments` segments
    """
    parameters: Parameters = integrator.parameters
    floor: Array = EXHAUSTION_FRACTION * parameters.atmosphere_mass(y0)

    t_start: Array = jnp.asarray(t0_seconds)
    y: Array = y0
    y_a: Array = jnp.zeros((t_a.shape[0], y0.shape[0]))
    restarts: list[Restart] = []

    for _ in range(max_segments):
        # diffrax chooses the first segment's initial step, as in a single `integrate` call
        dt0: Array | None = None if not restarts else _RESTART_DT0_FRACTION * (t_a[-1] - t_start)
        y_segment, _, stop = integrator.integrate(t_start, jnp.maximum(t_a, t_start), y, dt0)
        y_a = jnp.where((t_a >= t_start)[:, None], y_segment, y_a)

        if not bool(stop.mass_lost) or bool(parameters.atmosphere_mass(stop.y) <= floor):
            break

        carry, y = on_mass_lost(carry, stop.t, stop.y)
        restarts.append(Restart(t=stop.t, y_before=stop.y, y_after=y))

        if bool(parameters.atmosphere_mass(y) <= floor):
            y_a = jnp.where((t_a > stop.t)[:, None], y, y_a)
            break

        t_start = stop.t
    else:
        raise RuntimeError(f"Segmented integration did not finish within {max_segments} segments")

    alg_a: dict[str, ArrayLike] = jax.vmap(integrator.algebraic)(t_a, y_a)

    return y_a, alg_a, restarts, carry

