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

_INITIAL_DT0_FRACTION: float = 1e-6
"""Initial step of each integration, as a fraction of its time span.

diffrax's automatic choice can be effectively zero at late times (e.g. t ~ 1e16 s), which stalls
the solve, because the PID controller scales each new step from the previous one. The controller
adapts this step from there, and diffrax clips it to the end of the span.
"""


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

    @classmethod
    def from_solution(cls, sol: diffrax.Solution) -> "IntegrationStop":
        """Extracts the stop information from the solution returned by
        `IsocalcIntegrator.integrate`.

        Args:
            sol: Solution returned by `IsocalcIntegrator.integrate`

        Returns:
            Where and why the integration stopped
        """
        _, ys_stop = sol.ys  # pyright: ignore[reportGeneralTypeIssues]
        _, ts_stop = sol.ts  # pyright: ignore[reportGeneralTypeIssues]

        return cls(t=ts_stop[0], y=ys_stop[0], mass_lost=sol.event_mask)  # pyright: ignore


class AtmosphereState(eqx.Module):
    """Everything derivable from `(t, y)` alone, returned by `IsocalcIntegrator.algebraic`.

    `IsocalcIntegrator.diagnostics` returns the same module with every field batched over the
    output times.

    Args:
        mean_mu: Mean atmospheric particle mass [kg]
        atom_fractions: Atom fraction of each species [ndim]
        atmosphere_mass: Atmospheric mass [kg]
        atmosphere_mass_fraction: Atmospheric mass fraction [ndim]
        convective_envelope_thickness: Radial thickness of the convective envelope [m]
        escape_radius: Effective outer radius from which the atmosphere escapes [m]
        gravitational_potential: Gravitational potential at the escape radius, reduced by stellar
            tidal forces [J/kg]
        area: Area of the sphere at the escape radius [m2]
        mass_flux: Bulk escaping mass flux from the escape mechanism [kg/m2/s]
        critical_mass_flux: Bulk mass flux above which the heavier dominant species also escapes
            [kg/m2/s]
        number_flux: Escaping number flux of each species [atoms/m2/s]
    """

    mean_mu: ArrayLike
    atom_fractions: ArrayLike
    atmosphere_mass: ArrayLike
    atmosphere_mass_fraction: ArrayLike
    convective_envelope_thickness: ArrayLike
    escape_radius: ArrayLike
    gravitational_potential: ArrayLike
    area: ArrayLike
    mass_flux: ArrayLike
    critical_mass_flux: ArrayLike
    number_flux: ArrayLike


class IsocalcIntegrator(eqx.Module):
    """Time integration of the atmospheric-escape ODE for one isocalc run.

    The only integrated state is `y`, the per-species atmospheric abundances [atoms]

    Args:
        parameters: Simulation parameters
        rtol: Relative tolerance of the step-size controller. Defaults to ``1e-6``.
        atol: Absolute tolerance of the step-size controller [atoms]. Defaults to ``1e3``.
        max_steps: Maximum number of solver steps. Defaults to ``100_000``.
    """

    parameters: Parameters
    rtol: float = 1e-6
    atol: float = 1e3
    max_steps: int = 100_000

    def algebraic(self, t: ArrayLike, y: Array) -> AtmosphereState:
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
            The atmosphere state at `(t, y)`
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
        )
        # Bulk escaping mass flux [kg/m2/s] from the escape mechanism, partitioned into per-species
        # number fluxes [atoms/m2/s] by diffusion-limited fractionation; the critical mass flux
        # [kg/m2/s] is the bulk flux above which the heavier dominant species also escapes
        mass_flux: ArrayLike = escape.compute_mass_flux(state)
        number_flux, critical_mass_flux = escape_number_flux.get_number_flux(
            y, equilibrium_temperature, grav_acc, mass_flux
        )

        return AtmosphereState(
            mean_mu=mu,
            atom_fractions=x,
            atmosphere_mass=atmosphere_mass,
            atmosphere_mass_fraction=atmosphere_mass_fraction,
            convective_envelope_thickness=_convective_envelope_thickness,
            escape_radius=_escape_radius,
            gravitational_potential=gravitational_potential,
            area=area,
            mass_flux=mass_flux,
            critical_mass_flux=critical_mass_flux,
            number_flux=number_flux,
        )

    def vector_field(self, t: ArrayLike, y: Array, args) -> ArrayLike:
        """RHS of the isocalc ODE system (dy/dt), in the `vf(t, y, args)` form `diffrax.ODETerm`
        expects.

        Args:
            t: Time [s]
            y: Per-species abundances [atoms]
            args: Unused

        Returns:
            dy/dt [atoms/s]
        """
        del args

        atmosphere: AtmosphereState = self.algebraic(t, y)

        return -1 * atmosphere.number_flux * atmosphere.area

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
        del t
        del kwargs

        mass_loss_fraction: float = self.parameters.isocalc_options.mass_loss_fraction

        return self.parameters.atmosphere_mass(y) <= (1 - mass_loss_fraction) * args

    @eqx.filter_jit
    def integrate(self, t_start: float | Array, t_a: Array, y0: Array) -> diffrax.Solution:
        """Integrates `y` from `t_start`, saving on `t_a`.

        The integration stops early if the `mass_lost` event fires: loss of
        `parameters.isocalc_options.mass_loss_fraction` of the atmospheric mass at `t_start`
        (by default, effective exhaustion). The initial step is `_INITIAL_DT0_FRACTION` of the
        time span, rather than diffrax's automatic choice, which can stall at late times.

        Args:
            t_start: Integration start time [s]
            t_a: Output times [s]
            y0: Initial per-species abundances [atoms]

        Returns:
            The diffrax solution. `sol.ys` and `sol.ts` are pairs from two sub-saves: the
            trajectory on `t_a` (rows after the event, if it fired, are ``inf``, as diffrax leaves
            them) and the single state at the stop time. `sol.event_mask` is True if the event
            fired. See `IntegrationStop.from_solution` and `diagnostics`.
        """
        return diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            diffrax.Tsit5(),
            t0=t_start,
            # diffrax requires saveat.ts to lie within [t0, t1], so t1 is taken directly from t_a
            # (whose last entry runs one delta_t past the run's end time - a pre-existing quirk of
            # t_a's own formula, shared with isocalc).
            t1=t_a[-1],
            dt0=_INITIAL_DT0_FRACTION * (t_a[-1] - t_start),
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

    @eqx.filter_jit
    def diagnostics(self, t_a: Array, y_a: Array) -> AtmosphereState:
        """`algebraic` diagnostics for every entry of a saved trajectory.

        Back-computes every diagnostic (including the derived M_atm) in one vmapped pass, rather
        than a per-timestep Python loop.

        Args:
            t_a: Output times [s]
            y_a: Per-species abundances at each output time [atoms]

        Returns:
            The atmosphere state (see `algebraic`), with every field batched over the output times
        """
        return jax.vmap(self.algebraic)(t_a, y_a)


class Restarts(eqx.Module):
    """The restarts of a segmented integration (see `integrate_segments`), in fixed-size buffers
    so that the integration can be jitted.

    Only the first `count` entries are filled; the remaining entries are NaN.

    Args:
        t: Restart times [s]
        y_before: Per-species abundances when each mass-loss event fired [atoms]
        y_after: Per-species abundances after the `on_mass_lost` hook, from which the integration
            restarted [atoms]
        count: Number of restarts
        finished: True if the integration finished within `max_segments` segments
    """

    t: Array
    y_before: Array
    y_after: Array
    count: Array
    finished: Array


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


@eqx.filter_jit
def integrate_segments(
    integrator: IsocalcIntegrator,
    t_start: float | Array,
    t_a: Array,
    y0: Array,
    on_mass_lost: Callable[[Any, Array, Array], tuple[Any, Array]] = placeholder_reequilibration,
    carry: Any = None,
    max_segments: int = 1000,
) -> tuple[Array, AtmosphereState, Restarts, Any]:
    """Integrates in segments, restarting after each mass-loss event.

    diffrax events can only end a solve, so each restart is a new `integrator.integrate` call. With
    `parameters.isocalc_options.mass_loss_fraction` set to e.g. ``0.05``, every segment ends once
    5% of the atmospheric mass at its start has been lost. `on_mass_lost(carry, t, y)` then updates
    `y` at the fixed time `t` (e.g. by re-equilibrating with Atmodeller) and the integration
    restarts from `(t, y_new)`. Only the time and the updated state are passed on: no solver state,
    so the derivative is recomputed after the jump in `y`, and no step-size controller state (it
    only holds error ratios, not the step size).

    Every segment, including restarts, uses `integrate`'s explicit initial step
    (`_INITIAL_DT0_FRACTION` of its time span), so restarts at late times don't stall.

    The integration finishes at the last output time, or once the atmospheric mass drops below
    `EXHAUSTION_FRACTION` of its value at `t_start` (checked both when the event fires and after
    the hook).

    Every segment saves on the full output grid, clipped to its own start time, so the shapes are
    fixed. Each segment overwrites the rows from its start time onwards, so an output time exactly
    at a restart holds the post-hook state. The rows after a segment's event, which `integrate`
    leaves as ``inf``, are therefore overwritten by the next segment (`jnp.where` passes a zero
    gradient through the unselected ``inf`` constants); only a final segment that ended on
    exhaustion, or a run that ran out of `max_segments`, keeps ``inf`` rows after its stop. The
    diagnostics are computed once over the assembled trajectory; the per-segment diagnostics
    computed inside `integrate` are discarded.

    Jitted, so `on_mass_lost` must be jittable and must return `carry` with the same structure,
    shapes and dtypes it received. Following the pattern of Atmodeller's retry solver, the first
    segment runs before a `jax.lax.while_loop` over the restarts, and each restart is recorded in
    fixed-size buffers. `max_segments` sets the buffer size, so a different value retraces.

    Args:
        integrator: Integrator, reused for every segment
        t_start: Integration start time [s]
        t_a: Output times [s]
        y0: Initial per-species abundances [atoms]
        on_mass_lost: Jittable hook `(carry, t, y) -> (carry, y_new)` called at each restart.
            Defaults to `placeholder_reequilibration`.
        carry: Initial state carried between calls of `on_mass_lost`. Defaults to ``None``.
        max_segments: Maximum number of segments. Defaults to ``1000``.

    Returns:
        `(y_a, alg_a, restarts, carry)`: the trajectory on `t_a`, the `algebraic` diagnostics for
        every `t_a` entry, the restarts, and the final carried state

    Raises:
        EquinoxRuntimeError: If the integration has not finished after `max_segments` segments,
            but only if Equinox errors are enabled (`EQX_ON_ERROR`, which isofate turns off by
            default) - otherwise check `restarts.finished`
    """
    parameters: Parameters = integrator.parameters
    t_start = jnp.asarray(t_start)
    t_last: Array = t_a[-1]
    floor: Array = EXHAUSTION_FRACTION * parameters.atmosphere_mass(y0)
    max_restarts: int = max_segments - 1

    def needs_restart(stop: IntegrationStop) -> Array:
        """True if the segment ended on a mass-loss event before exhaustion and the last output
        time."""
        return stop.mass_lost & (parameters.atmosphere_mass(stop.y) > floor) & (stop.t < t_last)

    # First segment, before the loop
    sol: diffrax.Solution = integrator.integrate(t_start, jnp.maximum(t_a, t_start), y0)
    y_segment: Array = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
    stop: IntegrationStop = IntegrationStop.from_solution(sol)
    nan_buffer: Array = jnp.full((max_restarts, y0.shape[0]), jnp.nan)
    state = (
        y_segment,  # y_a
        stop,
        needs_restart(stop),
        carry,
        Restarts(
            t=jnp.full(max_restarts, jnp.nan),
            y_before=nan_buffer,
            y_after=nan_buffer,
            count=jnp.array(0),
            finished=jnp.array(False),
        ),
    )

    def cond_fn(state) -> Array:
        _, _, restart, _, restarts = state

        return restart & (restarts.count < max_restarts)

    def body_fn(state):
        y_a, stop, _, carry, restarts = state

        carry, y_new = on_mass_lost(carry, stop.t, stop.y)
        index: Array = restarts.count
        restarts = Restarts(
            t=restarts.t.at[index].set(stop.t),
            y_before=restarts.y_before.at[index].set(stop.y),
            y_after=restarts.y_after.at[index].set(y_new),
            count=index + 1,
            finished=restarts.finished,
        )

        def finish(y_a: Array) -> tuple[Array, IntegrationStop, Array]:
            """The hook left the atmosphere exhausted: hold its output for the remaining rows."""
            y_a = jnp.where((t_a > stop.t)[:, None], y_new, y_a)
            exhausted_stop = IntegrationStop(t=stop.t, y=y_new, mass_lost=jnp.array(False))

            return y_a, exhausted_stop, jnp.array(False)

        def restart(y_a: Array) -> tuple[Array, IntegrationStop, Array]:
            """Integrate the next segment from the hook's output."""
            sol = integrator.integrate(stop.t, jnp.maximum(t_a, stop.t), y_new)
            y_segment: Array = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
            next_stop = IntegrationStop.from_solution(sol)
            y_a = jnp.where((t_a >= stop.t)[:, None], y_segment, y_a)

            return y_a, next_stop, needs_restart(next_stop)

        y_a, stop, needs_another = jax.lax.cond(
            parameters.atmosphere_mass(y_new) <= floor, finish, restart, y_a
        )

        return y_a, stop, needs_another, carry, restarts

    y_a, _, unfinished, carry, restarts = jax.lax.while_loop(cond_fn, body_fn, state)
    restarts = eqx.tree_at(lambda r: r.finished, restarts, jnp.invert(unfinished))
    # Only raises if Equinox errors are enabled (isofate/__init__.py sets EQX_ON_ERROR="off"), so
    # callers should also check `restarts.finished`
    y_a = eqx.error_if(
        y_a, unfinished, f"Segmented integration did not finish within {max_segments} segments"
    )

    alg_a: AtmosphereState = integrator.diagnostics(t_a, y_a)

    return y_a, alg_a, restarts, carry
