# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time integration of the atmospheric-escape ODE for the isocalc drivers.

- `IsocalcIntegrator`: the ODE right-hand side (`vector_field`), its diagnostics (`algebraic`,
  `diagnostics`), and one adaptive diffrax solve (`integrate`) that stops on a mass-loss or
  elapsed-time event.
- `integrate_segments`: adaptive, event-driven integration in segments, re-equilibrating through a
  hook (e.g. Atmodeller) after each event.
- `integrate_segments_euler`: isocalc's fixed-step forward Euler march with a re-equilibration
  every n steps, using the same hook and return values.

The planet-radius and escape physics used by the right-hand side is in `isofate.engine`.
"""

from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jaxtyping import ArrayLike

from isofate.engine import (
    convective_envelope_thickness,
    escape_radius,
    tidal_gravitational_potential,
)
from isofate.escape.fractionation import EscapeNumberFluxBase
from isofate.escape.mechanisms import EscapeMechanism, EscapeState
from isofate.parameters import Parameters
from isofate.system import System
from isofate.utils import gravitational_acceleration, sphere_area

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


class IntegrationStop(eqx.Module):
    """Where and why an `IsocalcIntegrator.integrate` call stopped.

    Args:
        t: Stop time [s] - the end of the requested output times if no event fired
        y: Per-species abundances at the stop time [atoms]
        mass_lost: True if the mass-loss event fired (see `IsocalcIntegrator.mass_lost`)
        time_elapsed: True if the time event fired (see `IsocalcIntegrator.time_elapsed`)
    """

    t: Array
    y: Array
    mass_lost: Array
    time_elapsed: Array

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

        mass_lost, time_elapsed = sol.event_mask  # pyright: ignore[reportGeneralTypeIssues]

        return cls(t=ts_stop[0], y=ys_stop[0], mass_lost=mass_lost, time_elapsed=time_elapsed)


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
        reequilibration_interval: If set, the integration also stops once this much time has
            passed since its start (see `time_elapsed`), e.g. to re-equilibrate with Atmodeller at
            a fixed cadence like `isofate_coupler.isocalc` [s]. Defaults to ``None`` (off).
    """

    parameters: Parameters
    rtol: float = 1e-6
    atol: float = 1e3
    max_steps: int = 100_000
    reequilibration_interval: float | None = None

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
        integration, or (if `species_loss_fraction` is set) any species has dropped by that
        fraction of its own starting abundance.

        With the default mass fraction, ``1 - EXHAUSTION_FRACTION``, this fires once the
        atmosphere is effectively exhausted (see `EXHAUSTION_FRACTION` for why the threshold is not
        exactly zero). A smaller fraction (e.g. ``0.05``) stops the integration earlier, so that it
        can be restarted (see `integrate_segments`). The per-species condition catches trace
        species drained by escape while the total mass hardly changes; species absent or below an
        atom fraction of `EXHAUSTION_FRACTION` at the start are ignored, so that a vanishing trace
        species can't trigger an endless chain of restarts.

        Args:
            t: Time [s]
            y: Per-species abundances [atoms]
            args: `(y0, t_trigger)` - per-species abundances at the start of the integration
                [atoms], and the time event's trigger time (unused here)
            **kwargs: Passed by `diffrax.Event`; unused

        Returns:
            True once the given fraction has been lost
        """
        del t
        del kwargs

        options = self.parameters.isocalc_options
        y0, _ = args
        lost: Array = self.parameters.atmosphere_mass(y) <= (
            1 - options.mass_loss_fraction
        ) * self.parameters.atmosphere_mass(y0)

        if options.species_loss_fraction is not None:
            tracked: Array = y0 > EXHAUSTION_FRACTION * self.parameters.atmosphere_atoms(y0)
            species_lost: Array = tracked & (y <= (1 - options.species_loss_fraction) * y0)
            lost = lost | jnp.any(species_lost)

        return lost

    def time_elapsed(self, t: ArrayLike, y: Array, args: tuple[Array, Array], **kwargs) -> Array:
        """Event condition: `reequilibration_interval` has passed since the start of the
        integration.

        Real-valued (``t - t_trigger``), so that diffrax's root finder stops the integration
        exactly at the trigger time rather than at the end of the step that crosses it. The trigger
        time is ``inf`` when `reequilibration_interval` is ``None``, so it never fires.

        Args:
            t: Time [s]
            y: Per-species abundances [atoms]; unused
            args: `(y0, t_trigger)` - per-species abundances at the start of the integration
                (unused here) and the trigger time [s]
            **kwargs: Passed by `diffrax.Event`; unused

        Returns:
            ``t - t_trigger`` [s], which changes sign at the trigger time
        """
        del y
        del kwargs

        _, t_trigger = args

        return t - t_trigger

    @eqx.filter_jit
    def integrate(self, t_start: float | Array, t_a: Array, y0: Array) -> diffrax.Solution:
        """Integrates `y` from `t_start`, saving on `t_a`.

        The integration stops early if an event fires: `mass_lost` (loss of
        `parameters.isocalc_options.mass_loss_fraction` of the atmospheric mass at `t_start` - by
        default, effective exhaustion) or `time_elapsed` (`reequilibration_interval` after
        `t_start`, if set). The initial step is `_INITIAL_DT0_FRACTION` of the
        time span, rather than diffrax's automatic choice, which can stall at late times.

        Args:
            t_start: Integration start time [s]
            t_a: Output times [s]
            y0: Initial per-species abundances [atoms]

        Returns:
            The diffrax solution. `sol.ys` and `sol.ts` are pairs from two sub-saves: the
            trajectory on `t_a` (rows after the event, if it fired, are ``inf``, as diffrax leaves
            them) and the single state at the stop time. `sol.event_mask` is the pair
            `(mass_lost, time_elapsed)`, True for the event that fired. See
            `IntegrationStop.from_solution` and `diagnostics`.
        """
        interval: float = (
            jnp.inf if self.reequilibration_interval is None else self.reequilibration_interval
        )
        t_trigger: Array = jnp.asarray(t_start) + interval

        return diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            diffrax.Tsit5(),
            t0=t_start,
            # The last output time (the run's end time); diffrax requires saveat.ts to lie within
            # [t0, t1]
            t1=t_a[-1],
            dt0=_INITIAL_DT0_FRACTION * (t_a[-1] - t_start),
            y0=y0,
            args=(y0, t_trigger),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            # Separate sub-saves: the t_a outputs, and the state at the stop time (the event time,
            # if one fired). A single SaveAt(ts=..., t1=True) would not do, because after an event
            # diffrax writes the t1 state into the first unused ts slot rather than the last.
            saveat=diffrax.SaveAt(subs=(diffrax.SubSaveAt(ts=t_a), diffrax.SubSaveAt(t1=True))),
            # The real-valued time event is located exactly by the root finder; the boolean
            # mass event fires at the end of the step on which it becomes true
            event=diffrax.Event(
                cond_fn=(self.mass_lost, self.time_elapsed),
                root_finder=optx.Newton(rtol=1e-12, atol=1.0),
            ),
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
        carry: The `on_mass_lost` hook's carried state after each restart, with the same pytree
            structure as the carry and a leading axis per restart (beyond `count`, floating leaves
            are NaN-filled and others zero-filled)
    """

    t: Array
    y_before: Array
    y_after: Array
    count: Array
    finished: Array
    carry: Any


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
    # Restart times can coincide with output times (e.g. a time event every n output steps), but
    # only to the last bit, so rows are matched to a restart within half an output spacing: a row
    # at the restart time takes the post-hook state
    half_gap: Array = 0.5 * jnp.min(jnp.diff(t_a))

    def needs_restart(stop: IntegrationStop) -> Array:
        """True if the segment ended on an event (mass loss or elapsed time) before exhaustion
        and the last output time."""
        return (
            (stop.mass_lost | stop.time_elapsed)
            & (parameters.atmosphere_mass(stop.y) > floor)
            & (stop.t < t_last - half_gap)
        )

    # First segment, before the loop
    sol: diffrax.Solution = integrator.integrate(t_start, t_a, y0)
    y_segment: Array = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
    stop: IntegrationStop = IntegrationStop.from_solution(sol)
    nan_buffer: Array = jnp.full((max_restarts, y0.shape[0]), jnp.nan)

    def carry_buffer(leaf) -> Array:
        """One buffer per carry leaf, with a leading axis per restart."""
        leaf = jnp.asarray(leaf)
        fill = jnp.nan if jnp.issubdtype(leaf.dtype, jnp.floating) else 0

        return jnp.full((max_restarts, *leaf.shape), fill, dtype=leaf.dtype)

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
            carry=jax.tree.map(carry_buffer, carry),
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
            carry=jax.tree.map(
                lambda buffer, leaf: buffer.at[index].set(leaf), restarts.carry, carry
            ),
        )

        def finish(y_a: Array) -> tuple[Array, IntegrationStop, Array]:
            """The hook left the atmosphere exhausted: hold its output for the remaining rows."""
            y_a = jnp.where((t_a >= stop.t - half_gap)[:, None], y_new, y_a)
            exhausted_stop = IntegrationStop(
                t=stop.t, y=y_new, mass_lost=jnp.array(False), time_elapsed=jnp.array(False)
            )

            return y_a, exhausted_stop, jnp.array(False)

        def restart(y_a: Array) -> tuple[Array, IntegrationStop, Array]:
            """Integrate the next segment from the hook's output."""
            sol = integrator.integrate(stop.t, jnp.maximum(t_a, stop.t), y_new)
            y_segment: Array = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
            next_stop = IntegrationStop.from_solution(sol)
            y_a = jnp.where((t_a >= stop.t - half_gap)[:, None], y_segment, y_a)

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


@eqx.filter_jit
def integrate_segments_euler(
    integrator: IsocalcIntegrator,
    t_start: float | Array,
    t_a: Array,
    y0: Array,
    steps_per_segment: int,
    on_mass_lost: Callable[[Any, Array, Array], tuple[Any, Array]] = placeholder_reequilibration,
    carry: Any = None,
) -> tuple[Array, AtmosphereState, Restarts, Any]:
    """Fixed-step forward Euler march with a re-equilibration every `steps_per_segment` steps -
    the same scheme as `isofate_coupler.isocalc`, as an alternative to the adaptive, event-driven
    `integrate_segments` (same hook and return values).

    The output times must be the uniform grid ``t_a[j] = t_start + (j + 1) * delta_t``; row j holds
    the state after j + 1 steps. Each step is ``y <- max(y + delta_t * f(t, y), 0)`` with
    `integrator.vector_field` as f, clipped at zero after every step as in isocalc (diffrax's
    Euler solver offers no per-step clipping, hence the explicit step). After every
    `steps_per_segment` steps, `on_mass_lost(carry, t, y)` re-equilibrates the state, and the row
    at that time holds the re-equilibrated state. As in `integrate_segments`, the initial
    equilibration (before `y0`) and one at the last output time are left to the caller.

    Once the atmospheric mass drops below `EXHAUSTION_FRACTION` of its value in `y0`, the state is
    frozen and no further re-equilibrations happen (isocalc instead runs on until the abundances are
    exactly zero).

    Jitted with `jax.lax.scan` over segments and steps. `steps_per_segment` and the hook are
    static (a different value retraces); the hook must be jittable and return `carry` with the same
    structure, shapes and dtypes it received.

    Args:
        integrator: Integrator, whose `vector_field` and `diagnostics` are used
        t_start: Start time [s]
        t_a: Output times [s], ``t_start + (j + 1) * delta_t`` for j = 0, ..., n_steps - 1
        y0: Initial per-species abundances [atoms]
        steps_per_segment: Number of Euler steps between re-equilibrations
        on_mass_lost: Jittable hook `(carry, t, y) -> (carry, y_new)` called at each
            re-equilibration. Defaults to `placeholder_reequilibration`.
        carry: Initial state carried between calls of `on_mass_lost`. Defaults to ``None``.

    Returns:
        `(y_a, alg_a, restarts, carry)`, as for `integrate_segments`
    """
    parameters: Parameters = integrator.parameters
    t_start = jnp.asarray(t_start)
    n_steps: int = t_a.shape[0]
    delta_t: Array = (t_a[-1] - t_start) / n_steps
    floor: Array = EXHAUSTION_FRACTION * parameters.atmosphere_mass(y0)
    n_segments, remainder = divmod(n_steps, steps_per_segment)

    def euler_step(y: Array, n: Array) -> tuple[Array, Array]:
        """One forward Euler step from state n (time t_start + n * delta_t) to state n + 1."""
        dy_dt: Array = integrator.vector_field(t_start + n * delta_t, y, None)
        y_next: Array = jnp.maximum(y + delta_t * dy_dt, 0.0)
        # frozen once exhausted
        y_next = jnp.where(parameters.atmosphere_mass(y) <= floor, y, y_next)

        return y_next, y_next

    def segment(state, k: Array):
        """`steps_per_segment` Euler steps, then a re-equilibration at the end of the segment
        unless it ends at the last output time or the atmosphere is exhausted."""
        y, carry = state
        n0 = k * steps_per_segment
        y_end, ys = jax.lax.scan(euler_step, y, n0 + jnp.arange(steps_per_segment))
        t_end = t_start + (n0 + steps_per_segment) * delta_t
        is_last = (k == n_segments - 1) & (remainder == 0)
        hooked = jnp.invert(is_last) & (parameters.atmosphere_mass(y_end) > floor)
        carry, y_new = jax.lax.cond(
            hooked,
            lambda c, y_: on_mass_lost(c, t_end, y_),
            lambda c, y_: (c, y_),
            carry,
            y_end,
        )
        ys = ys.at[-1].set(y_new)  # the row at the re-equilibration holds the new state

        return (y_new, carry), (ys, t_end, y_end, y_new, carry, hooked)

    (y, carry), (ys, restart_t, y_before, y_after, carries, hooked) = jax.lax.scan(
        segment, (y0, carry), jnp.arange(n_segments)
    )
    y_a: Array = ys.reshape(n_segments * steps_per_segment, y0.shape[0])
    if remainder:
        steps_rest: Array = n_segments * steps_per_segment + jnp.arange(remainder)
        _, ys_rest = jax.lax.scan(euler_step, y, steps_rest)
        y_a = jnp.concatenate([y_a, ys_rest])

    # The re-equilibrations happen at a prefix of the segment ends (until exhaustion or the end)
    def masked(values: Array) -> Array:
        values = jnp.asarray(values)
        mask = hooked.reshape(hooked.shape + (1,) * (values.ndim - 1))
        fill = jnp.nan if jnp.issubdtype(values.dtype, jnp.floating) else 0

        return jnp.where(mask, values, fill)

    restarts = Restarts(
        t=masked(restart_t),
        y_before=masked(y_before),
        y_after=masked(y_after),
        count=jnp.sum(hooked),
        finished=jnp.array(True),
        carry=jax.tree.map(masked, carries),
    )
    alg_a: AtmosphereState = integrator.diagnostics(t_a, y_a)

    return y_a, alg_a, restarts, carry
