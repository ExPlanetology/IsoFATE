# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The IsoFATE driver, `isocalc`: atmospheric escape coupled to interior-atmosphere
equilibrium."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from isofate.atmodeller_coupler import AtmodellerCoupler, CouplerState
from isofate.constants import const
from isofate.integrators import (
    EXHAUSTION_FRACTION,
    AdaptiveIntegrator,
    EulerIntegrator,
    IsocalcModel,
    no_reequilibration,
)
from isofate.output import Equilibrations, Output
from isofate.parameters import Parameters
from isofate.species import SYMBOLS


def isocalc(parameters: Parameters, t_end=5e9) -> Output:
    """Atmospheric escape with event-triggered Atmodeller coupling.

    The integration runs through `isofate.integrators.AdaptiveIntegrator`, which stops on an
    event, re-equilibrates the atmosphere and interior with Atmodeller
    (`AtmodellerCoupler.reequilibrate`), and restarts. The atmosphere and interior are first
    equilibrated once at the start time.

    The events are:
        - Elapsed time: every `IsocalcOptions.n_atmodeller` output steps, i.e. at exactly
          ``t_start + k * n_atmodeller * delta_t``. `n_atmodeller = 0` switches the coupling off.
        - Mass loss: `IsocalcOptions.mass_loss_fraction` of the atmospheric mass lost since the
          last re-equilibration (by default only at exhaustion), and optionally
          `IsocalcOptions.species_loss_fraction` of any species.

    Output rows after the atmosphere is exhausted are ``inf``, as `AdaptiveIntegrator` leaves them
    (with `IsocalcOptions.euler` the atmosphere is instead frozen once exhausted). An empty initial
    atmosphere is not equilibrated.

    With `IsocalcOptions.euler` this is the scheme of the former NumPy loop driver (retired), a
    manual fixed time step re-equilibrating every `n_atmodeller` steps, which it reproduced to
    round-off.

    Args:
        parameters: Parameters
        t_end: Simulation end time, i.e. system age at the end of the run [yr]. Defaults to
            ``5e9``.

    Returns:
        The output (see `isofate.output.Output`): the atmospheric abundances on the output times,
        the record of the equilibrations, and the complete Atmodeller output for them

    Raises:
        RuntimeError: If an Atmodeller solve fails to converge
    """
    options = parameters.isocalc_options

    n_tot: int = options.n_steps
    t_start: Array = jnp.asarray(options.t_start / const.s2yr)  # start time [s]
    delta_t = (t_end / const.s2yr - options.t_start / const.s2yr) / n_tot  # time step [s]
    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a: Array = jnp.asarray(options.t_start / const.s2yr + delta_t * np.arange(1, n_tot + 1))
    y_initial: Array = jnp.asarray(parameters.initial_abundances(), dtype=float)

    model = IsocalcModel(parameters)
    integrator_class = EulerIntegrator if options.euler else AdaptiveIntegrator

    if options.n_atmodeller == 0:
        integrator = integrator_class(model, on_mass_lost=no_reequilibration)
        y_a, _, _, _ = integrator.integrate(t_start, t_a, y_initial)
        return Output(parameters, t_start, t_a, y_a)

    # Built once per run, outside jit (it builds the Atmodeller model)
    coupler = AtmodellerCoupler(parameters)
    rows: list[dict[str, Array]] = []  # one per equilibration, see _equilibrations

    # Initial equilibration (none for an empty atmosphere)
    state: CouplerState = coupler.initial_state()
    y0: Array = y_initial
    if float(parameters.atmosphere_mass(y_initial)) > 0:
        state, y0 = coupler.reequilibrate(state, t_start, y_initial)
        rows.append(_equilibration_row(t_start, y_initial, y0, state))

    # Integrate, re-equilibrating every n_atmodeller output steps
    integrator = integrator_class(
        model, on_mass_lost=coupler.reequilibrate, reequilibration_steps=options.n_atmodeller
    )
    y_a, _, restarts, state = integrator.integrate(t_start, t_a, y0, carry=state)
    for i in range(int(restarts.count)):
        carry = jax.tree.map(lambda leaf: leaf[i], restarts.carry)
        rows.append(
            _equilibration_row(restarts.t[i], restarts.y_before[i], restarts.y_after[i], carry)
        )

    # Re-equilibrate once more at the end time when it is on the schedule (n_steps a multiple of
    # n_atmodeller) - the integrators never re-equilibrate at the last output time. The last row
    # then holds the re-equilibrated state.
    floor = EXHAUSTION_FRACTION * float(parameters.atmosphere_mass(y0))
    if (
        n_tot % options.n_atmodeller == 0
        and bool(jnp.all(jnp.isfinite(y_a[-1])))
        and float(parameters.atmosphere_mass(y_a[-1])) > floor
    ):
        y_before: Array = y_a[-1]
        state, y_end = coupler.reequilibrate(state, t_a[-1], y_before)
        y_a = y_a.at[-1].set(y_end)
        rows.append(_equilibration_row(t_a[-1], y_before, y_end, state))

    # Stop if any equilibration (initial, in the loop, or final) failed to converge
    coupler.check_converged(state)

    equilibrations: Equilibrations = _equilibrations(rows, capacity=restarts.t.shape[0] + 2)
    atmodeller = None
    if rows:
        atmodeller = coupler.batched_output(
            equilibrations.surface_temperature_atmodeller[: len(rows)],
            equilibrations.mantle_melt_fraction[: len(rows)],
            equilibrations.mass_constraints[: len(rows)],
            equilibrations.solution[: len(rows)],
        )

    return Output(parameters, t_start, t_a, y_a, equilibrations, atmodeller)


def _equilibration_row(t: ArrayLike, y_before: Array, y_after: Array, state: CouplerState) -> dict:
    """One equilibration's entry for `Equilibrations`, from the coupler state after it."""
    return {
        "t": jnp.asarray(t, dtype=float),
        "y_before": jnp.asarray(y_before),
        "y_after": jnp.asarray(y_after),
        "y_int": state.y_int,
        "surface_temperature": state.surface_temperature,
        "surface_temperature_atmodeller": state.surface_temperature_atmodeller,
        "mantle_melt_fraction": state.mantle_melt_fraction,
        "mass_constraints": state.mass_constraints,
        # one row of the solution array (the coupler's model is a single problem)
        "solution": jnp.reshape(state.solution, -1),
    }


def _equilibrations(rows: list[dict[str, Array]], capacity: int) -> Equilibrations:
    """`Equilibrations` from the per-equilibration rows, padded with NaN to `capacity` rows."""
    template: dict[str, Array] = rows[0] if rows else _empty_row()

    def column(name: str) -> Array:
        filled = (
            jnp.stack([row[name] for row in rows])
            if rows
            else jnp.zeros((0,) + template[name].shape)
        )
        padding = jnp.full((capacity - len(rows),) + template[name].shape, jnp.nan)
        return jnp.concatenate([filled, padding])

    return Equilibrations(**{name: column(name) for name in template}, count=jnp.array(len(rows)))


def _empty_row() -> dict[str, Array]:
    """Shapes of an equilibration row, for a run without equilibrations."""
    n = len(SYMBOLS)
    return {
        "t": jnp.zeros(()),
        "y_before": jnp.zeros(n),
        "y_after": jnp.zeros(n),
        "y_int": jnp.zeros(n),
        "surface_temperature": jnp.zeros(()),
        "surface_temperature_atmodeller": jnp.zeros(()),
        "mantle_melt_fraction": jnp.zeros(()),
        "mass_constraints": jnp.zeros(n - 1),
        "solution": jnp.zeros(0),
    }
