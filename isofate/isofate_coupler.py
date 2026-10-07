# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The IsoFATE driver, `isocalc`: atmospheric escape coupled to interior-atmosphere
equilibrium."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from isofate.atmodeller_coupler import (
    AtmodellerCoupler,
    CouplerState,
    extract_full_output,
    nan_full_output,
)
from isofate.constants import const
from isofate.engine import escape_radius
from isofate.integrators import (
    EXHAUSTION_FRACTION,
    AdaptiveIntegrator,
    EulerIntegrator,
    IsocalcModel,
    no_reequilibration,
)
from isofate.parameters import Parameters
from isofate.species import SYMBOLS


def isocalc(
    parameters: Parameters,
    t_end=5e9,
):
    """Atmospheric escape with event-triggered Atmodeller coupling.

    The integration runs through `isofate.integrators.AdaptiveIntegrator`, which stops on an
    event, re-equilibrates the atmosphere and interior with Atmodeller
    (`AtmodellerCoupler.reequilibrate`), and restarts. The atmosphere and interior are first
    equilibrated once at the start time.

    The events are:
        - Elapsed time: every `IsocalcOptions.n_atmodeller` output steps, i.e. at exactly
          ``t_start + k * n_atmodeller * delta_t``. `n_atmodeller = 0`
          switches the coupling off.
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
        Dictionary of output arrays on the output times ``time`` [s] (abundances ``N_X`` and
        ``N_X_int``, fluxes, radii, surface temperatures, ...), ``t_atmodeller`` (the
        re-equilibration times [s]) and ``atmodeller_final`` (the final Atmodeller snapshot)
    """
    options = parameters.isocalc_options

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps
    t_start_seconds = np.asarray(options.t_start / const.s2yr)  # simulation start time [s]
    t_end_seconds = t_end / const.s2yr  # simulation end time [s]
    duration = t_end_seconds - t_start_seconds  # simulation duration [s]
    delta_t = duration / n_tot  # timestep [s]
    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a = t_start_seconds + delta_t * np.arange(1, n_tot + 1)

    model = IsocalcModel(parameters)
    integrator_class = EulerIntegrator if options.euler else AdaptiveIntegrator
    # Built once per run, outside jit (it builds the Atmodeller model). Also built when the
    # coupling is off, for the molecule output keys
    coupler = AtmodellerCoupler(parameters)
    tracked_species = coupler.tracked_species
    y_raw = jnp.asarray(parameters.initial_abundances(), dtype=float)

    # TODO: temporary output assembly (the former NumPy loop's keys) - clean up
    y_a_int = np.zeros((n_tot, 7))  # interior number array [atoms]
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from the atmosphere descent [K]
    T_surf_atmod_a = np.zeros(n_tot)  # surface temperature used by Atmodeller [K]
    gas_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    melt_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    fO2_a = np.zeros(n_tot)  # O2 activity
    t_atmodeller = np.zeros(0)  # re-equilibration times [s]
    atmod_full_output: dict = {}

    ###_____Integrate_____###

    if options.n_atmodeller == 0:
        integrator = integrator_class(model, on_mass_lost=no_reequilibration)
        y_a, alg_a, _, _ = integrator.integrate(
            jnp.asarray(t_start_seconds), jnp.asarray(t_a), y_raw
        )
    else:
        t_start_j = jnp.asarray(t_start_seconds)
        equilibrated0 = float(parameters.atmosphere_mass(y_raw)) > 0
        if equilibrated0:
            state0, y0 = coupler.reequilibrate(coupler.initial_state(), t_start_j, y_raw)
        else:
            # Nothing to equilibrate: skip Atmodeller for an empty atmosphere
            state0, y0 = coupler.initial_state(), y_raw
        # Re-equilibrate every n_atmodeller output steps
        integrator = integrator_class(
            model,
            on_mass_lost=coupler.reequilibrate,
            reequilibration_steps=options.n_atmodeller,
        )
        y_a, alg_a, restarts, state = integrator.integrate(
            t_start_j, jnp.asarray(t_a), y0, carry=state0
        )
        count = int(restarts.count)

        # TODO: temporary output assembly (the former NumPy loop's keys) - clean up.
        # One entry per equilibration: the initial one, then each restart.
        def stacked(initial, per_restart):
            return np.concatenate([np.asarray(initial)[None], np.asarray(per_restart)[:count]])

        carries: CouplerState = restarts.carry
        eq_t = stacked(t_start_seconds, restarts.t)
        eq_y_before = stacked(y_raw, restarts.y_before)
        y_int_states = stacked(state0.y_int, carries.y_int)
        solution_states = stacked(state0.solution, carries.solution)
        T_surf_states = stacked(state0.surface_temperature, carries.surface_temperature)
        T_atmod_states = stacked(
            state0.surface_temperature_atmodeller, carries.surface_temperature_atmodeller
        )

        # Re-equilibrate once more at the end time when it is on the schedule (n_steps a multiple
        # of n_atmodeller) - the integrators never re-equilibrate at the last
        # output time. The last row then holds the re-equilibrated state.
        y_a = np.array(y_a)
        floor = EXHAUSTION_FRACTION * float(parameters.atmosphere_mass(y0))
        if (
            n_tot % options.n_atmodeller == 0
            and np.all(np.isfinite(y_a[-1]))
            and float(parameters.atmosphere_mass(jnp.asarray(y_a[-1]))) > floor
        ):
            state, y_end = coupler.reequilibrate(state, jnp.asarray(t_a[-1]), jnp.asarray(y_a[-1]))
            eq_t = np.append(eq_t, t_a[-1])
            eq_y_before = np.concatenate([eq_y_before, y_a[-1][None]])
            y_int_states = np.concatenate([y_int_states, np.asarray(state.y_int)[None]])
            solution_states = np.concatenate([solution_states, np.asarray(state.solution)[None]])
            T_surf_states = np.append(T_surf_states, float(state.surface_temperature))
            T_atmod_states = np.append(T_atmod_states, float(state.surface_temperature_atmodeller))
            y_a[-1] = np.asarray(y_end)
            alg_a = model.diagnostics(jnp.asarray(t_a), jnp.asarray(y_a))
        # (the initial entry is a placeholder for the interior when the initial solve was skipped)
        t_atmodeller = eq_t if equilibrated0 else eq_t[1:]
        # Stop if any equilibration (initial, in the loop, or final) failed to converge
        coupler.check_converged(state)

        # The interior and surface temperatures are piecewise constant: each output time takes the
        # latest equilibration at or before it (within half an output spacing, since equilibration
        # times coincide with output times only to the last bit)
        half_gap = 0.5 * float(np.min(np.diff(t_a)))
        which = np.searchsorted(eq_t[1:] - half_gap, t_a, side="right")
        y_a_int = y_int_states[which]
        T_surf_analytic_a = T_surf_states[which]
        T_surf_atmod_a = T_atmod_states[which]

        if options.save_molecules:
            # Re-run each equilibration with full output, with exactly the inputs the hook had
            # (the pre-equilibration atmosphere, the previous interior and the previous solution
            # as the warm start), so it reproduces the same equilibrium
            y_int_before_states = np.concatenate([np.zeros((1, 7)), y_int_states[:-1]])
            guesses = [coupler.cold_guess()] + [jnp.asarray(g) for g in solution_states[:-1]]
            # Jitted once for all the re-runs (eagerly, the warm start's jax.lax.cond recompiles on
            # every call)
            run_full_output = eqx.filter_jit(coupler.run)
            for j in range(len(eq_t)):
                y_before = jnp.asarray(eq_y_before[j])
                t_j = jnp.asarray(eq_t[j])
                rerun = run_full_output(
                    escape_radius(parameters, y_before, t_j),
                    parameters.atmosphere_mean_mu(y_before),
                    y_before,
                    jnp.asarray(y_int_before_states[j]),
                    initial_guess=guesses[j],
                    full_output=True,
                )
                if not bool(rerun.success):
                    raise RuntimeError(
                        "Atmodeller failed to converge re-running the equilibration at "
                        f"t = {eq_t[j] * const.s2yr:.6g} yr for save_molecules"
                    )
                output = jax.device_get(rerun.output)
                rows = which == j
                for sp in tracked_species:
                    gas_num_a[sp.label][rows] = output[sp.gas_name]["gas"]["number_moles"][0][0]
                    if sp.melt_name is not None:
                        melt_num_a[sp.label][rows] = output[sp.melt_name]["silicate_melt"][
                            "number_moles"
                        ][0][0]
                fO2_a[rows] = output["O2_g"]["gas"]["activity"][0][0]

        # Final full snapshot (read-only diagnostic), "atmodeller_final"
        y_final = jnp.asarray(y_a[-1])
        if bool(jnp.all(jnp.isfinite(y_final))) and float(parameters.atmosphere_mass(y_final)) > 0:
            t_final = jnp.asarray(t_a[-1])
            final = coupler.run(
                escape_radius(parameters, y_final, t_final),
                parameters.atmosphere_mean_mu(y_final),
                y_final,
                state.y_int,
                initial_guess=state.solution,
                full_output=True,
            )
            if not bool(final.success):
                raise RuntimeError(
                    "Atmodeller failed to converge for the final snapshot (atmodeller_final)"
                )
            atmod_full_output = extract_full_output(jax.device_get(final.output))
        else:
            atmod_full_output = nan_full_output()

    y_a = np.asarray(y_a)
    Matm_a = np.asarray(alg_a.atmosphere_mass)
    fatm_a = np.asarray(alg_a.atmosphere_mass_fraction)
    Renv_a = np.asarray(alg_a.convective_envelope_thickness)
    Rp_a = np.asarray(alg_a.escape_radius)
    Vpot_a = np.asarray(alg_a.gravitational_potential)
    phi_a = np.asarray(alg_a.mass_flux)
    phic_a = np.asarray(alg_a.critical_mass_flux)
    Phi_a = np.asarray(alg_a.number_flux)
    x_a = np.asarray(alg_a.atom_fractions)
    Mloss_a = phi_a * np.asarray(alg_a.area) * delta_t

    # save results
    solutions = {
        "time": t_a,
        "Rp": Rp_a,
        "Ratm": Renv_a,
        "Matm": Matm_a,
        "Vpot": Vpot_a,
        "fatm": fatm_a,
        "Mloss": Mloss_a,
        "phi": phi_a,
        "phic": phic_a,
        **{f"N_{symbol}": y_a[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        **{f"N_{symbol}_int": y_a_int[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        **{f"x{i + 1}": x_a[:, i].copy() for i in range(len(SYMBOLS))},
        **{f"Phi_{symbol}": Phi_a[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
        "t_atmodeller": t_atmodeller,
    }
    if options.save_molecules:
        for label, arr in gas_num_a.items():
            solutions[f"n_{label}_a"] = arr
        for label, arr in melt_num_a.items():
            solutions[f"n_{label}_a_int"] = arr
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions
