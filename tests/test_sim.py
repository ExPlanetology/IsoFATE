# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression anchor for isofate/sim.py's driver scenario (LHS 1140 b), pinned separately from
tests/test_pipeline.py's synthetic toy scenario. sim.py itself can't be imported directly (it's a
plotting script with top-level side effects, including a blocking `plt.show()`), so this
reconstructs its driver inputs directly - kept in sync with sim.py by hand.

n_atmodeller=0 and dynamic_phi=False here pin the scenario as it was when the regression values
were taken (sim.py now runs with dynamic_phi=True and Atmodeller coupling); update the pinned
values if these change.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from isofate.constants import const
from isofate.integrators import (
    EXHAUSTION_FRACTION,
    AdaptiveIntegrator,
    IntegrationStop,
    IsocalcModel,
    no_reequilibration,
)
from isofate.escape.mechanisms import XUVEscape
from isofate.initial_condition import AbundanceInitialCondition, ProtosolarInitialCondition
from isofate.atmodeller_coupler import AtmodellerCoupler
from isofate.isofate_coupler import isocalc_jax3
from isofate.parameters import IsocalcOptions, Parameters
from isofate.presets import LHS1140b, LHS1140Star
from isofate.species import DEFAULT_SPECIES, SYMBOLS
from isofate.system import System


# isofate/sim.py's current run settings, kept in sync with that script by hand (the remaining
# inputs are reconstructed by _sim_isocalc_kwargs)
SIM_PY = dict(n_steps=int(1e5), n_atmodeller=int(1e4), f_atm=0.01, dynamic_phi=True)


def _sim_isocalc_kwargs(n_steps=int(1e5), n_atmodeller=0, f_atm=0.01, dynamic_phi=False):
    """Reconstructs isofate/sim.py's driver call.

    `n_steps` defaults to sim.py's own value but can be overridden (e.g. by the isocalc/
    isocalc_jax3 cross-checks below, which don't need full resolution to confirm agreement).
    `n_atmodeller` defaults to 0 (Atmodeller disabled) but can be overridden to exercise the
    Atmodeller coupling. `f_atm` and `dynamic_phi` default to the values the regression tests
    were pinned with; `SIM_PY` holds sim.py's current values.
    """
    star = LHS1140Star
    planet = LHS1140b
    system = System(star=star, planet=planet)

    t_jump = star.t_jump

    Fp = system.insolation
    F0 = Fp * 1e-3  # use for M star
    F_final = 0.17  # LHS 1140; 0.170 for GJ 699 MUSCLES
    flux_model = "power law"
    stellar_type = "M1"
    t_sat = t_jump * 1e9  # XUV saturation time [yr]
    time = 5e9  # total simulation time [yr]
    t0 = 1e6  # start time [yr]
    t_pms = 0  # pms phase duration [yr]
    step_fn = True
    RR = True
    eps = 0.15
    rad_evol = True
    thermal = True
    save_molecules = False

    escape = XUVEscape(
        F0=F0,
        eps=eps,
        t0=t0,
        t_sat=t_sat,
        beta=-1.23,
        step_fn=step_fn,
        F_final=F_final,
        t_pms=t_pms,
        pms_factor=1e2,
        flux_model=flux_model,
        activity="medium",
        stellar_type=stellar_type,
        RR=RR,
    )
    options = IsocalcOptions(
        rad_evol=rad_evol,
        n_steps=n_steps,
        t_start=t0,
        thermal=thermal,
        n_atmodeller=n_atmodeller,
        save_molecules=save_molecules,
        dynamic_phi=dynamic_phi,
    )
    parameters = Parameters(
        system,
        escape_mechanism=escape,
        initial_condition=ProtosolarInitialCondition(atmosphere_mass_fraction=f_atm),
        isocalc_options=options,
    )
    return dict(parameters=parameters, t_end=time)


def _with_initial_abundances(kwargs, abundances):
    """`kwargs` with the initial condition replaced by the given abundances [atoms]."""
    initial_condition = AbundanceInitialCondition(tuple(abundances))
    parameters = dataclasses.replace(kwargs["parameters"], initial_condition=initial_condition)
    return {**kwargs, "parameters": parameters}


def test_protosolar_initial_condition():
    """ProtosolarInitialCondition gives exactly the protosolar mole ratios to H, an atmospheric
    mass of exactly f_atm times the planet mass (in the species' atomic masses), and an O
    enhancement that only changes O/H."""
    Mp = LHS1140b.mass
    f_atm = 0.01
    y0 = np.asarray(ProtosolarInitialCondition(f_atm).abundances(Mp))
    ratios = ProtosolarInitialCondition.PROTOSOLAR_MOLE_RATIOS_TO_H
    H = DEFAULT_SPECIES.index("H")

    for i, symbol in enumerate(DEFAULT_SPECIES.species):
        assert y0[i] / y0[H] == pytest.approx(ratios[symbol], rel=1e-12), symbol
    assert float(np.dot(y0, DEFAULT_SPECIES.atomic_masses)) == pytest.approx(
        f_atm * float(Mp), rel=1e-12
    )

    y_enhanced = np.asarray(ProtosolarInitialCondition(f_atm, OtoH_enhancement=2).abundances(Mp))
    for i, symbol in enumerate(DEFAULT_SPECIES.species):
        expected = 2 * ratios[symbol] if symbol == "O" else ratios[symbol]
        assert y_enhanced[i] / y_enhanced[H] == pytest.approx(expected, rel=1e-12), symbol


def test_abundance_initial_condition():
    """AbundanceInitialCondition returns its abundances, and rejects a mismatched length."""
    abundances = (1e45, 1e44, 1e41, 1.5e45, 1e44, 1e43, 1e43)
    np.testing.assert_array_equal(
        np.asarray(AbundanceInitialCondition(abundances).abundances(LHS1140b.mass)), abundances
    )
    with pytest.raises(ValueError):
        AbundanceInitialCondition(abundances[:-1]).abundances(LHS1140b.mass)


def test_sim_regression():
    """Pins isofate/sim.py's current driver scenario end to end - not cross-checked against an
    independent reference (unlike test_pipeline.py's toy scenario), just a regression tripwire for
    this specific, real-world (LHS 1140 b) case that the test suite otherwise doesn't cover.

    Re-pinned when isocalc's dynamic_phi=False path switched from the hardcoded const.mu_H/mu_He
    atomic masses to IsoFATESpecies.atomic_masses (molmass-derived IUPAC values) via
    EscapeNumberFlux - a deliberate ~1e-5 relative accuracy improvement, not a regression.

    Re-pinned again when isocalc stopped seeding the initial atmosphere mass from
    `Mp * planet.f_atm` and instead derives M_atm/f_atm fresh from y every iteration (matching
    the JAX drivers) - `dot(isofate_species_abund, atomic_masses)` differs from
    `Mp * planet.f_atm` by ~3e-4 relative here, traced to sim.py's N_N abundance formula omitting
    the `/(1 + HetoH_protosolar_mass)` normalization every other species' formula has. A deliberate
    accuracy improvement (removes a redundant, slightly-inconsistent second bookkeeping of
    atmosphere mass), not a regression.

    Re-pinned again (~2e-5 relative) when isocalc started evaluating its age-dependent physics
    (R_env's contraction term, the XUV flux history) at the integration time t0 + n*delta_t rather
    than the output label t_a[n], which runs about one step ahead. This moves isocalc towards the
    adaptive JAX driver (final Matm difference 6.7e-5 -> 4.7e-5), so it is a correction, not a
    regression.

    Re-pinned again when the initial abundances moved to ProtosolarInitialCondition, which uses
    exact protosolar mole ratios to H normalised by the total mass (in the species' atomic
    masses). This fixes N's missing normalisation (initial N about 1.39x lower) and the former
    recipe's H "remainder", which put every X/H about 1.5% above protosolar.

    Moved from the legacy isocalc to isocalc_jax3(euler=True) when isocalc was retired: the two
    agreed to round-off (about 1e-13) on every output, so the pins are unchanged.
    """
    sol = isocalc_jax3(**_sim_isocalc_kwargs(), euler=True)

    assert sol["Matm"][-1] == pytest.approx(2.783430554314534e23, rel=1e-6)
    assert sol["N_H"][-1] == pytest.approx(1.1152201767281153e50, rel=1e-6)
    assert sol["N_He"][-1] == pytest.approx(1.3278748701495348e49, rel=1e-6)
    assert sol["N_D"][-1] == pytest.approx(2.3224981664139336e45, rel=1e-6)
    assert sol["N_O"][-1] == pytest.approx(8.267505240308852e46, rel=1e-6)
    assert sol["N_C"][-1] == pytest.approx(4.1256558350670687e46, rel=1e-6)
    assert sol["N_N"][-1] == pytest.approx(1.1396109241276901e46, rel=1e-6)
    assert sol["N_S"][-1] == pytest.approx(2.568944911327128e45, rel=1e-6)
    assert sol["Rp"][-1] == pytest.approx(13827208.10305326, rel=1e-6)
    assert sol["Vpot"][-1] == pytest.approx(153853762.14169946, rel=1e-6)


def test_isocalc_jax3_adaptive_matches_euler():
    """Without Atmodeller, the adaptive solver (Tsit5) and the fixed-step Euler scheme agree to
    about 6e-4 relative at 2000 steps (the Euler error), within rel=1e-2."""
    kwargs = _sim_isocalc_kwargs(n_steps=2000)
    sol_euler = isocalc_jax3(**kwargs, euler=True)
    sol = isocalc_jax3(**kwargs)

    for key in ("Matm", "N_H", "N_He", "N_D", "N_O", "N_C", "N_N", "N_S", "Rp", "Vpot"):
        assert sol[key][-1] == pytest.approx(sol_euler[key][-1], rel=1e-2), key


def test_isocalc_jax3_coupled_adaptive_matches_euler():
    """With Atmodeller re-equilibrating every 100 of 2000 steps, both schemes follow the same
    schedule and coupling, so over the whole run they agree to about 6e-4 relative for the bulk
    quantities and 4e-3 for the trace species N and S (the Euler error at 2000 steps), within
    rel=1e-2.
    """
    kwargs = _sim_isocalc_kwargs(n_steps=2000, n_atmodeller=100)
    sol_euler = isocalc_jax3(**kwargs, euler=True)
    sol = isocalc_jax3(**kwargs)

    keys = [f"N_{symbol}" for symbol in SYMBOLS] + [f"N_{symbol}_int" for symbol in SYMBOLS]
    for key in keys + ["Matm", "Rp", "Vpot", "T_surf_atmod"]:
        np.testing.assert_allclose(sol[key], sol_euler[key], rtol=1e-2, err_msg=key)


@pytest.mark.parametrize("euler", [False, True])
def test_isocalc_jax3_empty_atmosphere(euler):
    """An empty initial atmosphere is not equilibrated, so no atmosphere is
    created from nothing: the atmosphere stays empty (rows are 0, or inf for the adaptive solver,
    which marks exhausted rows that way), the interior stays empty, and atmodeller_final is NaN.
    """
    kwargs = _with_initial_abundances(_sim_isocalc_kwargs(n_steps=40, n_atmodeller=10), (0.0,) * 7)
    sol = isocalc_jax3(**kwargs, euler=euler)

    assert len(sol["t_atmodeller"]) == 0
    for symbol in SYMBOLS:
        assert np.all((sol[f"N_{symbol}"] == 0) | np.isinf(sol[f"N_{symbol}"])), symbol
        assert np.all(sol[f"N_{symbol}_int"] == 0), symbol
    assert np.all(np.isnan(list(sol["atmodeller_final"].values())))


def test_reequilibrate_warm_start_matches_cold_start():
    """`AtmodellerCoupler.reequilibrate` warm-starts from the previous solution in its carry; the
    result is the same equilibrium as a cold start."""
    model, t_start, _, y0 = _sim_integrator_setup()
    coupler = AtmodellerCoupler(model.parameters)
    state, y_eq = coupler.reequilibrate(coupler.initial_state(), t_start, y0)
    assert np.all(np.isfinite(np.asarray(state.solution)))  # a real solution to warm-start from

    # A later re-equilibration after some escape (5% of every species lost)
    y_lost = 0.95 * y_eq
    warm_state, y_warm = coupler.reequilibrate(state, t_start, y_lost)
    cold = dataclasses.replace(state, solution=coupler.cold_guess())
    cold_state, y_cold = coupler.reequilibrate(cold, t_start, y_lost)

    np.testing.assert_allclose(np.asarray(y_warm), np.asarray(y_cold), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(warm_state.y_int), np.asarray(cold_state.y_int), rtol=1e-6
    )


def _sim_integrator_setup(mass_loss_fraction=None):
    """The LHS 1140 b scenario set up for direct `IsocalcModel` use (no Atmodeller), with the
    given `IsocalcOptions.mass_loss_fraction` (``None`` keeps the default). This scenario is
    escape-dominated (about 17% of the atmospheric mass is lost over the run), unlike
    test_pipeline's toy scenario.

    Returns:
        `(model, t_start, t_a, y0)`
    """
    kwargs = _sim_isocalc_kwargs(n_steps=200)
    options = kwargs["parameters"].isocalc_options
    if mass_loss_fraction is not None:
        options = dataclasses.replace(options, mass_loss_fraction=mass_loss_fraction)
    parameters = dataclasses.replace(kwargs["parameters"], isocalc_options=options)

    t_start = options.t_start / const.s2yr
    t_end = kwargs["t_end"] / const.s2yr
    delta_t = (t_end - t_start) / options.n_steps
    t_a = jnp.asarray(delta_t * np.linspace(1, options.n_steps + 1, options.n_steps) + t_start)
    y0 = parameters.initial_abundances()

    return IsocalcModel(parameters), jnp.asarray(t_start), t_a, y0


def _integrate_sim(mass_loss_fraction=None):
    """One `IsocalcModel.integrate` call on the LHS 1140 b scenario."""
    model, t_start, t_a, y0 = _sim_integrator_setup(mass_loss_fraction)
    sol = model.integrate(t_start, t_a, y0)
    y_a, stop = sol.ys[0], IntegrationStop.from_solution(sol)

    return model.parameters, np.asarray(t_a), y0, y_a, stop


def test_integrator_mass_loss_event():
    """With `mass_loss_fraction=0.05`, the integration stops once 5% of the initial atmospheric
    mass is lost - slightly past it, since the boolean condition fires at the end of the step on
    which it becomes true - and the outputs after the stop are inf."""
    parameters, t_a, y0, y_a, stop = _integrate_sim(0.05)

    assert bool(stop.mass_lost)
    assert float(stop.t) < t_a[-1]

    mass_ratio = float(parameters.atmosphere_mass(stop.y) / parameters.atmosphere_mass(y0))
    assert 0.9 < mass_ratio <= 0.95

    # Rows after the event are left as diffrax leaves them
    after_event = t_a > float(stop.t)
    assert np.all(np.isfinite(y_a[~after_event]))
    assert np.all(np.isinf(y_a[after_event]))


def test_integrator_default_runs_to_end():
    """With the default `mass_loss_fraction` (stop only at exhaustion), this scenario never
    exhausts, so the integration runs to the last output time."""
    _, t_a, _, _, stop = _integrate_sim()

    assert not bool(stop.mass_lost)
    assert float(stop.t) == pytest.approx(t_a[-1])


def test_adaptive_integrator_identity_hook_matches_single_solve():
    """Plumbing check: restarting every 5% of mass loss with a hook that leaves the state unchanged
    must reproduce one uninterrupted integration."""
    model, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = model.parameters

    y_a, _, restarts, _ = AdaptiveIntegrator(model, on_mass_lost=no_reequilibration).integrate(
        t_start, t_a, y0
    )

    reference_model, _, _, _ = _sim_integrator_setup()
    y_reference = reference_model.integrate(t_start, t_a, y0).ys[0]

    count = int(restarts.count)
    assert count >= 2
    assert bool(restarts.finished)
    assert np.all(np.isnan(restarts.t[count:]))
    segment_start_mass = parameters.atmosphere_mass(y0)
    for i in range(count):
        assert np.all(restarts.y_after[i] == restarts.y_before[i])
        # Slightly past 5%: the boolean condition fires at the end of the step on which it becomes
        # true, and later segments take larger steps
        mass = parameters.atmosphere_mass(restarts.y_before[i])
        assert 0.9 < float(mass / segment_start_mass) <= 0.95
        segment_start_mass = parameters.atmosphere_mass(restarts.y_after[i])

    assert np.all(np.isfinite(y_a))
    # Restarts change the step-size sequence, so the two numerical solutions differ at about the
    # accumulated solver tolerance
    np.testing.assert_allclose(y_a, y_reference, rtol=1e-3)


def test_adaptive_integrator_placeholder_hook():
    """The default placeholder hook removes 1% of every abundance at each restart, and the next
    segment continues from the reduced state."""
    model, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = model.parameters

    y_a, _, restarts, _ = AdaptiveIntegrator(model).integrate(t_start, t_a, y0)
    y_identity, _, restarts_identity, _ = AdaptiveIntegrator(
        model, on_mass_lost=no_reequilibration
    ).integrate(t_start, t_a, y0)

    count = int(restarts.count)
    assert count >= 2
    assert int(restarts_identity.count) >= 2
    for i in range(count):
        np.testing.assert_allclose(restarts.y_after[i], 0.99 * restarts.y_before[i])
        i_next = int(np.searchsorted(np.asarray(t_a), float(restarts.t[i]), side="right"))
        if i_next < t_a.shape[0]:
            assert float(parameters.atmosphere_mass(y_a[i_next])) < float(
                parameters.atmosphere_mass(restarts.y_before[i])
            )

    assert np.all(np.isfinite(y_a))
    assert float(parameters.atmosphere_mass(y_a[-1])) < float(
        parameters.atmosphere_mass(y_identity[-1])
    )


def test_adaptive_integrator_hook_and_floor():
    """The hook is called once per restart with its carried state threaded through, and a hook
    that removes almost all of the atmosphere ends the integration at the exhaustion floor."""
    model, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = model.parameters

    def counting_hook(carry, t, y):
        return carry + 1, y

    _, _, restarts, n_calls = AdaptiveIntegrator(model, on_mass_lost=counting_hook).integrate(
        t_start, t_a, y0, carry=jnp.array(0)
    )
    assert int(n_calls) == int(restarts.count) >= 2

    def removing_hook(carry, t, y):
        return carry, 1e-7 * y

    y_a, _, restarts, _ = AdaptiveIntegrator(model, on_mass_lost=removing_hook).integrate(
        t_start, t_a, y0
    )
    assert int(restarts.count) == 1
    i_next = int(np.searchsorted(np.asarray(t_a), float(restarts.t[0]), side="right"))
    np.testing.assert_allclose(
        y_a[i_next:], np.broadcast_to(restarts.y_after[0], y_a[i_next:].shape)
    )
    assert float(parameters.atmosphere_mass(restarts.y_after[0])) <= EXHAUSTION_FRACTION * float(
        parameters.atmosphere_mass(y0)
    )


def test_adaptive_integrator_max_segments():
    """Running out of segments is reported by `restarts.finished` (and raises only if Equinox
    errors are enabled, which isofate turns off by default)."""
    model, t_start, t_a, y0 = _sim_integrator_setup(0.05)

    _, _, restarts, _ = AdaptiveIntegrator(model, max_segments=2).integrate(t_start, t_a, y0)
    assert int(restarts.count) == 1
    assert not bool(restarts.finished)


def _jax3_kwargs(n_steps=200, mass_loss_fraction=0.05, save_molecules=False):
    """LHS 1140 b with the Atmodeller coupling on, re-equilibrating every `mass_loss_fraction` of
    atmospheric mass lost (`isocalc_jax3`)."""
    kwargs = _sim_isocalc_kwargs(n_steps=n_steps, n_atmodeller=1)
    options = dataclasses.replace(
        kwargs["parameters"].isocalc_options,
        mass_loss_fraction=mass_loss_fraction,
        save_molecules=save_molecules,
    )
    kwargs["parameters"] = dataclasses.replace(kwargs["parameters"], isocalc_options=options)
    return kwargs


def test_reequilibrate_conserves_each_element():
    """Each Atmodeller re-equilibration in the segmented integration only redistributes atoms
    between the atmosphere and the interior: per element (D counted with H), atmosphere + interior
    is the same before and after."""
    model, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    coupler = AtmodellerCoupler(model.parameters)
    state0, y_eq = coupler.reequilibrate(coupler.initial_state(), t_start, y0)

    def total(y_atm, y_int):
        return np.asarray(coupler.aggregate_D_into_H(jnp.asarray(y_atm) + jnp.asarray(y_int)))

    np.testing.assert_allclose(total(y_eq, state0.y_int), total(y0, 0.0), rtol=1e-10)

    _, _, restarts, _ = AdaptiveIntegrator(model, on_mass_lost=coupler.reequilibrate).integrate(
        t_start, t_a, y_eq, carry=state0
    )
    count = int(restarts.count)
    assert count >= 1
    y_int_before = state0.y_int
    for i in range(count):
        np.testing.assert_allclose(
            total(restarts.y_after[i], restarts.carry.y_int[i]),
            total(restarts.y_before[i], y_int_before),
            rtol=1e-10,
        )
        y_int_before = restarts.carry.y_int[i]


@pytest.mark.parametrize("save_molecules", [False, True])
def test_isocalc_jax3_outputs(save_molecules):
    """isocalc_jax3 returns the documented keys (the legacy isocalc's, plus `t_atmodeller` and,
    with save_molecules, the molecule histories), all finite, with the interior outputs piecewise
    constant between re-equilibrations."""
    kwargs = _jax3_kwargs(save_molecules=save_molecules)
    sol = isocalc_jax3(**kwargs)

    expected_keys = (
        {"time", "Rp", "Ratm", "Matm", "Vpot", "fatm", "Mloss", "phi", "phic"}
        | {"T_surf_analytic", "T_surf_atmod", "t_atmodeller", "atmodeller_final"}
        | {f"N_{symbol}" for symbol in SYMBOLS}
        | {f"N_{symbol}_int" for symbol in SYMBOLS}
        | {f"Phi_{symbol}" for symbol in SYMBOLS}
        | {f"x{i + 1}" for i in range(len(SYMBOLS))}
    )
    molecule_keys = {key for key in sol if key.startswith("n_") or key == "fO2_a"}
    assert set(sol) - molecule_keys == expected_keys
    assert bool(molecule_keys) == save_molecules
    for key, value in sol.items():
        if key != "atmodeller_final":
            assert np.all(np.isfinite(value)), key
    assert np.all(np.isfinite(list(sol["atmodeller_final"].values())))

    # The interior only changes at the output rows straddling a re-equilibration
    restart_rows = set(np.searchsorted(sol["time"], sol["t_atmodeller"][1:]).tolist())
    for symbol in ("H", "O", "C"):
        change_rows = set((np.nonzero(np.diff(sol[f"N_{symbol}_int"]))[0] + 1).tolist())
        assert change_rows <= restart_rows, symbol



@pytest.mark.parametrize("euler", [False, True])
def test_isocalc_jax3_fixed_radius(euler):
    """With `rad_evol=False` the escape radius is fixed at the rocky radius, with no envelope and
    no Bondi/Hill cap. Without Atmodeller: with the radius at the rocky surface, the atmosphere
    descent to the surface has zero thickness and the Atmodeller solve does not converge (see the
    FIXME in AtmodellerCoupler.run)."""
    kwargs = _sim_isocalc_kwargs(n_steps=200, n_atmodeller=0)
    options = dataclasses.replace(kwargs["parameters"].isocalc_options, rad_evol=False)
    kwargs["parameters"] = dataclasses.replace(kwargs["parameters"], isocalc_options=options)
    sol = isocalc_jax3(**kwargs, euler=euler)

    rocky_radius = float(kwargs["parameters"].system.planet.rocky_radius)
    np.testing.assert_allclose(sol["Rp"], rocky_radius, rtol=1e-12)
    assert np.all(sol["Ratm"] == 0)


# Final values of isocalc_jax3(euler=True), pinned when the legacy isocalc was retired; isocalc
# (an independent implementation of the same scheme) agreed with these to round-off (<= 3e-14)
EULER_PINS = {
    "no_coupling": {
        "N_H": 1.1063425993631312e50, "N_He": 1.3202485174289307e49,
        "N_D": 2.3054541453965797e45, "N_O": 8.225110774486556e46,
        "N_C": 4.1030698391591003e46, "N_N": 1.133580251719231e46,
        "N_S": 2.560959078964655e45, "Matm": 2.7633265628293022e23, "Rp": 13811712.72503741,
    },
    "coupled": {
        "N_H": 1.0105605048959224e50, "N_He": 1.2494906606052938e49,
        "N_D": 2.1025197819225204e45, "N_O": 2.108190671684607e44,
        "N_C": 4.115071015930003e46, "N_N": 3.4255950806050936e36,
        "N_S": 5.2060018653347905e31, "N_H_int": 1.235102492873567e49,
        "N_He_int": 9.212546895526311e47, "N_D_int": 2.5696901980509725e44,
        "N_O_int": 8.257528520715939e46, "N_C_int": 1.7832140522299958e40,
        "N_N_int": 1.1418784569042538e46, "N_S_int": 2.5692265288053014e45,
        "Matm": 2.5302054365595094e23, "Rp": 13634928.079489931,
        "T_surf_atmod": 3166.941216947133, "final:O2_fugacity": 1.0190482674105137e-13,
        "final:H2O_atm": 3.482683755461788e20, "final:H2_atm": 8.374919983983618e25,
        "final:CO2_atm": 214838581589.85864,
    },
    "sim_py": {
        "N_H": 1.015405862967759e50, "N_He": 1.2538893809509016e49,
        "N_D": 2.111721787959363e45, "N_O": 2.0941887958672807e44,
        "N_C": 4.1323021237984494e46, "N_N": 3.288181226035627e36,
        "N_S": 5.10252015738741e31, "N_H_int": 1.24225642518573e49,
        "N_He_int": 9.275243592284056e47, "N_D_int": 2.5834989288224265e44,
        "N_O_int": 8.257675541168726e46, "N_C_int": 1.751699812297779e40,
        "N_N_int": 1.1418784569179982e46, "N_S_int": 2.569226528805283e45,
        "Matm": 2.5412731368390358e23, "Rp": 13643750.793812025,
        "T_surf_atmod": 3170.997201663295, "final:O2_fugacity": 1.0206628479229042e-13,
        "final:H2O_atm": 3.459586606070278e20, "final:H2_atm": 8.415086681061429e25,
        "final:CO2_atm": 210720992918.0445,
    },
}


@pytest.mark.parametrize(
    "case,settings",
    [
        ("no_coupling", dict(n_steps=200, n_atmodeller=0)),
        ("coupled", dict(n_steps=200, n_atmodeller=20)),
        # sim.py's settings (dynamic phi, 10 re-equilibrations) at 1/50 of its resolution, with
        # f_atm = 0.01: with sim.py's smaller atmosphere some Atmodeller solves sit on the edge of
        # convergence at this resolution
        (
            "sim_py",
            {
                **SIM_PY,
                "n_steps": SIM_PY["n_steps"] // 50,
                "n_atmodeller": SIM_PY["n_atmodeller"] // 50,
                "f_atm": 0.01,
            },
        ),
    ],
)
def test_isocalc_jax3_euler_regression(case, settings):
    """Pins isocalc_jax3(euler=True), isocalc's fixed-step scheme, on short LHS 1140 b runs."""
    sol = isocalc_jax3(**_sim_isocalc_kwargs(**settings), euler=True)

    for key, expected in EULER_PINS[case].items():
        if key.startswith("final:"):
            value = sol["atmodeller_final"][key.removeprefix("final:")]
        else:
            value = sol[key][-1]
        assert float(value) == pytest.approx(expected, rel=1e-8), key
