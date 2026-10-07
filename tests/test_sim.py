# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression anchor for isofate/sim.py's driver scenario (LHS 1140 b), pinned separately from
tests/test_pipeline.py's synthetic toy scenario. sim.py itself can't be imported directly (it's a
plotting script with top-level side effects, including a blocking `plt.show()`), so this
reconstructs its isocalc() inputs directly - kept in sync with sim.py by hand.

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
    IntegrationStop,
    IsocalcIntegrator,
    integrate_segments,
)
from isofate.escape.mechanisms import XUVEscape
from isofate.atmodeller_coupler_new import AtmodellerCoupler
from isofate.isofate_coupler import isocalc, isocalc_jax, isocalc_jax2, isocalc_jax3
from isofate.parameters import IsocalcOptions, Parameters
from isofate.presets import LHS1140b, LHS1140Star
from isofate.system import System


def _sim_isocalc_kwargs(n_steps=int(1e5), n_atmodeller=0):
    """Reconstructs isofate/sim.py's isocalc() call, as of the current version of that script.

    `n_steps` defaults to sim.py's own value but can be overridden (e.g. by the isocalc/
    isocalc_jax cross-check below, which doesn't need full resolution to confirm agreement).
    `n_atmodeller` defaults to sim.py's current value (0, Atmodeller disabled) but can be
    overridden to exercise isocalc_jax2's segmented-diffeqsolve Atmodeller coupling.
    """
    star = LHS1140Star
    planet = LHS1140b
    system = System(star=star, planet=planet)

    # Not a Planet field (removed) - matches sim.py's own scripting-level constant, used only to
    # scale this function's solar-abundance-ratio construction of N_H/N_He/etc. below.
    f_atm = 0.01
    Mp = planet.mass
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
    melt_fraction_override = False
    save_molecules = False
    mantle_iron = None
    dynamic_phi = False

    M_atm = Mp * f_atm  # initial atmospheric mass [kg]
    OtoH_enhanced_mass = const.OtoH_protosolar * (const.mu_O / const.mu_H)

    N_He = (const.HetoH_protosolar_mass / (1 + const.HetoH_protosolar_mass)) * M_atm / const.mu_He
    N_H = (
        (
            1
            - const.DtoH_solar_mass
            - OtoH_enhanced_mass
            - const.CtoH_protosolar_mass
            - const.StoH_protosolar_mass
            - const.NtoH_protosolar_mass
        )
        * M_atm
        / (1 + const.HetoH_protosolar_mass)
        / const.mu_H
    )
    N_D = const.DtoH_solar_mass * M_atm / (1 + const.HetoH_protosolar_mass) / const.mu_D
    N_O = OtoH_enhanced_mass * M_atm / (1 + const.HetoH_protosolar_mass) / const.mu_O
    N_C = const.CtoH_protosolar_mass * M_atm / (1 + const.HetoH_protosolar_mass) / const.mu_C
    N_N = const.NtoH_protosolar_mass * M_atm / const.mu_N
    N_S = const.StoH_protosolar_mass * M_atm / (1 + const.HetoH_protosolar_mass) / const.mu_S

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
        melt_fraction_override=melt_fraction_override,
        n_steps=n_steps,
        t_start=t0,
        thermal=thermal,
        n_atmodeller=n_atmodeller,
        save_molecules=save_molecules,
        mantle_iron=mantle_iron,
        dynamic_phi=dynamic_phi,
    )
    parameters = Parameters(system, escape_mechanism=escape, isocalc_options=options)
    return dict(
        parameters=parameters,
        t_end=time,
        # ordered per isofate.species.SYMBOLS
        isofate_species_abund=(N_H, N_He, N_D, N_O, N_C, N_N, N_S),
    )


def test_sim_regression():
    """Pins isofate/sim.py's current driver scenario end to end - not cross-checked against an
    independent reference (unlike test_pipeline.py's toy scenario), just a regression tripwire for
    this specific, real-world (LHS 1140 b) case that the test suite otherwise doesn't cover.

    Re-pinned when isocalc's dynamic_phi=False path switched from the hardcoded const.mu_H/mu_He
    atomic masses to IsoFATESpecies.atomic_masses (molmass-derived IUPAC values) via
    EscapeNumberFlux - a deliberate ~1e-5 relative accuracy improvement, not a regression.

    Re-pinned again when isocalc stopped seeding the initial atmosphere mass from
    `Mp * planet.f_atm` and instead derives M_atm/f_atm fresh from y every iteration (matching
    isocalc_jax) - `dot(isofate_species_abund, atomic_masses)` differs from `Mp * planet.f_atm` by
    ~3e-4 relative here, traced to sim.py's N_N abundance formula omitting the
    `/(1 + HetoH_protosolar_mass)` normalization every other species' formula has. A deliberate
    accuracy improvement (removes a redundant, slightly-inconsistent second bookkeeping of
    atmosphere mass), not a regression.

    Re-pinned again (~2e-5 relative) when isocalc started evaluating its age-dependent physics
    (R_env's contraction term, the XUV flux history) at the integration time t0 + n*delta_t rather
    than the output label t_a[n], which runs about one step ahead. This moves isocalc towards
    isocalc_jax (final Matm difference 6.7e-5 -> 4.7e-5), so it is a correction, not a regression.
    """
    sol = isocalc(**_sim_isocalc_kwargs())

    assert sol["Matm"][-1] == pytest.approx(2.784654750240057e23, rel=1e-6)
    assert sol["N_H"][-1] == pytest.approx(1.1097311942220029e50, rel=1e-6)
    assert sol["N_He"][-1] == pytest.approx(1.3414512951085343e49, rel=1e-6)
    assert sol["N_D"][-1] == pytest.approx(2.3455061586587238e45, rel=1e-6)
    assert sol["N_O"][-1] == pytest.approx(8.353162454098547e46, rel=1e-6)
    assert sol["N_C"][-1] == pytest.approx(4.1684331956265637e46, rel=1e-6)
    assert sol["N_N"][-1] == pytest.approx(1.595305687580649e46, rel=1e-6)
    assert sol["N_S"][-1] == pytest.approx(2.595544854318647e45, rel=1e-6)
    assert sol["Rp"][-1] == pytest.approx(13824623.332857858, rel=1e-6)
    assert sol["Vpot"][-1] == pytest.approx(153883944.20291469, rel=1e-6)


def test_isocalc_isocalc_jax_cross_check():
    """isocalc and isocalc_jax should agree closely on the same scenario: both now derive
    M_atm/f_atm fresh from y (see isocalc's and IsocalcIntegrator's docstrings), so the only
    remaining difference is the integration scheme itself - isocalc's fixed-step Euler loop vs.
    isocalc_jax's adaptive Tsit5 (diffrax). They also now share the exact same call signature
    (`parameters`, `time`, `isofate_species_abund`), so the same kwargs work for both.

    Uses n_steps=2000 rather than test_sim_regression's full 1e5 - at 1e5 the two agree to
    ~2e-5 relative (verified by hand), but that isocalc run alone takes ~5 minutes; this only
    needs enough resolution to confirm the two implementations agree, not to test isocalc's own
    convergence. At n_steps=2000, isocalc runs in a few seconds and the two agree to ~5e-4
    relative - rel=1e-2 below leaves ample margin without being sensitive to incidental changes
    in either integrator.
    """
    kwargs = _sim_isocalc_kwargs(n_steps=2000)
    sol_isocalc = isocalc(**kwargs)
    sol_jax = isocalc_jax(**kwargs)

    for key in ("Matm", "N_H", "N_He", "N_D", "N_O", "N_C", "N_N", "N_S", "Rp", "Vpot"):
        assert sol_jax[key][-1] == pytest.approx(sol_isocalc[key][-1], rel=1e-2), key


def test_isocalc_isocalc_jax2_cross_check():
    """isocalc_jax2 counterpart of test_isocalc_isocalc_jax_cross_check, on the more realistic
    LHS 1140 b scenario (n_atmodeller=100 with n_steps=2000 forces 20 segments) - a structural
    check only, not a close numerical cross-check.

    Close agreement isn't achievable here: this combination of escape-dominated dynamics and
    periodic Atmodeller re-equilibration exhibits genuine sensitive dependence on tiny input
    differences (confirmed by hand - isocalc/isocalc_jax2 land on different trajectory branches
    that diverge by ~15-17% by the end, an effect that does *not* shrink with resolution: both
    isocalc and isocalc_jax2 are already independently well-converged to their own respective
    trajectories at n_steps=2000, e.g. re-running isocalc at n_steps=20000 changes its own final
    Matm by <0.01%). This mirrors the same "trajectories diverge by orders of magnitude between
    otherwise equivalent runs" sensitivity this module's docstring already documents for
    escape-dominated long runs (`n_atmodeller=0` case) - Atmodeller re-coupling just gives it
    more opportunities (each call boundary) to inject the tiny perturbations that trigger it.
    """
    kwargs = _sim_isocalc_kwargs(n_steps=2000, n_atmodeller=100)
    sol_isocalc = isocalc(**kwargs)
    sol_jax2 = isocalc_jax2(**kwargs)

    for key in ("Matm", "N_H", "N_He", "N_D", "N_O", "N_C", "N_N", "N_S", "Rp", "Vpot"):
        assert np.all(np.isfinite(sol_jax2[key]))
        assert sol_jax2[key][-1] > 0, key
        # Same order of magnitude, not close agreement - see docstring above.
        assert sol_jax2[key][-1] == pytest.approx(sol_isocalc[key][-1], rel=1.0), key

    final = sol_jax2["atmodeller_final"]
    assert np.all(np.isfinite(list(final.values())))


def _sim_integrator_setup(mass_loss_fraction=None):
    """The LHS 1140 b scenario set up for direct `IsocalcIntegrator` use (no Atmodeller), with the
    given `IsocalcOptions.mass_loss_fraction` (``None`` keeps the default). This scenario is
    escape-dominated (about 17% of the atmospheric mass is lost over the run), unlike
    test_pipeline's toy scenario.

    Returns:
        `(integrator, t_start, t_a, y0)`
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
    y0 = jnp.asarray(kwargs["isofate_species_abund"], dtype=float)

    return IsocalcIntegrator(parameters), jnp.asarray(t_start), t_a, y0


def _integrate_sim(mass_loss_fraction=None):
    """One `IsocalcIntegrator.integrate` call on the LHS 1140 b scenario."""
    integrator, t_start, t_a, y0 = _sim_integrator_setup(mass_loss_fraction)
    sol = integrator.integrate(t_start, t_a, y0)
    y_a, stop = sol.ys[0], IntegrationStop.from_solution(sol)

    return integrator.parameters, np.asarray(t_a), y0, y_a, stop


def _identity_hook(carry, t, y):
    """`on_mass_lost` hook that leaves the state unchanged."""
    return carry, y


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


def test_integrate_segments_identity_hook_matches_single_solve():
    """Plumbing check: restarting every 5% of mass loss with a hook that leaves the state unchanged
    must reproduce one uninterrupted integration."""
    integrator, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = integrator.parameters

    y_a, _, restarts, _ = integrate_segments(
        integrator, t_start, t_a, y0, on_mass_lost=_identity_hook
    )

    reference_integrator, _, _, _ = _sim_integrator_setup()
    y_reference = reference_integrator.integrate(t_start, t_a, y0).ys[0]

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


def test_integrate_segments_placeholder_hook():
    """The default placeholder hook removes 1% of every abundance at each restart, and the next
    segment continues from the reduced state."""
    integrator, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = integrator.parameters

    y_a, _, restarts, _ = integrate_segments(integrator, t_start, t_a, y0)
    y_identity, _, restarts_identity, _ = integrate_segments(
        integrator, t_start, t_a, y0, on_mass_lost=_identity_hook
    )

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


def test_integrate_segments_hook_and_floor():
    """The hook is called once per restart with its carried state threaded through, and a hook
    that removes almost all of the atmosphere ends the integration at the exhaustion floor."""
    integrator, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    parameters = integrator.parameters

    def counting_hook(carry, t, y):
        return carry + 1, y

    _, _, restarts, n_calls = integrate_segments(
        integrator, t_start, t_a, y0, on_mass_lost=counting_hook, carry=jnp.array(0)
    )
    assert int(n_calls) == int(restarts.count) >= 2

    def removing_hook(carry, t, y):
        return carry, 1e-7 * y

    y_a, _, restarts, _ = integrate_segments(
        integrator, t_start, t_a, y0, on_mass_lost=removing_hook
    )
    assert int(restarts.count) == 1
    i_next = int(np.searchsorted(np.asarray(t_a), float(restarts.t[0]), side="right"))
    np.testing.assert_allclose(
        y_a[i_next:], np.broadcast_to(restarts.y_after[0], y_a[i_next:].shape)
    )
    assert float(parameters.atmosphere_mass(restarts.y_after[0])) <= EXHAUSTION_FRACTION * float(
        parameters.atmosphere_mass(y0)
    )


def test_integrate_segments_max_segments():
    """Running out of segments is reported by `restarts.finished` (and raises only if Equinox
    errors are enabled, which isofate turns off by default)."""
    integrator, t_start, t_a, y0 = _sim_integrator_setup(0.05)

    _, _, restarts, _ = integrate_segments(integrator, t_start, t_a, y0, max_segments=2)
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
    integrator, t_start, t_a, y0 = _sim_integrator_setup(0.05)
    coupler = AtmodellerCoupler(integrator.parameters)
    state0, y_eq = coupler.reequilibrate(coupler.initial_state(), t_start, y0)

    def total(y_atm, y_int):
        return np.asarray(coupler.aggregate_D_into_H(jnp.asarray(y_atm) + jnp.asarray(y_int)))

    np.testing.assert_allclose(total(y_eq, state0.y_int), total(y0, 0.0), rtol=1e-10)

    _, _, restarts, _ = integrate_segments(
        integrator, t_start, t_a, y_eq, on_mass_lost=coupler.reequilibrate, carry=state0
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
    """isocalc_jax3 returns the same keys as isocalc (plus `t_atmodeller`), all finite, with the
    interior outputs piecewise constant between re-equilibrations."""
    kwargs = _jax3_kwargs(save_molecules=save_molecules)
    sol = isocalc_jax3(**kwargs)
    sol_isocalc = isocalc(**kwargs)

    assert set(sol) == set(sol_isocalc) | {"t_atmodeller"}
    for key, value in sol.items():
        if key != "atmodeller_final":
            assert np.all(np.isfinite(value)), key
    assert np.all(np.isfinite(list(sol["atmodeller_final"].values())))

    # The interior only changes at the output rows straddling a re-equilibration
    restart_rows = set(np.searchsorted(sol["time"], sol["t_atmodeller"][1:]).tolist())
    for symbol in ("H", "O", "C"):
        change_rows = set((np.nonzero(np.diff(sol[f"N_{symbol}_int"]))[0] + 1).tolist())
        assert change_rows <= restart_rows, symbol



@pytest.mark.parametrize("n_atmodeller", [0, 20])
def test_isocalc_jax3_euler_reproduces_isocalc(n_atmodeller):
    """isocalc_jax3(euler=True) marches with isocalc's own scheme (forward Euler clipped at zero,
    re-equilibrating every n_atmodeller steps on the shared schedule) via
    isofate.integrators.integrate_segments_euler, so every output agrees with isocalc to round-off."""
    kwargs = _sim_isocalc_kwargs(n_steps=200, n_atmodeller=n_atmodeller)
    sol_isocalc = isocalc(**kwargs)
    sol = isocalc_jax3(**kwargs, euler=True)

    for key, expected in sol_isocalc.items():
        if key == "atmodeller_final":
            continue
        np.testing.assert_allclose(sol[key], expected, rtol=1e-10, atol=0, err_msg=key)
