# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression anchor for isofate/sim.py's driver scenario (LHS 1140 b), pinned separately from
tests/test_pipeline.py's synthetic toy scenario. sim.py itself can't be imported directly (it's a
plotting script with top-level side effects, including a blocking `plt.show()`), so this
reconstructs its isocalc() inputs directly - kept in sync with sim.py by hand.

n_atmodeller=0 and dynamic_phi=False here match sim.py's current values (temporarily simplified
while isocalc is decoupled from Atmodeller/refactored towards a JAX driver - see the FIXME/TODO
comments in sim.py); update both together if sim.py's parameters change.
"""

import pytest

from isofate.constants import const
from isofate.escape import XUVEscape
from isofate.isofate_coupler import isocalc, isocalc_jax
from isofate.options import IsocalcOptions
from isofate.parameters import Parameters
from isofate.presets import LHS1140b, LHS1140Star
from isofate.system import System


def _sim_isocalc_kwargs(n_steps=int(1e5)):
    """Reconstructs isofate/sim.py's isocalc() call, as of the current version of that script.

    `n_steps` defaults to sim.py's own value but can be overridden (e.g. by the isocalc/
    isocalc_jax cross-check below, which doesn't need full resolution to confirm agreement).
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
    n_atmodeller = 0
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
    mu_avg = (
        N_H * const.mu_H
        + N_He * const.mu_He
        + N_D * const.mu_D
        + N_O * const.mu_O
        + N_C * const.mu_C
        + N_N * const.mu_N
        + N_S * const.mu_S
    ) / (N_H + N_He + N_D + N_O + N_C + N_N + N_S)

    escape = XUVEscape(
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
        mu=mu_avg,
        n_steps=n_steps,
        t0=t0,
        thermal=thermal,
        n_atmodeller=n_atmodeller,
        save_molecules=save_molecules,
        mantle_iron=mantle_iron,
        dynamic_phi=dynamic_phi,
    )
    parameters = Parameters(system, escape, isocalc_options=options)
    return dict(
        parameters=parameters,
        F0=F0,
        time=time,
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
    """
    sol = isocalc(**_sim_isocalc_kwargs())

    assert sol["Matm"][-1] == pytest.approx(2.7847119860674774e23, rel=1e-6)
    assert sol["N_H"][-1] == pytest.approx(1.109756853479713e50, rel=1e-6)
    assert sol["N_He"][-1] == pytest.approx(1.341472117896385e49, rel=1e-6)
    assert sol["N_D"][-1] == pytest.approx(2.3455560513061053e45, rel=1e-6)
    assert sol["N_O"][-1] == pytest.approx(8.353266024867263e46, rel=1e-6)
    assert sol["N_C"][-1] == pytest.approx(4.168490941832542e46, rel=1e-6)
    assert sol["N_N"][-1] == pytest.approx(1.5953265918420376e46, rel=1e-6)
    assert sol["N_S"][-1] == pytest.approx(2.5955614470233587e45, rel=1e-6)
    assert sol["Rp"][-1] == pytest.approx(13824654.981228502, rel=1e-6)
    assert sol["Vpot"][-1] == pytest.approx(153883574.58031628, rel=1e-6)


def test_isocalc_isocalc_jax_cross_check():
    """isocalc and isocalc_jax should agree closely on the same scenario: both now derive
    M_atm/f_atm fresh from y (see isocalc's and _integrate_isocalc_jax's docstrings), so the only
    remaining difference is the integration scheme itself - isocalc's fixed-step Euler loop vs.
    isocalc_jax's adaptive Tsit5 (diffrax). They also now share the exact same call signature
    (`parameters`, `F0`, `time`, `isofate_species_abund`), so the same kwargs work for both.

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
