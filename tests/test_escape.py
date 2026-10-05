# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression anchor for isofate.escape's EscapeMechanism classes.

The five pinned values below were captured directly from the pre-refactor `compute_mass_flux`
dispatch (isofate_coupler.py, since removed) on the same EscapeState inputs, for every
mechanism/RR combination it supported - "XUV" (RR=True and RR=False), "CPML", "phi kill", and
"XUV+CPML". Only the "XUV" (RR=True) path is exercised by tests/test_pipeline.py's isocalc()
regression tests (via the default options), so these pins are what actually confirm the other
three mechanisms still behave exactly as before the class-hierarchy refactor.

The CPML/"XUV+CPML" pins were re-pinned after `EscapeState.A` was removed (`phiE_CP` now derives
area from `radius_p` internally, matching `phi_kill`'s existing pattern) - `STATE.A` had been a
manually-rounded `5.3e14`, not the exact `4*pi*radius_p**2` (~5.309e14), so CPML's mass flux
shifts by ~0.2% relative. A precision fix, not a regression.

`EscapeState` reads planet mass, equilibrium temperature and orbital distance from a `System`.
The toy `SYSTEM` below reproduces the originally pinned inputs (Mp=5e24 kg, T=900 K,
d=7.5e9 m) by choosing the planet period from the orbital distance and setting the stellar
luminosity so that the zero-albedo equilibrium temperature, (L / (16 pi d^2 sigma))^(1/4), is
900 K - so the pins are unchanged.
"""

import numpy as np
import pytest

from isofate.constants import const
from isofate.escape.mechanisms import (
    CombinedEscape,
    CPMLEscape,
    EscapeState,
    Fxuv,
    PhiKillEscape,
    XUVEscape,
    phi_E,
    phi_kill,
    phiE_CP,
    phi_RR,
)
from isofate.system import Planet, Star, System

F0 = 500.0

_SEMI_MAJOR_AXIS = 7.5e9  # [m]
_EQUILIBRIUM_TEMPERATURE = 900.0  # [K]
_STAR = Star(
    radius=const.Rs,
    mass=1.989e30,
    temperature=5000,
    luminosity=16 * np.pi * _SEMI_MAJOR_AXIS**2 * const.sbc * _EQUILIBRIUM_TEMPERATURE**4,
)
SYSTEM = System(
    star=_STAR,
    planet=Planet(mass=5e24, period=_STAR.period_for_semi_major_axis(_SEMI_MAJOR_AXIS)),
)

STATE = EscapeState(
    system=SYSTEM,
    escape_radius=6.5e6,
    gravitational_potential=5e7,
    mean_mu=3.3e-27,
    convective_envelope_thickness=1e5,
    atmosphere_mass_fraction=0.01,
    t_current=3e13,
    t_end=1.5e17,
)


def test_xuv_escape_matches_old_mechanism_dispatch():
    assert XUVEscape(F0=F0, RR=True).compute_mass_flux(STATE) == pytest.approx(
        1.6720024065164486e-08
    )
    assert XUVEscape(F0=F0, RR=False).compute_mass_flux(STATE) == pytest.approx(3.75e-07)


def test_cpml_escape_matches_old_mechanism_dispatch():
    assert CPMLEscape().compute_mass_flux(STATE) == pytest.approx(2.1740389601731796e-07)


def test_phi_kill_escape_matches_old_mechanism_dispatch():
    assert PhiKillEscape().compute_mass_flux(STATE) == pytest.approx(6.279557414121694e-09)


def test_combined_escape_matches_old_xuv_plus_cpml_dispatch():
    combined = CombinedEscape((XUVEscape(F0=F0, RR=True), CPMLEscape()))
    assert combined.compute_mass_flux(STATE) == pytest.approx(2.3412392008248244e-07)


def test_combined_escape_sums_components():
    """Direct test of the composition-over-inheritance design: CombinedEscape's result must
    equal the sum of its components' individually-computed results, not just match a pinned
    number."""
    xuv = XUVEscape(F0=F0, RR=True)
    cpml = CPMLEscape()
    combined = CombinedEscape((xuv, cpml))
    expected = xuv.compute_mass_flux(STATE) + cpml.compute_mass_flux(STATE)
    assert combined.compute_mass_flux(STATE) == pytest.approx(expected)


def test_xuv_escape_wraps_phi_e_and_phi_rr():
    escape = XUVEscape(F0=F0, RR=True)
    phi_energy_limited = phi_E(
        STATE.t_current,
        STATE.gravitational_potential,
        STATE.system.semi_major_axis,
        F0,
        eps=escape.eps,
        t0=escape.t0,
    )
    phi_recombination_limited = phi_RR(
        STATE.escape_radius,
        STATE.system.planet.mass,
        STATE.system.equilibrium_temperature,
        STATE.t_current,
        F0,
        t0=escape.t0,
    )
    assert escape.compute_mass_flux(STATE) == pytest.approx(
        min(phi_recombination_limited, phi_energy_limited)
    )

    escape_no_rr = XUVEscape(F0=F0, RR=False)
    assert escape_no_rr.compute_mass_flux(STATE) == pytest.approx(
        phi_E(
            STATE.t_current,
            STATE.gravitational_potential,
            STATE.system.semi_major_axis,
            F0,
            eps=escape_no_rr.eps,
            t0=escape_no_rr.t0,
        )
    )


def test_cpml_escape_wraps_phie_cp():
    escape = CPMLEscape()
    expected = phiE_CP(
        STATE.system.equilibrium_temperature,
        STATE.system.planet.mass,
        escape.rho_rcb,
        escape.eps,
        STATE.gravitational_potential,
        STATE.escape_radius,
        STATE.mean_mu,
        STATE.convective_envelope_thickness,
    )
    assert escape.compute_mass_flux(STATE) == pytest.approx(expected)


def test_phi_kill_escape_wraps_phi_kill():
    escape = PhiKillEscape()
    expected = phi_kill(
        STATE.system.planet.mass * STATE.atmosphere_mass_fraction,
        STATE.escape_radius,
        STATE.t_end - STATE.t_current,
    )
    assert escape.compute_mass_flux(STATE) == pytest.approx(expected)


def test_fxuv_still_importable_from_escape():
    # Fxuv itself is separately covered in depth by tests/test_Fxuv.py; this just confirms the
    # relocation didn't change its default behavior.
    assert Fxuv(3e13, 500.0) > 0
