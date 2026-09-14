"""Regression anchor for the isofate <-> atmodeller coupling (see isofate/sim.py for the
interactive/plotting driver this is derived from).

These pin the current (atmodeller v2preview) behavior of AtmodellerCoupler/isocalc so future
refactoring has something concrete to check against. isocalc() now requires a `system=System(...)`
argument (Mstar/d/T/Fp are derived from it; only F0 stays a separate, independent input - see
isofate_coupler.py). The toy scenario below was re-cross-checked, by hand, against
../IsoFATE_main running the pre-v2 atmodeller API (0.9.1) with the equivalent Mp/f_atm/Mstar/F0/
Fp/T/d passed directly: the single-solve values agree to ~1e-4 relative, and the short isocalc()
run agrees to ~1e-4 relative (small residual differences are expected from atmodeller's own
solver/thermodynamic-data changes between versions, not from this coupling code). Longer,
escape-dominated runs are NOT suitable for tight value-pinning: with as few as ~50 timesteps, the
explicit time integration is sensitive enough to tiny per-step differences that trajectories
diverge by orders of magnitude between otherwise equivalent runs, so only structural/invariant
checks are used for those.

Tolerances here are deliberately loose (rtol=1e-2) relative to the ~1e-4 cross-check agreement, to
avoid false failures from ordinary atmodeller version bumps while still catching real regressions
(wrong species/element mapping, broken unit conversions, wrong dict keys, crashes).
"""
import numpy as np
import pytest

from isofate.atmodeller_coupler import AtmodellerCoupler, build_atmodeller
from isofate.constants import const
from isofate.isofate_coupler import isocalc
from isofate.system import Planet, Star, System


def _assert_finite(*values):
    for value in values:
        assert np.all(np.isfinite(value))


def test_atmodeller_coupler_single_solve():
    """A single equilibrium solve for a realistic, H2-dominated/reducing composition."""
    Mp = 5.6 * const.Me
    interior_atmosphere = build_atmodeller(Mp)

    results, sol, mantle_iron_dict = AtmodellerCoupler(
        Teq=900.0, Mp=Mp, Rp=1.5 * 6.371e6, mu=const.mu_H, melt_fraction=1.0, mantle_iron_dict=False,
        N_H_atm=5e46, N_He_atm=1e44, N_O_atm=1e43, N_C_atm=1e43, N_N_atm=1e42, N_S_atm=1e42,
        N_H_int=0.0, N_He_int=0.0, N_O_int=0.0, N_C_int=0.0, N_N_int=0.0, N_S_int=0.0,
        interior_atmosphere=interior_atmosphere,
    )

    _assert_finite(*results.values())
    assert mantle_iron_dict is False

    # Cross-checked by hand against ../IsoFATE_main (atmodeller 0.9.1) to ~3e-5 relative
    expected = {
        'N_H_atm': 1.8414296100483874e+44,
        'N_H_int': 4.981878379760501e+46,
        'N_He_atm': 6.976600095847358e+43,
        'N_He_int': 3.0233949074030287e+43,
        'N_O_atm': 1.420365335653906e+35,
        'N_O_int': 9.999746723550029e+42,
        'N_C_atm': 9.852869272804816e+42,
        'N_C_int': 1.4734720011859115e+41,
        'N_N_atm': 8.276480200829949e+19,
        'N_N_int': 1.000021204133477e+42,
        'N_S_atm': 469002.46607402986,
        'N_S_int': 1.0000062373693309e+42,
        'M_atm': 9.6843412253191e+17,
        'T_surface': 689.3859493619807,
        'T_surface_atmod': 689.3859493619807,
    }
    for key, value in expected.items():
        assert results[key] == pytest.approx(value, rel=1e-2), key


def _toy_system():
    """A close-in, sun-like-star scenario. Not physically tuned to anything in particular - it
    only exists to give isocalc() a small, fast, non-escape-dominated System to run."""
    star = Star(radius=const.Rs, mass=1.989e30, temperature=5000)
    period = star.period_for_semi_major_axis(0.05 * 1.496e11)  # ~0.05 au orbital distance
    planet = Planet(mass=5.0 * const.Me, period=period, f_atm=0.01)
    return System(star=star, planet=planet)


def _toy_isocalc_kwargs(**overrides):
    kwargs = dict(
        system=_toy_system(), F0=500.0,
        time=1e6, n_steps=20, n_atmodeller=5,
        N_H=1e45, N_D=1e41, N_He=1e44, N_O=1.5e45, N_C=1e44, N_N=1e43, N_S=1e43,
    )
    kwargs.update(overrides)
    return kwargs


def test_isocalc_regression():
    """Short, non-escape-dominated run; cross-checked against ../IsoFATE_main to ~1e-4 relative."""
    sol = isocalc(**_toy_isocalc_kwargs())

    _assert_finite(sol['Matm'], sol['N_H'], sol['N_O_int'], sol['N_C_int'])

    assert sol['Matm'][-1] == pytest.approx(2.883283328826164e+19, rel=1e-2)
    assert sol['N_H'][-1] == pytest.approx(4.324375408000459e+38, rel=1e-2)
    assert sol['N_O_int'][-1] == pytest.approx(4.89707843982286e+44, rel=1e-2)
    assert sol['N_C_int'][-1] == pytest.approx(2.052401665343788e+43, rel=1e-2)

    final = sol['atmodeller_final']
    assert final['O2_fugacity'] == pytest.approx(4.409428671983565, rel=1e-2)
    assert final['log10dIW_1_bar'] == pytest.approx(0.03379258263388761, abs=1e-2)
    assert final['H2O_atm'] == pytest.approx(184845395611210.97, rel=1e-2)
    assert final['H2O_mantle'] == pytest.approx(7.681862036973651e+20, rel=1e-2)


@pytest.mark.parametrize("mantle_iron_type,expected_N_O_int", [
    ("dynamic", 1.4096757259007388e+45),
    ("static", 1.4096757259007388e+45),
])
def test_isocalc_mantle_iron_dict(mantle_iron_type, expected_N_O_int):
    """Regression-only pin (no reference to check against): ../IsoFATE_main crashes on this path
    with an UnboundLocalError-adjacent shape bug (an un-indexed array leaks into the escape
    integration once mass_Fe2 != 0); the v2preview port fixed that incidentally by scalarizing
    every atmodeller output access. See atmodeller_coupler.py for details.
    """
    mantle_iron_dict = {'type': mantle_iron_type, 'Fe_mass_fraction': 0.06}
    sol = isocalc(**_toy_isocalc_kwargs(mantle_iron_dict=mantle_iron_dict, save_molecules=True))

    _assert_finite(sol['Matm'], sol['N_O_int'], sol['n_H2O_a'])
    assert sol['N_O_int'][-1] == pytest.approx(expected_N_O_int, rel=1e-2)


def test_isocalc_save_molecules():
    sol = isocalc(**_toy_isocalc_kwargs(save_molecules=True))

    for key in ('n_H2O_a', 'n_H2_a', 'n_O2_a', 'n_CO2_a', 'n_CO_a', 'n_CH4_a', 'n_N2_a', 'n_S2_a'):
        assert key in sol
        _assert_finite(sol[key])
