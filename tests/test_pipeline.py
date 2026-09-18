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

Values were re-pinned after `isojax.make_atmosphere_descent_jax` switched from a fixed-step
`Euler()` integrator (249 steps) to adaptive `Tsit5()`: the fixed-step scheme had real truncation
error (~1e-3 to ~3e-2 relative, benchmarked against a tight-tolerance reference), so this was an
accuracy fix, not a neutral refactor - trace-abundance and near-zero quantities (`N_S_atm`,
`log10dIW_1_bar`) shifted by more than rtol=1e-2/abs=1e-2 against the old pins even though the
underlying physics is now more accurate, not different.
"""

import numpy as np
import pytest

from isofate.atmodeller_coupler import AtmodellerCoupler, build_atmodeller, get_tracked_gas_species
from isofate.constants import const
from isofate.isofate_coupler import IsocalcOptions, isocalc
from isofate.system import Planet, Star, System


def _assert_finite(*values):
    for value in values:
        assert np.all(np.isfinite(value))


def test_get_tracked_gas_species_matches_atmodeller_order():
    """Order/labels/melt-name mapping must be derived from interior_atmosphere itself, not
    hand-typed - this pins that derivation against atmodeller's actual species order."""
    interior_atmosphere = build_atmodeller(5.6 * const.Me, surface_radius=1.5 * 6.371e6)
    tracked = get_tracked_gas_species(interior_atmosphere)

    labels = tuple(sp.label for sp in tracked)
    assert labels == ("H2", "H2O", "O2", "CO2", "CO", "CH4", "N2", "S2", "H2O4S", "SO2")

    by_label = {sp.label: sp for sp in tracked}
    # atmodeller canonicalizes SO2 to Hill notation "O2S_g", not "SO2_g" (read straight off
    # atmodeller's own ChemicalSpeciesData.name - no isofate-side conversion or override).
    assert by_label["SO2"].gas_name == "O2S_g"

    no_melt_reservoir = {"O2", "H2O4S", "SO2"}
    for sp in tracked:
        if sp.label in no_melt_reservoir:
            assert sp.melt_name is None
        else:
            assert sp.melt_name == f"{sp.label}_d"


def test_atmodeller_coupler_single_solve():
    """A single equilibrium solve for a realistic, H2-dominated/reducing composition."""
    Mp = 5.6 * const.Me
    radius_rocky = Planet(mass=Mp, period=1.0, f_atm=0.0).rocky_radius
    interior_atmosphere = build_atmodeller(Mp, surface_radius=radius_rocky)

    results, sol, mantle_iron_dict, _initial_guess = AtmodellerCoupler(
        Teq=900.0,
        Rp=1.5 * 6.371e6,
        mu=const.mu_H,
        melt_fraction=1.0,
        mantle_iron_dict=False,
        N_H_atm=5e46,
        N_He_atm=1e44,
        N_O_atm=1e43,
        N_C_atm=1e43,
        N_N_atm=1e42,
        N_S_atm=1e42,
        N_H_int=0.0,
        N_He_int=0.0,
        N_O_int=0.0,
        N_C_int=0.0,
        N_N_int=0.0,
        N_S_int=0.0,
        interior_atmosphere=interior_atmosphere,
    )

    _assert_finite(*results.values())
    assert mantle_iron_dict is False

    # Cross-checked by hand against ../IsoFATE_main (atmodeller 0.9.1) to ~3e-5 relative; re-pinned
    # for the Tsit5/adaptive-step make_atmosphere_descent_jax (see module docstring)
    expected = {
        "N_H_atm": 1.841429610047864e44,
        "N_H_int": 4.981878379760501e46,
        "N_He_atm": 6.976600095850282e43,
        "N_He_int": 3.023394907400085e43,
        "N_O_atm": 1.4204294868939488e35,
        "N_O_int": 9.999746723543633e42,
        "N_C_atm": 9.8528692727472e42,
        "N_C_int": 1.4734720017612394e41,
        "N_N_atm": 8.230705125763134e19,
        "N_N_int": 1.000021204133477e42,
        "N_S_atm": 475639.7427271265,
        "N_S_int": 1.0000062373693309e42,
        "M_atm": 9.684341225310362e17,
        "T_surface": 689.6894914051486,
        "T_surface_atmod": 689.6894914051486,
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


def _toy_isocalc_kwargs(**option_overrides):
    """`option_overrides` are forwarded to `IsocalcOptions`; n_steps/n_atmodeller are kept small
    so this stays a fast, non-escape-dominated test scenario."""
    return dict(
        system=_toy_system(),
        F0=500.0,
        time=1e6,
        # isofate_species_abund ordered per isofate.species.SYMBOLS: (H, He, D, O, C, N, S)
        isofate_species_abund=(1e45, 1e44, 1e41, 1.5e45, 1e44, 1e43, 1e43),
        options=IsocalcOptions(n_steps=20, n_atmodeller=5, **option_overrides),
    )


def test_isocalc_regression():
    """Short, non-escape-dominated run; cross-checked against ../IsoFATE_main to ~1e-4 relative."""
    sol = isocalc(**_toy_isocalc_kwargs())

    _assert_finite(sol["Matm"], sol["N_H"], sol["N_O_int"], sol["N_C_int"])

    assert sol["Matm"][-1] == pytest.approx(2.8852458471813018e19, rel=1e-2)
    assert sol["N_H"][-1] == pytest.approx(4.316098051849231e38, rel=1e-2)
    assert sol["N_O_int"][-1] == pytest.approx(4.888599902382618e44, rel=1e-2)
    assert sol["N_C_int"][-1] == pytest.approx(2.0517344351712084e43, rel=1e-2)

    final = sol["atmodeller_final"]
    assert final["O2_fugacity"] == pytest.approx(4.415226249622124, rel=1e-2)
    assert final["log10dIW_1_bar"] == pytest.approx(0.004890098041127358, abs=1e-2)
    assert final["H2O_atm"] == pytest.approx(181146406710961.88, rel=1e-2)
    assert final["H2O_mantle"] == pytest.approx(7.670751470352576e20, rel=1e-2)


@pytest.mark.parametrize(
    "mantle_iron_type,expected_N_O_int",
    [
        ("dynamic", 1.41002544208791e45),
        ("static", 1.41002544208791e45),
    ],
)
def test_isocalc_mantle_iron_dict(mantle_iron_type, expected_N_O_int):
    """Regression-only pin (no reference to check against): ../IsoFATE_main crashes on this path
    with an UnboundLocalError-adjacent shape bug (an un-indexed array leaks into the escape
    integration once mass_Fe2 != 0); the v2preview port fixed that incidentally by scalarizing
    every atmodeller output access. See atmodeller_coupler.py for details.
    """
    mantle_iron_dict = {"type": mantle_iron_type, "Fe_mass_fraction": 0.06}
    sol = isocalc(**_toy_isocalc_kwargs(mantle_iron_dict=mantle_iron_dict, save_molecules=True))

    _assert_finite(sol["Matm"], sol["N_O_int"], sol["n_H2O_a"])
    assert sol["N_O_int"][-1] == pytest.approx(expected_N_O_int, rel=1e-2)


def test_isocalc_save_molecules():
    sol = isocalc(**_toy_isocalc_kwargs(save_molecules=True))

    for key in ("n_H2O_a", "n_H2_a", "n_O2_a", "n_CO2_a", "n_CO_a", "n_CH4_a", "n_N2_a", "n_S2_a"):
        assert key in sol
        _assert_finite(sol[key])
