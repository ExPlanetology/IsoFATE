"""Regression anchor for the isofate <-> atmodeller coupling (see isofate/sim.py for the
interactive/plotting driver this is derived from).

These pin the current (atmodeller v2preview) behavior of AtmodellerCoupler and isocalc_jax3 so
future refactoring has something concrete to check against. The toy scenario below was
re-cross-checked (with the legacy isocalc, since retired; isocalc_jax3(euler=True) reproduces it to
round-off), by hand, against
../IsoFATE_main running the pre-v2 atmodeller API (0.9.1) with the equivalent Mp/f_atm/Mstar/F0/
Fp/T/d passed directly: the single-solve values agree to ~1e-4 relative, and the short run
agrees to ~1e-4 relative (small residual differences are expected from atmodeller's own
solver/thermodynamic-data changes between versions, not from this coupling code). Longer,
escape-dominated runs are NOT suitable for tight value-pinning: with as few as ~50 timesteps, the
explicit time integration is sensitive enough to tiny per-step differences that trajectories
diverge by orders of magnitude between otherwise equivalent runs, so only structural/invariant
checks are used for those.

Tolerances here are deliberately loose (rtol=1e-2) relative to the ~1e-4 cross-check agreement, to
avoid false failures from ordinary atmodeller version bumps while still catching real regressions
(wrong species/element mapping, broken unit conversions, wrong dict keys, crashes).

Values were re-pinned after the atmosphere descent (now `AtmosphereModel.make_atmosphere_descent`)
switched from a fixed-step
`Euler()` integrator (249 steps) to adaptive `Tsit5()`: the fixed-step scheme had real truncation
error (~1e-3 to ~3e-2 relative, benchmarked against a tight-tolerance reference), so this was an
accuracy fix, not a neutral refactor - trace-abundance and near-zero quantities (`N_S_atm`,
`log10dIW_1_bar`) shifted by more than rtol=1e-2/abs=1e-2 against the old pins even though the
underlying physics is now more accurate, not different.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from isofate.atmodeller_coupler import AtmodellerCoupler
from isofate.constants import const
from isofate.escape.mechanisms import XUVEscape
from isofate.initial_condition import AbundanceInitialCondition
from isofate.isofate_coupler import isocalc_jax3
from isofate.parameters import IsocalcOptions, Parameters
from isofate.species import DEFAULT_SPECIES
from isofate.system import Planet, Star, System


def _assert_finite(*values):
    for value in values:
        assert np.all(np.isfinite(value))


def _single_solve_parameters():
    """Parameters for a single Atmodeller solve: Mp = 5.6 Earth masses, an equilibrium temperature
    of 900 K (the stellar luminosity is chosen so that the zero-albedo equilibrium temperature,
    (L / (16 pi d^2 sigma))^(1/4), is 900 K) and a fully molten mantle."""
    semi_major_axis = 7.5e9  # [m]
    star = Star(
        radius=const.Rs,
        mass=1.989e30,
        temperature=5000,
        luminosity=16 * np.pi * semi_major_axis**2 * const.sbc * 900.0**4,
    )
    planet = Planet(
        mass=5.6 * const.Me,
        period=star.period_for_semi_major_axis(semi_major_axis),
        mantle_melt_fraction=1.0,
    )
    return Parameters(
        System(star=star, planet=planet),
        escape_mechanism=XUVEscape(F0=500.0),
        initial_condition=AbundanceInitialCondition((0.0,) * 7),
    )


def test_tracked_gas_species_matches_atmodeller_order():
    """Order/labels/melt-name mapping must be derived from the Atmodeller model itself, not
    hand-typed - this pins that derivation against atmodeller's actual species order."""
    tracked = AtmodellerCoupler(_single_solve_parameters()).tracked_species

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
    parameters = _single_solve_parameters()
    coupler = AtmodellerCoupler(parameters)
    # ordered per isofate.species.SYMBOLS: (H, He, D, O, C, N, S)
    y = jnp.array([5e46, 1e44, 0.0, 1e43, 1e43, 1e42, 1e42])
    result = coupler.run(
        1.5 * 6.371e6, DEFAULT_SPECIES.mass_by_symbol["H"], y, jnp.zeros(7), full_output=False
    )
    assert bool(result.success)

    symbols = ("H", "He", "O", "C", "N", "S")
    results = {
        **{f"N_{s}_atm": float(result.y[DEFAULT_SPECIES.index(s)]) for s in symbols},
        **{f"N_{s}_int": float(result.y_int[DEFAULT_SPECIES.index(s)]) for s in symbols},
        "M_atm": float(result.atmosphere_mass),
        "T_surface": float(result.surface_temperature),
        "T_surface_atmod": float(result.surface_temperature_atmodeller),
    }

    _assert_finite(*results.values())

    # Cross-checked by hand against ../IsoFATE_main (atmodeller 0.9.1) to ~3e-5 relative; re-pinned
    # for the Tsit5/adaptive-step atmosphere descent (see module docstring)
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
    only exists to give the driver a small, fast, non-escape-dominated System to run."""
    star = Star(radius=const.Rs, mass=1.989e30, temperature=5000)
    period = star.period_for_semi_major_axis(0.05 * 1.496e11)  # ~0.05 au orbital distance
    planet = Planet(mass=5.0 * const.Me, period=period)
    return System(star=star, planet=planet)


def _toy_isocalc_kwargs(**option_overrides):
    """`option_overrides` are forwarded to `IsocalcOptions`; n_steps/n_atmodeller are kept small
    so this stays a fast, non-escape-dominated test scenario. dynamic_phi is off (static H/He
    pair), as when the regression values were pinned.

    Note that t_end equals the default `IsocalcOptions.t_start` (1e6 yr), so no time passes: every
    output row is at the same time and the run is only repeated Atmodeller solves. This exercises
    the coupling, not the time integration (see tests/test_sim.py for that)."""
    option_overrides = {"dynamic_phi": False, **option_overrides}
    options = IsocalcOptions(n_steps=20, n_atmodeller=5, **option_overrides)
    parameters = Parameters(
        _toy_system(),
        escape_mechanism=XUVEscape(F0=500.0),
        # ordered per isofate.species.SYMBOLS: (H, He, D, O, C, N, S)
        initial_condition=AbundanceInitialCondition((1e45, 1e44, 1e41, 1.5e45, 1e44, 1e43, 1e43)),
        isocalc_options=options,
    )
    return dict(parameters=parameters, t_end=1e6)


def test_isocalc_jax3_toy_regression():
    """Short, non-escape-dominated run with isocalc_jax3(euler=True); the legacy isocalc's values,
    cross-checked against ../IsoFATE_main to ~1e-4 relative."""
    sol = isocalc_jax3(**_toy_isocalc_kwargs(), euler=True)

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


def test_isocalc_jax3_save_molecules():
    sol = isocalc_jax3(**_toy_isocalc_kwargs(save_molecules=True), euler=True)

    for key in ("n_H2O_a", "n_H2_a", "n_O2_a", "n_CO2_a", "n_CO_a", "n_CH4_a", "n_N2_a", "n_S2_a"):
        assert key in sol
        _assert_finite(sol[key])
