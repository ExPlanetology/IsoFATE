# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for isofate.output: the Output of isocalc, its jittable derived quantities, its exports,
and its one-to-one link with the separate Atmodeller output."""

import equinox as eqx
import numpy as np
import pytest
from test_sim import _sim_isocalc_kwargs

from isofate.constants import const
from isofate.isofate_coupler import isocalc
from isofate.output import read_pickle
from isofate.species import SYMBOLS


@pytest.fixture(scope="module")
def output():
    """A short coupled LHS 1140 b run (Euler, re-equilibrating every 20 of 200 steps)."""
    return isocalc(**_sim_isocalc_kwargs(n_steps=200, n_atmodeller=20, euler=True))


def test_to_dict_is_jittable(output):
    """to_dict (JAX arrays) runs under jit and matches the eager result."""
    eager = output.to_dict()
    jitted = eqx.filter_jit(lambda o: o.to_dict())(output)

    assert set(jitted) == set(eager)
    for key, value in eager.items():
        np.testing.assert_array_equal(np.asarray(jitted[key]), np.asarray(value), err_msg=key)


def test_equilibration_index(output):
    """Each output time maps onto the latest equilibration at or before it, so the interior is
    constant between equilibrations and changes at the rows of the equilibration times."""
    count = int(output.equilibrations.count)
    index = np.asarray(output.equilibration_index())
    t_eq = np.asarray(output.equilibrations.t[:count])

    assert index[0] == 0
    assert index[-1] == count - 1
    assert np.all(np.diff(index) >= 0)
    # the row at each equilibration time (after the initial one) takes that equilibration
    rows = np.searchsorted(np.asarray(output.t_a), t_eq[1:] - 1.0)
    np.testing.assert_array_equal(index[rows], np.arange(1, count))


def test_atmodeller_output_matches_equilibrations(output):
    """Batch row i of the Atmodeller output is equilibration i: the gas and melt element amounts
    it reports are the atmosphere and interior after that equilibration (D counted with H)."""
    count = int(output.equilibrations.count)
    assert output.atmodeller.parameters.batch_size == count
    atmodeller = output.atmodeller.to_dict(output_format="elements_species", to_numpy=True)

    y_after = np.asarray(output.equilibrations.y_after[:count])
    y_int = np.asarray(output.equilibrations.y_int[:count])
    for symbol in ("H", "He", "O", "C", "N"):
        i = SYMBOLS.index(symbol)
        gas, melt = y_after[:, i], y_int[:, i]
        if symbol == "H":
            gas, melt = gas + y_after[:, SYMBOLS.index("D")], melt + y_int[:, SYMBOLS.index("D")]
        np.testing.assert_allclose(
            np.ravel(atmodeller[symbol]["gas"]["number_moles"]) * const.avogadro, gas, rtol=1e-10
        )
        np.testing.assert_allclose(
            np.ravel(atmodeller[symbol]["silicate_melt"]["number_moles"]) * const.avogadro,
            melt,
            rtol=1e-10,
        )


def test_to_dataframes(output):
    """One time-series row per output time, one equilibration row per equilibration."""
    dataframes = output.to_dataframes()

    assert len(dataframes["time_series"]) == output.t_a.shape[0]
    assert "N_H" in dataframes["time_series"]
    assert len(dataframes["equilibrations"]) == int(output.equilibrations.count)
    assert {"t", "N_H_after", "N_H_int", "mass_constraint_H"} <= set(dataframes["equilibrations"])


def test_pickle_round_trip(output, tmp_path):
    """to_pickle writes IsoFATE's DataFrames and, separately, Atmodeller's; read_pickle reads them
    back."""
    path = tmp_path / "output.pkl"
    output.to_pickle(path)
    content = read_pickle(path)

    assert set(content) == {"time_series", "equilibrations", "atmodeller"}
    expected = output.to_dataframes()
    for name in ("time_series", "equilibrations"):
        np.testing.assert_array_equal(content[name].to_numpy(), expected[name].to_numpy())
    assert set(content["atmodeller"]) == set(output.atmodeller.to_dataframes())


def test_output_without_coupling():
    """Without Atmodeller there are no equilibrations, no Atmodeller output and an empty
    interior."""
    output = isocalc(**_sim_isocalc_kwargs(n_steps=50, n_atmodeller=0, euler=True))
    sol = output.to_dict(to_numpy=True)

    assert output.equilibrations is None
    assert output.atmodeller is None
    assert np.all(sol["N_H_int"] == 0)
    assert len(sol["t_atmodeller"]) == 0
    assert set(output.to_dataframes()) == {"time_series"}
