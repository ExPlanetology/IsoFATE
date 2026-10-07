# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output of an IsoFATE run (`isofate.isofate_coupler.isocalc`).

`Output` holds only the raw results of the time integration - the output times, the atmospheric
abundances and the record of the Atmodeller equilibrations (`Equilibrations`) - and derives
everything else on demand, following atmodeller's own `Output`. The derived quantities
(`diagnostics`, `interior`, `to_dict`, ...) are pure JAX and can be used under `jax.jit`; the
conversions to NumPy, pandas and pickle are post-processing and cannot.

The complete Atmodeller output is kept separately, as atmodeller's own `Output`
(`Output.atmodeller`), with one batch row per equilibration: row ``i`` of `Output.equilibrations`
is batch row ``i`` of `Output.atmodeller`, and `Output.equilibration_index` maps each output time
onto those rows.
"""

import pickle
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from atmodeller.output import Output as AtmodellerOutput
from jax import Array

from isofate.integrators import AtmosphereState, IsocalcModel
from isofate.parameters import Parameters
from isofate.species import SYMBOLS

_COUPLING_ELEMENTS: tuple[str, ...] = tuple(symbol for symbol in SYMBOLS if symbol != "D")
"""Elements of the Atmodeller mass constraints, as in `isofate.atmodeller_coupler`."""


class Equilibrations(eqx.Module):
    """The Atmodeller equilibrations of a run, in fixed-size buffers so that they can be used under
    `jax.jit`.

    The rows are in time order - the initial equilibration, those during the integration and one
    at the end time if it is on the schedule - and only the first `count` are filled; the rest
    are NaN.

    Args:
        t: Equilibration times [s]
        y_before: Atmospheric abundances passed to the solve [atoms], ordered per
            `isofate.species.SYMBOLS`
        y_after: Atmospheric abundances after the solve [atoms]
        y_int: Interior (dissolved) abundances after the solve [atoms]
        surface_temperature: Surface temperature from the atmosphere descent [K]
        surface_temperature_atmodeller: Surface temperature used by Atmodeller [K]
        mantle_melt_fraction: Mantle melt fraction used by Atmodeller [ndim]
        mass_constraints: Element mass constraints given to Atmodeller [kg], ordered per the
            coupling elements (H, He, O, C, N, S)
        solution: Converged Atmodeller solution array
        count: Number of equilibrations
    """

    t: Array
    y_before: Array
    y_after: Array
    y_int: Array
    surface_temperature: Array
    surface_temperature_atmodeller: Array
    mantle_melt_fraction: Array
    mass_constraints: Array
    solution: Array
    count: Array

    @property
    def mask(self) -> Array:
        """True for the filled rows."""
        return jnp.arange(self.t.shape[0]) < self.count


class Output(eqx.Module):
    """Output of an IsoFATE run.

    Args:
        parameters: Parameters of the run
        t_start: Start time [s]
        t_a: Output times [s]; row j is the state after j + 1 steps of ``(t_a[-1] - t_start) /
            len(t_a)``
        y_a: Atmospheric abundances at each output time [atoms], ordered per
            `isofate.species.SYMBOLS`
        equilibrations: The Atmodeller equilibrations, or ``None`` without the coupling
        atmodeller: The complete Atmodeller output, one batch row per equilibration (see the
            module docstring), or ``None`` without the coupling or equilibrations
    """

    parameters: Parameters
    t_start: Array
    t_a: Array
    y_a: Array
    equilibrations: Equilibrations | None = None
    atmodeller: AtmodellerOutput | None = None

    @property
    def delta_t(self) -> Array:
        """Time step [s]."""
        return (self.t_a[-1] - self.t_start) / self.t_a.shape[0]

    def diagnostics(self) -> AtmosphereState:
        """Everything derivable from each output time and atmosphere (radii, fluxes, masses, ...).

        Returns:
            The atmosphere state, batched over the output times
        """
        return IsocalcModel(self.parameters).diagnostics(self.t_a, self.y_a)

    def equilibration_index(self) -> Array:
        """For each output time, the row of the latest equilibration at or before it - the link
        between the output times and `equilibrations`/`atmodeller`.

        Equilibration times coincide with output times only to the last bit, so they are matched
        within half an output spacing. Without equilibrations, every index is 0.

        Returns:
            Equilibration row for each output time
        """
        if self.equilibrations is None:
            return jnp.zeros(self.t_a.shape[0], dtype=int)
        # Unfilled rows never match
        t_eq: Array = jnp.where(self.equilibrations.mask, self.equilibrations.t, jnp.inf)
        half_gap: Array = 0.5 * jnp.min(jnp.diff(self.t_a))

        return jnp.searchsorted(t_eq[1:] - half_gap, self.t_a, side="right")

    def _piecewise(self, values: Array) -> Array:
        """Per-equilibration `values` on the output times (zero without equilibrations)."""
        if self.equilibrations is None:
            return jnp.zeros((self.t_a.shape[0],) + values.shape[1:])
        on_grid: Array = values[self.equilibration_index()]
        filled: Array = self.equilibrations.count > 0

        return jnp.where(filled, on_grid, jnp.zeros_like(on_grid))

    def interior(self) -> Array:
        """Interior (dissolved) abundances at each output time [atoms], piecewise constant
        between equilibrations.

        Returns:
            Interior abundances, ordered per `isofate.species.SYMBOLS`
        """
        if self.equilibrations is None:
            return jnp.zeros_like(self.y_a)
        return self._piecewise(self.equilibrations.y_int)

    def surface_temperatures(self) -> tuple[Array, Array]:
        """Surface temperatures at each output time [K], piecewise constant between
        equilibrations.

        Returns:
            The surface temperature from the atmosphere descent, and the one used by Atmodeller
        """
        if self.equilibrations is None:
            zeros: Array = jnp.zeros(self.t_a.shape[0])
            return zeros, zeros
        return (
            self._piecewise(self.equilibrations.surface_temperature),
            self._piecewise(self.equilibrations.surface_temperature_atmodeller),
        )

    def to_dict(self, to_numpy: bool = False) -> dict[str, Any]:
        """IsoFATE's time series, by key.

        Jittable with ``to_numpy=False``. Then ``t_atmodeller`` has the fixed size of the
        equilibration buffers, NaN beyond the last equilibration; with ``to_numpy=True`` it holds
        only the equilibrations.

        Args:
            to_numpy: Return NumPy arrays (outside jit). Defaults to ``False``.

        Returns:
            The output times ``time`` [s], abundances ``N_X`` and ``N_X_int`` [atoms], atom
            fractions ``x1``-``x7``, number fluxes ``Phi_X``, radii ``Rp`` and ``Ratm`` [m],
            ``Matm`` [kg], ``fatm``, ``Vpot`` [J/kg], ``phi``, ``phic`` [kg/m2/s], ``Mloss`` [kg],
            surface temperatures ``T_surf_analytic`` and ``T_surf_atmod`` [K], and the
            equilibration times ``t_atmodeller`` [s]
        """
        alg: AtmosphereState = self.diagnostics()
        y_int: Array = self.interior()
        T_surf_analytic, T_surf_atmod = self.surface_temperatures()
        mass_flux = jnp.asarray(alg.mass_flux)
        t_atmodeller: Array = (
            jnp.zeros(0) if self.equilibrations is None else self.equilibrations.t
        )

        out: dict[str, Any] = {
            "time": self.t_a,
            "Rp": alg.escape_radius,
            "Ratm": alg.convective_envelope_thickness,
            "Matm": alg.atmosphere_mass,
            "Vpot": alg.gravitational_potential,
            "fatm": alg.atmosphere_mass_fraction,
            "Mloss": mass_flux * jnp.asarray(alg.area) * self.delta_t,
            "phi": mass_flux,
            "phic": alg.critical_mass_flux,
            **{f"N_{symbol}": self.y_a[:, i] for i, symbol in enumerate(SYMBOLS)},
            **{f"N_{symbol}_int": y_int[:, i] for i, symbol in enumerate(SYMBOLS)},
            **{
                f"x{i + 1}": jnp.asarray(alg.atom_fractions)[:, i] for i in range(len(SYMBOLS))
            },
            **{
                f"Phi_{symbol}": jnp.asarray(alg.number_flux)[:, i]
                for i, symbol in enumerate(SYMBOLS)
            },
            "T_surf_analytic": T_surf_analytic,
            "T_surf_atmod": T_surf_atmod,
            "t_atmodeller": t_atmodeller,
        }
        if to_numpy:
            out = {key: np.asarray(value) for key, value in out.items()}
            if self.equilibrations is not None:
                out["t_atmodeller"] = out["t_atmodeller"][: int(self.equilibrations.count)]

        return out

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """IsoFATE's output as DataFrames. Not jittable.

        Returns:
            ``time_series`` (one row per output time, the columns of `to_dict`) and, with the
            coupling, ``equilibrations`` (one row per equilibration, aligned with the batch rows
            of `atmodeller`)
        """
        series: dict[str, np.ndarray] = self.to_dict(to_numpy=True)
        series.pop("t_atmodeller")
        dataframes: dict[str, pd.DataFrame] = {"time_series": pd.DataFrame(series)}

        if self.equilibrations is not None:
            eq: Equilibrations = self.equilibrations
            n: int = int(eq.count)
            columns: dict[str, np.ndarray] = {
                "t": np.asarray(eq.t[:n]),
                "surface_temperature": np.asarray(eq.surface_temperature[:n]),
                "surface_temperature_atmodeller": np.asarray(
                    eq.surface_temperature_atmodeller[:n]
                ),
                "mantle_melt_fraction": np.asarray(eq.mantle_melt_fraction[:n]),
            }
            for name, values in (("before", eq.y_before), ("after", eq.y_after), ("int", eq.y_int)):
                for i, symbol in enumerate(SYMBOLS):
                    columns[f"N_{symbol}_{name}"] = np.asarray(values[:n, i])
            for i, element in enumerate(_COUPLING_ELEMENTS):
                columns[f"mass_constraint_{element}"] = np.asarray(eq.mass_constraints[:n, i])
            dataframes["equilibrations"] = pd.DataFrame(columns)

        return dataframes

    def to_pickle(self, path: Path | str) -> None:
        """Writes the output to a pickle file, as DataFrames (`to_dataframes`), with the
        Atmodeller output's own DataFrames kept separately under ``atmodeller``. Not jittable.

        The `Parameters` are not pickled. Read the file back with `read_pickle`.

        Args:
            path: Output file
        """
        content: dict[str, Any] = dict(self.to_dataframes())
        if self.atmodeller is not None:
            content["atmodeller"] = self.atmodeller.to_dataframes()
        with open(path, "wb") as handle:
            pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Path | str) -> dict[str, Any]:
    """Reads an output written by `Output.to_pickle`.

    Args:
        path: Pickle file

    Returns:
        The DataFrames: ``time_series``, and with the coupling ``equilibrations`` and
        ``atmodeller`` (a dictionary of Atmodeller's own DataFrames)
    """
    with open(path, "rb") as handle:
        return pickle.load(handle)
