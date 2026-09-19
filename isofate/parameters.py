# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import dataclasses

import equinox as eqx
from jaxtyping import ArrayLike

from isofate.escape import EscapeMechanism
from isofate.escape_fractionation import EscapeNumberFlux
from isofate.mantle_iron import MantleIronConfig
from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    DEFAULT_SPECIES,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)
from isofate.system import System


class IsocalcOptions(eqx.Module):
    """Mode switches and tuning constants for `isocalc`, fixed for the whole run.

    These are numerical/modeling choices, as opposed to `isocalc`'s other arguments (`system`,
    `F0`, `time`, and the initial `N_x` abundances), which describe the actual physical
    initial-value problem being solved and so stay direct `isocalc` arguments.

    Args:
        rad_evol: Set to False to fix the planet radius at the rocky radius.
        melt_fraction_override: Fixed mantle melt fraction; if False, it is instead calculated
            from Mp and T_surface.
        n_steps: Number of timesteps; convergence occurs at 1e6.
        t0: Simulation start time [yr].
        thermal: Toggles planet radius contraction in the Lopez/Fortney equations (False removes
            the age term).
        n_atmodeller: Interval of timesteps between each Atmodeller call.
        save_molecules: Save molecular abundances at every timestep (True) or only the final
            abundances (False).
        mantle_iron: Allows Fe in the mantle to react with O2. `reaction_type="dynamic"` reacts
            only molten mantle Fe; `reaction_type="static"` reacts all mantle Fe; also specify
            `fe_mass_fraction`. None disables this.
        dynamic_phi: Toggle dynamic phi calculation based on the most abundant species (True) or
            static phi calculation (False).
    """

    rad_evol: bool = True
    melt_fraction_override: ArrayLike | bool = False
    n_steps: int = int(1e5)
    t0: ArrayLike = 1e6
    thermal: bool = True
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    mantle_iron: MantleIronConfig | None = None
    dynamic_phi: bool = False


class Parameters(eqx.Module):
    """Parameters for an IsoFATE simulation."""

    system: System
    escape_mechanism: EscapeMechanism
    _: dataclasses.KW_ONLY
    isofate_species: IsoFATESpecies = DEFAULT_SPECIES
    binary_diffusion_coefficients: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION
    escape_number_flux: EscapeNumberFlux = EscapeNumberFlux()
    isocalc_options: IsocalcOptions = IsocalcOptions()
