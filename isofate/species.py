# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Canonical registry of the elements tracked by IsoFATE's escape/interior model.

Several places in isocalc()/isofunks.py need the same 7 species, in the same order, either as a
list of atomic masses or as a list of symbols for name-based lookups (e.g. get_binary_diffusion_coeff).
Previously each of those sites hand-typed its own copy of the list; this module is the single
source of truth they should all import from instead.
"""

import equinox as eqx
import numpy as np
from atmodeller.jax_utils import NpFloat
from atmodeller.sci_utils import constants, unit_conversion
from molmass import Formula

from isofate.constants import binary_diffusion

SYMBOLS: tuple[str, ...] = ("H", "He", "D", "O", "C", "N", "S")

# Built once at import time rather than on every get_binary_diffusion_coeff call: with
# dynamic_phi=True, get_binary_diffusion_coeff runs ~16 times per timestep (via
# Phi_minor_species), so rebuilding this dict per call meant ~1.6e6 redundant dict constructions
# over a full n_steps=1e5 run.
_BINARY_DIFFUSION_COEFF_FUNCS = {
    ("H", "D"): binary_diffusion.H_D,
    ("H", "He"): binary_diffusion.H_He,
    ("H", "O"): binary_diffusion.H_O,
    ("H", "C"): binary_diffusion.H_C,
    ("H", "N"): binary_diffusion.H_N,
    ("H", "S"): binary_diffusion.H_S,
    ("He", "D"): binary_diffusion.He_D,
    ("He", "O"): binary_diffusion.He_O,
    ("He", "C"): binary_diffusion.He_C,
    ("He", "N"): binary_diffusion.He_N,
    ("He", "S"): binary_diffusion.He_S,
}


class IsoFATESpecies(eqx.Module):
    """IsoFATE species container.

    Atomic masses [kg/atom] are computed once at construction from `molmass` (IUPAC standard
    atomic weights) rather than hand-typed constants, so they stay correct for any symbol without
    maintaining a lookup table by hand.

    Args:
        species: Element/isotope symbols, in the canonical tracked order (default: the 7 species
            IsoFATE tracks - H, He, D, O, C, N, S; see `SYMBOLS` above).
    """

    species: tuple[str, ...] = eqx.field(converter=tuple, default=SYMBOLS)
    atomic_masses: NpFloat = eqx.field(init=False)
    mass_by_symbol: dict[str, float] = eqx.field(init=False)

    def __init__(self, species: tuple[str, ...] = SYMBOLS):
        self.species = tuple(species)
        self.atomic_masses = np.array(
            [
                Formula(symbol).mass * unit_conversion.g_to_kg / constants.Avogadro
                for symbol in self.species
            ]
        )
        self.mass_by_symbol = dict(zip(self.species, self.atomic_masses.tolist()))


DEFAULT_SPECIES = IsoFATESpecies()


def get_binary_diffusion_coeff(
    species1: str, species2: str, T, species: IsoFATESpecies = DEFAULT_SPECIES
):
    """Binary diffusion coefficient between two species at temperature T [K].

    Always returns b_light_heavy regardless of input order.

    A plain function rather than an `IsoFATESpecies` method: `eqx.Module` wraps every bound
    method access in a fresh `BoundMethod` pytree (so methods survive `jax.jit`/`vmap`), which
    costs ~10us per attribute lookup - fine occasionally, but this runs ~16 times per timestep
    (via Phi_minor_species) and added ~20s to a full n_steps=1e5 isocalc run.
    """
    mass_by_symbol = species.mass_by_symbol
    # Always sort to ensure consistent lookup (lighter element first by atomic mass)
    if mass_by_symbol.get(species1, 0) <= mass_by_symbol.get(species2, 0):
        key = (species1, species2)
    else:
        key = (species2, species1)

    # Default to H-He if pair not found
    func = _BINARY_DIFFUSION_COEFF_FUNCS.get(key, binary_diffusion.H_He)
    return func(T)
