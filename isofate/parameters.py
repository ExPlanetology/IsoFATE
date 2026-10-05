# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from atmodeller.jax_utils import safe_divide
from jaxtyping import Array, ArrayLike

from isofate.atmosphere import AtmosphereModel
from isofate.escape.fractionation import EscapeNumberFlux, EscapeNumberFluxBase
from isofate.escape.mechanisms import EscapeMechanism
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
    # TODO: This can be removed once the JAX version is swapped in for the original isocalc.
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
    escape_number_flux: EscapeNumberFluxBase = EscapeNumberFlux()
    isocalc_options: IsocalcOptions = IsocalcOptions()
    atmosphere: AtmosphereModel = AtmosphereModel()

    def atmosphere_mass(self, y: Array) -> Array:
        """Total atmospheric mass [kg].

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`

        Returns:
            Total atmospheric mass [kg]
        """
        return jnp.dot(y, self.isofate_species.atomic_masses)

    def atmosphere_mass_fraction(self, y: Array) -> Array:
        """Atmospheric mass fraction [ndim]: `atmosphere_mass(y)` divided by the planet mass.

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`

        Returns:
            Atmospheric mass fraction [ndim]
        """
        return self.atmosphere_mass(y) / self.system.planet.mass

    def atmosphere_atoms(self, y: Array) -> Array:
        """Total atmospheric atoms [atoms].

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`

        Returns:
            Total atmospheric atoms [atoms]
        """
        return jnp.sum(y)

    def atmosphere_mean_mu(self, y: Array) -> Array:
        """Mean atmospheric particle mass [kg], given per-species abundances `y` [atoms],
        ordered per `self.species`.

        This is the mean mass per *atom*, `atmosphere_mass(y) / atmosphere_atoms(y)`, because `y`
        counts elements, not molecules. It therefore differs from the mean *molecular* mass that
        Atmodeller reports from its gas-phase speciation: for a pure-H2 atmosphere this gives
        ~m_H, whereas the molecular value is ~2 m_H. Quantities that scale with this value (e.g.
        the scale height and Bondi radius) differ accordingly. Using the per-atom value is
        defensible for the escaping upper atmosphere, where hydrogen is largely atomic, but is a
        modelling assumption for the deeper envelope.

        Returns 0 when the total abundance is zero, rather than letting 0/0 propagate as NaN -
        this keeps the value finite even when a caller only conditionally uses it (e.g. near-total
        atmospheric exhaustion in isocalc_jax), so a discarded branch can't corrupt a gradient
        through the selecting `jnp.where`.

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`.

        Returns:
            Mean atmospheric particle mass [kg]
        """
        return safe_divide(self.atmosphere_mass(y), self.atmosphere_atoms(y))

    def envelope_mass(self, y: Array) -> Array:
        """Alias for atmosphere mass"""
        return self.atmosphere_mass(y)

    def envelope_mass_fraction(self, y: Array) -> Array:
        """Alias for atmosphere mass fraction"""
        return self.atmosphere_mass_fraction(y)

    def envelope_mean_mu(self, y: Array) -> Array:
        """Alias for atmosphere mean mu"""
        return self.atmosphere_mean_mu(y)
