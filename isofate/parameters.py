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
        t_start: Simulation start time, i.e. system age at the start of the run [yr].
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
        mass_loss_fraction: Fraction of the atmospheric mass at the start of an integration whose
            loss stops the integration (a diffrax event). The default ``1 - 1e-6`` stops it once
            the atmosphere is effectively exhausted; a smaller value (e.g. ``0.05``) stops it
            earlier so that it can be restarted, e.g. after re-equilibrating with Atmodeller (see
            `isofate.engine.integrate_segments`). Defaults to ``1 - 1e-6``.
    """

    rad_evol: bool = True
    # TODO: This can be removed once the JAX version is swapped in for the original isocalc.
    melt_fraction_override: ArrayLike | bool = False
    n_steps: int = int(1e5)
    t_start: ArrayLike = 1e6
    thermal: bool = True
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    mantle_iron: MantleIronConfig | None = None
    dynamic_phi: bool = False
    # 1 - isofate.engine.EXHAUSTION_FRACTION (not imported, to avoid an import cycle)
    mass_loss_fraction: float = 1 - 1e-6


class Parameters(eqx.Module):
    """Parameters for an IsoFATE simulation.

    The `atmosphere_*`/`envelope_*` methods take `y`, the per-species abundances [atoms] ordered
    per `isofate_species`. IsoFATE's species are all elemental (H, He, D, O, C, N, S), so `y`
    counts atoms of each element regardless of which molecules they are bound in. This is in
    contrast to Atmodeller, whose species are molecules (e.g. H2, H2O, CO2). Quantities derived
    here are therefore elemental: atom counts and fractions, and a mean mass per atom. Where they
    have a molecular counterpart in Atmodeller's output, such as the volume mixing ratio or the
    mean molecular mass, the two differ.
    """

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

    def atmosphere_atom_fractions(self, y: Array) -> Array:
        """Atom fraction of each species [ndim]: `y / atmosphere_atoms(y)`.

        Because `y` counts elements, not molecules, this is each element's share of all atoms -
        not a volume mixing ratio (the mole fraction of molecules), which Atmodeller reports from
        its gas-phase speciation.

        Returns 0 for every species when the total abundance is zero, rather than letting 0/0
        propagate as NaN (see `atmosphere_mean_mu`).

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`

        Returns:
            Atom fraction of each species [ndim], ordered per `self.species`
        """
        return safe_divide(y, self.atmosphere_atoms(y))

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
