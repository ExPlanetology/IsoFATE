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
from isofate.escape.fractionation import EscapeNumberFluxBase, make_escape_number_flux
from isofate.escape.mechanisms import EscapeMechanism
from isofate.initial_condition import InitialCondition
from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    DEFAULT_SPECIES,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)
from isofate.system import System


class IsocalcOptions(eqx.Module):
    """Mode switches and tuning constants for the driver (`isocalc`), fixed for the whole run.

    These are numerical/modeling choices, as opposed to the rest of `Parameters` (`system`, the
    escape mechanism, the initial condition) and the end time, which describe the physical
    initial-value problem being solved.

    Args:
        rad_evol: Set to False to fix the planet (escape) radius at the rocky radius, with no
            envelope (see `isofate.engine.escape_radius`). Defaults to ``True``.
        n_steps: Number of timesteps; convergence occurs at 1e6.
        t_start: Simulation start time, i.e. system age at the start of the run [yr].
        thermal: Toggles planet radius contraction in the Lopez/Fortney equations (False removes
            the age term).
        n_atmodeller: Interval of timesteps between each Atmodeller call.
        save_molecules: Save molecular abundances at every timestep (True) or only the final
            abundances (False).
        dynamic_phi: Toggle dynamic phi calculation based on the most abundant species (True) or
            static phi calculation, always using H/He as the dominant pair (False). This selects
            `Parameters.escape_number_flux` (see `make_escape_number_flux`), which every driver
            uses. Defaults to ``True``.
        mass_loss_fraction: Fraction of the atmospheric mass at the start of an integration whose
            loss stops the integration (a diffrax event). The default ``1 - 1e-6`` stops it once
            the atmosphere is effectively exhausted; a smaller value (e.g. ``0.05``) stops it
            earlier so that it can be restarted, e.g. after re-equilibrating with Atmodeller (see
            `isofate.integrators.AdaptiveIntegrator`). Defaults to ``1 - 1e-6``.
        species_loss_fraction: If set, the integration also stops once any species has lost this
            fraction of its own atmospheric abundance at the start of the integration (e.g.
            ``0.05``), so that trace species drained by escape trigger a restart even when the
            total mass hardly changes. Species with an atom fraction below ``1e-6`` at the start
            are ignored. Defaults to ``None`` (off).
    """

    rad_evol: bool = True
    n_steps: int = int(1e5)
    t_start: ArrayLike = 1e6
    thermal: bool = True
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    dynamic_phi: bool = True
    # 1 - isofate.integrators.EXHAUSTION_FRACTION (not imported, to avoid an import cycle)
    mass_loss_fraction: float = 1 - 1e-6
    species_loss_fraction: float | None = None


class Parameters(eqx.Module):
    """Parameters for an IsoFATE simulation.

    The `atmosphere_*`/`envelope_*` methods take `y`, the per-species abundances [atoms] ordered
    per `isofate_species`. IsoFATE's species are all elemental (H, He, D, O, C, N, S), so `y`
    counts atoms of each element regardless of which molecules they are bound in. This is in
    contrast to Atmodeller, whose species are molecules (e.g. H2, H2O, CO2). Quantities derived
    here are therefore elemental: atom counts and fractions, and a mean mass per atom. Where they
    have a molecular counterpart in Atmodeller's output, such as the volume mixing ratio or the
    mean molecular mass, the two differ.

    Args:
        system: Star-planet system
        escape_mechanism: Escape mechanism setting the bulk escaping mass flux
        initial_condition: Initial atmospheric abundances (see `initial_abundances`)
        isofate_species: Tracked species. Defaults to `DEFAULT_SPECIES`.
        binary_diffusion_coefficients: Binary diffusion coefficients. Defaults to
            `DEFAULT_BINARY_DIFFUSION`.
        isocalc_options: Mode switches and tuning constants. Defaults to `IsocalcOptions()`.
        atmosphere: Atmosphere model. Defaults to `AtmosphereModel()`.
    """

    system: System
    escape_mechanism: EscapeMechanism
    _: dataclasses.KW_ONLY
    initial_condition: InitialCondition
    isofate_species: IsoFATESpecies = DEFAULT_SPECIES
    binary_diffusion_coefficients: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION
    isocalc_options: IsocalcOptions = IsocalcOptions()
    atmosphere: AtmosphereModel = AtmosphereModel()

    @property
    def escape_number_flux(self) -> EscapeNumberFluxBase:
        """Escape number flux model selected by `isocalc_options.dynamic_phi`, built with
        `isofate_species` and `binary_diffusion_coefficients` (see `make_escape_number_flux`).

        A property rather than a stored field, so that it is rebuilt from those fields inside a
        traced function: gradients (e.g. with respect to `binary_diffusion_coefficients`) reach
        the fields themselves, and it can't go stale if they are changed with `eqx.tree_at`.
        """
        return make_escape_number_flux(
            self.isocalc_options.dynamic_phi,
            species=self.isofate_species,
            binary_diffusion=self.binary_diffusion_coefficients,
        )

    def __check_init__(self):
        if self.initial_condition.species.species != self.isofate_species.species:
            raise ValueError(
                f"initial_condition.species {self.initial_condition.species.species} differ from "
                f"isofate_species {self.isofate_species.species}"
            )

    def initial_abundances(self) -> Array:
        """Initial atmospheric abundances [atoms], ordered per `isofate_species`, from
        `initial_condition`."""
        return self.initial_condition.abundances(self.system.planet.mass)

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
        atmospheric exhaustion in isocalc), so a discarded branch can't corrupt a gradient
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
