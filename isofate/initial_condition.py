# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Initial atmospheric abundances for an isocalc run, held on `Parameters.initial_condition`.

- `InitialCondition`: base class; `abundances` returns the initial per-species abundances.
- `ProtosolarInitialCondition`: protosolar elemental ratios scaled to an atmospheric mass fraction
  of the planet, with an optional O/H enhancement.
- `AbundanceInitialCondition`: explicitly given abundances.
"""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from isofate.species import DEFAULT_SPECIES, IsoFATESpecies


class InitialCondition(eqx.Module):
    """Initial atmospheric abundances for an isocalc run.

    Args:
        species: Tracked species, which set the order of the abundances; must match
            `Parameters.isofate_species`. Keyword-only. Defaults to `DEFAULT_SPECIES`.
    """

    species: IsoFATESpecies = eqx.field(default=DEFAULT_SPECIES, kw_only=True)

    @abstractmethod
    def abundances(self, planet_mass: ArrayLike) -> Array:
        """Initial atmospheric abundances.

        Args:
            planet_mass: Planet mass [kg]

        Returns:
            Initial atmospheric abundances [atoms], ordered per `species.species`
        """


class ProtosolarInitialCondition(InitialCondition):
    """Protosolar elemental composition (Lodders 2003) for an atmospheric mass of
    `atmosphere_mass_fraction` times the planet mass, with the O/H ratio enhanced by
    `OtoH_enhancement`.

    Every species has its mole ratio to H (`mole_ratios_to_H`), so each abundance is that ratio
    times one normalisation, the H abundance (`hydrogen_abundance`), which is set by the total
    atmospheric mass. The initial X/H ratios are therefore exactly protosolar, and the atmospheric
    mass, in the species' atomic masses (as `Parameters.atmosphere_mass`), is exactly the target.

    Args:
        atmosphere_mass_fraction: Initial atmospheric mass as a fraction of the planet mass [ndim]
        OtoH_enhancement: Factor by which the O/H ratio is enhanced over protosolar [ndim].
            Defaults to ``1``.
        species: See `InitialCondition`; each must be in `PROTOSOLAR_MOLE_RATIOS_TO_H`
    """

    PROTOSOLAR_MOLE_RATIOS_TO_H: ClassVar[dict[str, float]] = {
        "H": 1.0,
        "He": 0.09709,  # proto-solar, Lodders 2003 (0.2741/0.711*mu_H/mu_He)
        "D": 0.0000194,  # solar, Lodders 2003
        "O": 0.00058,  # proto-solar, Lodders 2003 Table 2 (1.413e7/2.431e10)
        "C": 0.00029,  # proto-solar, Lodders 2003 Table 2
        "N": 0.000080,  # proto-solar, Lodders 2003 Table 2 (1.950e6/2.431e10)
        "S": 0.000018,  # proto-solar, Lodders 2003 Table 2 (4.449e5/2.431e10)
    }
    """Mole (number) ratio of each species to H [ndim]."""

    atmosphere_mass_fraction: float
    OtoH_enhancement: float = 1

    @property
    def mole_ratios_to_H(self) -> Array:
        """Mole ratio of each species to H [ndim], ordered per `species.species`: the protosolar
        ratios, with O/H enhanced by `OtoH_enhancement`.

        Raises:
            ValueError: If a species has no protosolar ratio
        """
        ratios: dict[str, float] = dict(self.PROTOSOLAR_MOLE_RATIOS_TO_H)
        ratios["O"] = ratios["O"] * self.OtoH_enhancement
        missing = [symbol for symbol in self.species.species if symbol not in ratios]
        if missing:
            raise ValueError(f"No protosolar mole ratio to H for species {missing}")

        return jnp.asarray([ratios[symbol] for symbol in self.species.species], dtype=float)

    def hydrogen_abundance(self, planet_mass: ArrayLike) -> Array:
        """H abundance [atoms], the normalisation of every abundance: the atmospheric mass divided
        by the mass per H atom of the whole composition, ``sum(ratio_X * m_X)``.

        Args:
            planet_mass: Planet mass [kg]

        Returns:
            H abundance [atoms]
        """
        atmosphere_mass = planet_mass * self.atmosphere_mass_fraction
        mass_per_H_atom = jnp.dot(self.mole_ratios_to_H, self.species.atomic_masses)

        return atmosphere_mass / mass_per_H_atom

    def abundances(self, planet_mass: ArrayLike) -> Array:
        """Initial atmospheric abundances: each species' mole ratio to H times the H abundance.

        Args:
            planet_mass: Planet mass [kg]

        Returns:
            Initial atmospheric abundances [atoms], ordered per `species.species`
        """
        return self.mole_ratios_to_H * self.hydrogen_abundance(planet_mass)


class AbundanceInitialCondition(InitialCondition):
    """Explicitly given initial atmospheric abundances.

    Args:
        initial_abundances: Initial atmospheric abundances [atoms], ordered per `species`
        species: See `InitialCondition`; their number must match the given abundances
    """

    initial_abundances: tuple[float, ...]

    def abundances(self, planet_mass: ArrayLike) -> Array:
        """Initial atmospheric abundances.

        Args:
            planet_mass: Planet mass [kg]; unused

        Returns:
            Initial atmospheric abundances [atoms], ordered per `species.species`
        """
        del planet_mass
        if len(self.initial_abundances) != len(self.species.species):
            raise ValueError(
                f"{len(self.initial_abundances)} initial abundances given for "
                f"{len(self.species.species)} species {self.species.species}"
            )

        return jnp.asarray(self.initial_abundances, dtype=float)
