# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE."""

import equinox as eqx
import jax.numpy as jnp
from atmodeller import ChemicalSpecies, EquilibriumModel, ReservoirSpecies
from atmodeller import Planet as AtmodellerPlanet
from atmodeller.constants import SILICATE_MELT_PHASE_INDEX
from atmodeller.solubility import get_solubility_models
from jax.typing import ArrayLike
from jaxtyping import Array

from isofate.mantle_iron import MantleIronState
from isofate.parameters import Parameters
from isofate.species import SYMBOLS

solubility_models = get_solubility_models()

_COUPLING_ELEMENTS: tuple[str, ...] = tuple(symbol for symbol in SYMBOLS if symbol != "D")
"""Elements coupled to Atmodeller, ordered per `SYMBOLS` with D removed (Atmodeller has no
separate D reservoir, so D is aggregated into H); must match the mass_constraints keys."""

# Precomputed once: which isofate.species.SYMBOLS index feeds each _COUPLING_ELEMENTS slot, and
# where "D" and "H" themselves sit in each ordering - used by aggregate_D_into_H below.
_COUPLING_ELEMENT_INDICES: tuple[int, ...] = tuple(
    SYMBOLS.index(element) for element in _COUPLING_ELEMENTS
)
_D_INDEX: int = SYMBOLS.index("D")
_H_POSITION: int = _COUPLING_ELEMENTS.index("H")


class TrackedGasSpecies(eqx.Module):
    """One isofate-tracked gas-phase species, derived from atmodeller's own species lists."""

    gas_name: str
    """Atmodeller canonical gas-phase name, e.g. "H2_g", "O2S_g"."""
    label: str
    """Isofate's human-readable output label, e.g. "H2", "SO2"."""
    melt_name: str | None
    """Atmodeller canonical melt-phase name, or ``None`` if this species has no melt reservoir."""


class AtmodellerCoupler(eqx.Module):
    """Atmodeller coupler for IsoFATE, which builds an Atmodeller model for the interior-atmosphere
    coupling.

    `AtmodellerCoupler(parameters)` (and so `construct_equilibrium_model`, which `__init__` calls)
    must be run outside any `jax.jit`/`eqx.filter_jit` trace, never inside one: it does
    Python-level construction (species/thermodynamic-data lookups, not jax.numpy/lax ops), and
    `EquilibriumModel.from_state` itself builds a new jitted solver as part of construction, which
    can't happen from inside an already-jitted trace. Construct one instance once per run, then
    call `.update_state()`/`.update_constraints()` and `.solve()` on it repeatedly - those are safe
    (and intended) to jit.

    Args:
        parameters: Parameters object containing system, escape mechanism, and options
        max_surface_temperature: Maximum surface temperature to allow in the Atmodeller model
            (K). Defaults to ``6000`` K. Above 6000 K data might be missing for some species.
    """

    parameters: Parameters
    equilibrium_model: EquilibriumModel
    max_surface_temperature: float
    tracked_species: tuple[TrackedGasSpecies, ...]

    def __init__(self, parameters: Parameters, max_surface_temperature: float = 6000):
        self.parameters = parameters
        self.equilibrium_model = self.construct_equilibrium_model()
        self.max_surface_temperature = max_surface_temperature
        self.tracked_species = self.set_tracked_gas_species()

    def construct_equilibrium_model(self) -> EquilibriumModel:
        """Constructs an equilibrium model for the interior-atmosphere coupling.

        This currently hard-codes the species set and solubility models, but could be made more
        flexible in the future.

        Reads planet mass, core mass fraction, surface radius, and temperature from
        `self.parameters.system.planet` - fixed for the whole isocalc run and never updated in
        the time integration.

        Returns:
            EquilibriumModel: An equilibrium model for the interior-atmosphere coupling
        """
        # Required planet parameters for Atmodeller; these are fixed for the whole isocalc run and
        # never updated in the time integration
        planet_mass: Array = self.parameters.system.planet.mass
        core_mass_fraction: Array = self.parameters.system.planet.core_mass_fraction
        surface_radius: Array = self.parameters.system.planet.rocky_radius
        temperature: Array = self.parameters.system.planet.temperature

        # Gas-phase species
        He_g: ChemicalSpecies = ChemicalSpecies.create_gas("He")
        H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
        H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
        O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
        CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
        CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
        CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")
        N2_g: ChemicalSpecies = ChemicalSpecies.create_gas("N2")
        S2_g: ChemicalSpecies = ChemicalSpecies.create_gas("S2")
        H2O4S_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O4S")
        SO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("SO2")

        gas_species: tuple[ChemicalSpecies, ...] = (
            He_g,
            H2_g,
            H2O_g,
            O2_g,
            CO2_g,
            CO_g,
            CH4_g,
            N2_g,
            S2_g,
            H2O4S_g,
            SO2_g,
        )

        # Dissolved (melt-reservoir) species; O2 and H2O4S have no solubility model, so they are
        # gas-only and have no counterpart here
        He_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "He", solubility=solubility_models["He_basalt_jambon86"]
        )
        H2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "H2", solubility=solubility_models["H2_basalt_hirschmann12"]
        )
        H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "H2O", solubility=solubility_models["H2O_basalt_dixon95"]
        )
        CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "CO2", solubility=solubility_models["CO2_basalt_dixon95"]
        )
        CO_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "CO", solubility=solubility_models["CO_basalt_yoshioka19"]
        )
        CH4_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "CH4", solubility=solubility_models["CH4_basalt_ardia13"]
        )
        N2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "N2", solubility=solubility_models["N2_basalt_libourel03"]
        )
        S2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
            "S2", solubility=solubility_models["S2_sulfide_basalt_boulliung23"]
        )

        melt_species: tuple[ReservoirSpecies, ...] = (
            He_d,
            H2_d,
            H2O_d,
            CO2_d,
            CO_d,
            CH4_d,
            N2_d,
            S2_d,
        )

        planet: AtmodellerPlanet = AtmodellerPlanet.from_species(
            gas_species,
            planet_mass=planet_mass,
            core_mass_fraction=core_mass_fraction,
            surface_radius=surface_radius,
            temperature=temperature,
            silicate_melt_species=melt_species,
        )
        model: EquilibriumModel = EquilibriumModel.from_state(planet)

        return model

    # TODO: keep isocalc_values or use isofate_values?
    # Perhaps also the method name should eventually change
    def aggregate_D_into_H(self, isocalc_values: ArrayLike) -> Array:
        """Maps an isocalc per-species array (ordered per :data:`isofate.species.SYMBOLS`) onto the
        layout AtmodellerCoupler expects (:data:`_COUPLING_ELEMENTS`).

        Atmodeller has no separate D reservoir, so D's abundance is folded into H's. The inverse
        step, splitting H back into H and D, uses the D/H ratio `X_DH`.

        Args:
            isocalc_values: Per-species values (e.g. number of atoms) with leading axis ordered
                per :data:`isofate.species.SYMBOLS`

        Returns:
            Per-element values with leading axis ordered per :data:`_COUPLING_ELEMENTS`, with D
            added to H
        """
        values: Array = jnp.asarray(isocalc_values, dtype=float)
        coupled: Array = values[jnp.array(_COUPLING_ELEMENT_INDICES)]
        coupled = coupled.at[_H_POSITION].add(values[_D_INDEX])

        return coupled

    def set_tracked_gas_species(self) -> tuple[TrackedGasSpecies, ...]:
        """Ordered set of gas-phase species isofate tracks as molecular output - every gas species
        except He, which isofate tracks separately as one of its own 7 core H/He/D/O/C/N/S species.

        Derived directly from the equilibrium model's own species objects, which already carry both
        the original formula (used as isofate's output label, e.g. "SO2") and the
        Hill-canonicalized name atmodeller actually indexes its solve output by (e.g. "O2S_g" -
        atmodeller internally derives this from molmass, see ChemicalSpeciesData) - so no separate
        Hill-notation conversion or per-species override table is needed on isofate's side.
        """
        gas_species: tuple[ChemicalSpecies, ...] = (
            self.equilibrium_model.parameters.reaction_system.phase_system.gas.species.species
        )
        melt_names: tuple[str, ...] = (
            self.equilibrium_model.parameters.reaction_system.phase_system.phases[
                SILICATE_MELT_PHASE_INDEX
            ].species_names
        )
        melt_by_stem: dict[str, str] = {name.removesuffix("_d"): name for name in melt_names}

        tracked: list[TrackedGasSpecies] = []
        for sp in gas_species:
            if sp.data.formula == "He":
                continue
            tracked.append(
                TrackedGasSpecies(
                    sp.data.name, sp.data.formula, melt_by_stem.get(sp.data.hill_formula)
                )
            )

        return tuple(tracked)

    def run(
        self,
        Teq,
        Rp,
        mu,
        melt_fraction,
        mantle_iron_state: MantleIronState | None,
        N_H_atm,
        N_He_atm,
        N_O_atm,
        N_C_atm,
        N_N_atm,
        N_S_atm,
        N_H_int,
        N_He_int,
        N_O_int,
        N_C_int,
        N_N_int,
        N_S_int,
        interior_atmosphere: EquilibriumModel,
        initial_guess=None,
        full_output: bool = True,
    ):
        """To mimic AtmodellerCoupler arguments"""
