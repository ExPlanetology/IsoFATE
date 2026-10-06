# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE."""

import equinox as eqx
import jax.numpy as jnp
from atmodeller import ChemicalSpecies, EquilibriumModel, ReservoirSpecies
from atmodeller import Planet as AtmodellerPlanet
from atmodeller.constants import GAS_PHASE_INDEX, SILICATE_MELT_PHASE_INDEX
from atmodeller.output_base import OutputNamedArraysDict
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

    @eqx.filter_jit
    def update_solve_extract(
        self,
        mantle_melt_fraction: Array,
        temperature: Array,
        mass_constraints: dict,
        base_solution_array: Array,
    ):
        """Fuses state/constraint updates, the solve, and output extraction into one jitted call.

        ``planet_mass``/``surface_radius`` are deliberately not updated here: both are fixed for
        the whole isocalc run and already set on ``self.equilibrium_model`` by
        ``construct_equilibrium_model``, so leaving them out of ``update_state`` (which defaults
        each omitted field to "no change") avoids re-asserting a value that never differs from
        what the model already holds.

        ``update_state``/``update_constraints`` are built on ``eqx.tree_at``, which does several
        full Python-level traversals of the whole model pytree per call (O(num_leaves); ~3479
        leaves here) - calling them un-jitted cost ~40ms/call regardless of how much the actual
        solve needed to change. Wrapping the whole update+solve+extract sequence in one
        eqx.filter_jit call instead pays that Python-level tree rebuilding cost once, at trace
        time; every later call just replays the compiled XLA graph (~9ms/call: solve +
        extraction). This mirrors atmodeller's own tested idiom for this exact situation (see
        atmodeller/tests/test_retracing.py's `call_solver`/`workflow` pattern). ``self`` is
        partitioned by eqx.filter_jit like any other argument, so reusing one coupler instance
        per run compiles once.

        `solve_with_default()` is exactly `solve(all-NaN array of this shape)` (see
        atmodeller/classes.py), so passing an all-NaN `base_solution_array` reproduces the cold-
        start path without needing a separate branch inside this jitted method.
        """
        model = self.equilibrium_model.update_state(
            mantle_melt_fraction=mantle_melt_fraction, temperature=temperature
        ).update_constraints(mass_constraints=mass_constraints)
        output = model.solve(base_solution_array)
        sol = output.to_dict(output_format="elements_species", to_numpy=False)

        return sol, output.solution, output.multi_attempt_solution.success

    @eqx.filter_jit
    def update_solve_extract_narrow(
        self,
        mantle_melt_fraction: Array,
        temperature: Array,
        mass_constraints: dict,
        base_solution_array: Array,
    ):
        """Like ``update_solve_extract``, but skips every diagnostic the coupling never reads.

        ``to_dict("elements_species")`` computes, for *every* species in *every* phase: activity
        (a real-gas EOS volume root-find for non-ideal species), mass/mole fractions, partial
        pressure, phase volume, log10dIW, and metallicity. The coupling only ever reads element
        ``number_moles`` (gas + silicate_melt) for the six mass-constrained elements, the gas
        phase's total mass, and O2_g's number_moles (for the iron-buffer branches). All of those
        need only ``log_number_moles`` and the (static) formula matrix - no EOS, no activity.
        Building only these keeps the traced graph much smaller without changing any value the
        coupling actually uses from this path.

        Not a substitute for ``update_solve_extract`` when species-level diagnostics are wanted
        (``save_molecules=True``, or the end-of-run full snapshot) - those still need the full
        ``to_dict`` output.
        """
        model = self.equilibrium_model.update_state(
            mantle_melt_fraction=mantle_melt_fraction, temperature=temperature
        ).update_constraints(mass_constraints=mass_constraints)
        output = model.solve(base_solution_array)

        out_dict = OutputNamedArraysDict(output.parameters, output.multi_attempt_solution)
        gas_output = out_dict.get_phase_output_from_index(GAS_PHASE_INDEX)
        melt_output = out_dict.get_phase_output_from_index(SILICATE_MELT_PHASE_INDEX)

        sol: dict = {}
        for element in _COUPLING_ELEMENTS:
            gas_idx = gas_output.phase.species.get_element_index(element)
            melt_idx = melt_output.phase.species.get_element_index(element)
            if gas_idx == -1 or melt_idx == -1:
                raise ValueError(
                    f"Element {element!r} missing from the gas or silicate_melt phase species; "
                    "the narrow extraction assumes both phases carry all six coupling elements."
                )
            sol[element] = {
                "gas": {
                    "number_moles": gas_output.element_number_moles[..., gas_idx : gas_idx + 1]
                },
                "silicate_melt": {
                    "number_moles": melt_output.element_number_moles[..., melt_idx : melt_idx + 1]
                },
            }

        o2_index: int = gas_output.phase.species_names.index("O2_g")
        sol["O2_g"] = {
            "gas": {"number_moles": gas_output.species_number_moles[..., o2_index : o2_index + 1]}
        }
        sol["gas"] = {"phase": {"mass": gas_output.phase_mass}}

        return sol, output.solution, output.multi_attempt_solution.success

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
