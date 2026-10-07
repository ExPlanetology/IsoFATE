# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE.

The time integration re-equilibrates the atmosphere and interior with `AtmodellerCoupler.run`
(through `AtmodellerCoupler.reequilibrate`), which extracts only what feeds back into the
calculation: the element amounts in the gas and the melt, the converged solution (warm start) and
whether the solve converged. The complete Atmodeller output for every equilibration is built
afterwards, separately, by `AtmodellerCoupler.batched_output`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from atmodeller import ChemicalSpecies, EquilibriumModel, ReservoirSpecies
from atmodeller import Planet as AtmodellerPlanet
from atmodeller.constants import GAS_PHASE_INDEX, SILICATE_MELT_PHASE_INDEX
from atmodeller.jax_utils import safe_divide
from atmodeller.output import Output as AtmodellerOutput
from atmodeller.output_base import OutputNamedArraysDict
from atmodeller.solubility import get_solubility_models
from jax.typing import ArrayLike
from jaxtyping import Array

from isofate.constants import const
from isofate.engine import escape_radius
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
_H_INDEX: int = SYMBOLS.index("H")
_H_POSITION: int = _COUPLING_ELEMENTS.index("H")


class AtmodellerResult(eqx.Module):
    """Result of one Atmodeller re-equilibration (see `AtmodellerCoupler.run`).

    Args:
        y: Atmospheric abundances after re-equilibration [atoms], ordered per
            `isofate.species.SYMBOLS`
        y_int: Interior (dissolved) abundances after re-equilibration [atoms], ordered as `y`
        atmosphere_mass: Atmospheric mass [kg]. Derived from `y` (the gas phase's total mass is
            exactly the sum of its constituent atoms' masses) rather than Atmodeller's own gas
            phase mass, which is computed before the H/D split and so treats the aggregated H+D
            count as pure H (one atomic mass) instead of the correct per-isotope split.
        surface_temperature: Surface temperature from the atmosphere descent [K]
        surface_temperature_atmodeller: Surface temperature used by Atmodeller, capped at
            `AtmodellerCoupler.max_surface_temperature` [K]
        mantle_melt_fraction: Mantle melt fraction used by Atmodeller [ndim]
        mass_constraints: Total mass of each coupling element given to Atmodeller [kg], ordered
            per `_COUPLING_ELEMENTS`
        solution: Atmodeller solution array, for warm-starting the next call
        success: Whether the solve converged. If not, the other fields are not a valid
            equilibrium (they need not even conserve each element) and must not be used.
    """

    y: Array
    y_int: Array
    atmosphere_mass: Array
    surface_temperature: Array
    surface_temperature_atmodeller: Array
    mantle_melt_fraction: Array
    mass_constraints: Array
    solution: Array
    success: Array


class CouplerState(eqx.Module):
    """State carried between Atmodeller re-equilibrations (see `AtmodellerCoupler.reequilibrate`).

    Fixed structure, so it can be carried through `jax.lax.while_loop`/`jax.lax.scan` by the
    `isofate.integrators.Integrator` subclasses.

    Args:
        y_int: Interior (dissolved) abundances [atoms], ordered per `isofate.species.SYMBOLS`.
            Carried because Atmodeller's mass constraint is the atmosphere plus the interior, and
            escape only drains the atmosphere.
        solution: Last Atmodeller solution array, used to warm-start the next solve
        surface_temperature: Surface temperature from the atmosphere descent at the last solve [K]
        surface_temperature_atmodeller: Surface temperature used by Atmodeller at the last solve
            [K]
        mantle_melt_fraction: Mantle melt fraction used by Atmodeller at the last solve [ndim]
        mass_constraints: Element mass constraints given to Atmodeller at the last solve [kg],
            ordered per `_COUPLING_ELEMENTS`. With the surface temperature, melt fraction and
            solution, these are what `AtmodellerCoupler.batched_output` needs to rebuild the
            complete Atmodeller output.
        first_failure_time: Time of the first re-equilibration whose solve failed to converge
            [s], or ``inf`` if none has. A failure can't raise inside the jitted segment loop, so
            it is recorded here and the caller checks it afterwards (see
            `AtmodellerCoupler.check_converged`). Defaults to ``inf``.
    """

    y_int: Array
    solution: Array
    surface_temperature: Array
    surface_temperature_atmodeller: Array
    mantle_melt_fraction: Array
    mass_constraints: Array
    first_failure_time: Array = eqx.field(default_factory=lambda: jnp.array(jnp.inf))


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
    max_surface_temperature: ArrayLike

    def __init__(self, parameters: Parameters, max_surface_temperature: ArrayLike = 6000):
        self.parameters = parameters
        self.equilibrium_model = self.construct_equilibrium_model()
        self.max_surface_temperature = max_surface_temperature

    def construct_equilibrium_model(
        self,
        temperature: ArrayLike | None = None,
        mantle_melt_fraction: ArrayLike | None = None,
        mass_constraints: dict[str, ArrayLike] | None = None,
    ) -> EquilibriumModel:
        """Constructs an equilibrium model for the interior-atmosphere coupling.

        This currently hard-codes the species set and solubility models, but could be made more
        flexible in the future.

        Reads planet mass, core mass fraction and surface radius from
        `self.parameters.system.planet` - fixed for the whole run. The model for the time
        integration (no arguments) is a single problem whose state and constraints are updated
        before every solve. Passing arrays builds a batched model instead (Atmodeller sets the
        batch size from them), as `batched_output` does.

        Args:
            temperature: Surface temperature [K]. Defaults to ``None`` (the planet's temperature).
            mantle_melt_fraction: Mantle melt fraction [ndim]. Defaults to ``None`` (Atmodeller's
                default).
            mass_constraints: Total mass of each element [kg], by symbol. Defaults to ``None``.

        Returns:
            EquilibriumModel: An equilibrium model for the interior-atmosphere coupling
        """
        planet_mass: Array = self.parameters.system.planet.mass
        core_mass_fraction: Array = self.parameters.system.planet.core_mass_fraction
        surface_radius: Array = self.parameters.system.planet.rocky_radius
        if temperature is None:
            temperature = self.parameters.system.planet.temperature
        melt_fraction_kwargs: dict = (
            {} if mantle_melt_fraction is None else {"mantle_melt_fraction": mantle_melt_fraction}
        )

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
            **melt_fraction_kwargs,
        )
        model: EquilibriumModel = EquilibriumModel.from_state(
            planet, mass_constraints=mass_constraints
        )

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

    def disaggregate_H_into_D(self, element_values: ArrayLike, X_DH: ArrayLike) -> Array:
        """Maps per-element values (ordered per :data:`_COUPLING_ELEMENTS`) back onto the isocalc
        per-species layout (ordered per :data:`isofate.species.SYMBOLS`) - the inverse of
        `aggregate_D_into_H`.

        H is split into H and D with the D/H ratio `X_DH`, assuming the same D/H in every
        reservoir.

        Args:
            element_values: Per-element values (e.g. number of atoms), ordered per
                :data:`_COUPLING_ELEMENTS`
            X_DH: D/H ratio, as the fraction D / (H + D) [ndim]

        Returns:
            Per-species values ordered per :data:`isofate.species.SYMBOLS`
        """
        element_values = jnp.asarray(element_values, dtype=float)
        values: Array = (
            jnp.zeros(len(SYMBOLS)).at[jnp.array(_COUPLING_ELEMENT_INDICES)].set(element_values)
        )
        hydrogen: Array = element_values[_H_POSITION]

        return values.at[_H_INDEX].set((1 - X_DH) * hydrogen).at[_D_INDEX].set(X_DH * hydrogen)

    @staticmethod
    def mass_constraints_dict(mass_constraints: ArrayLike) -> dict[str, Array]:
        """Atmodeller's mass constraints, by element symbol.

        Args:
            mass_constraints: Total mass of each coupling element [kg], with the last axis ordered
                per `_COUPLING_ELEMENTS`

        Returns:
            Mass constraints [kg], by element symbol
        """
        mass_constraints = jnp.asarray(mass_constraints, dtype=float)

        return {element: mass_constraints[..., i] for i, element in enumerate(_COUPLING_ELEMENTS)}

    def batched_output(
        self,
        surface_temperature: ArrayLike,
        mantle_melt_fraction: ArrayLike,
        mass_constraints: ArrayLike,
        solution: ArrayLike,
    ) -> AtmodellerOutput:
        """The complete Atmodeller output for a set of equilibrations, as one batched solve.

        Builds a second, batched Atmodeller model with each equilibration's inputs (one batch row
        per equilibration) and solves it warm-started from each equilibration's converged
        solution, so it converges immediately to the same equilibria. Only for the output: the
        time integration uses the single-problem model (`equilibrium_model`). Outside jit (it
        constructs a model).

        Args:
            surface_temperature: Surface temperature used by Atmodeller for each equilibration [K]
            mantle_melt_fraction: Mantle melt fraction for each equilibration [ndim]
            mass_constraints: Element mass constraints for each equilibration [kg], shape
                ``(n, len(_COUPLING_ELEMENTS))``
            solution: Converged solution array of each equilibration, shape ``(n, 2 * species)``

        Returns:
            Atmodeller output with one batch row per equilibration
        """
        model: EquilibriumModel = self.construct_equilibrium_model(
            temperature=jnp.asarray(surface_temperature, dtype=float),
            mantle_melt_fraction=jnp.asarray(mantle_melt_fraction, dtype=float),
            mass_constraints=self.mass_constraints_dict(mass_constraints),
        )

        return model.solve(jnp.asarray(solution, dtype=float))

    def cold_guess(self) -> Array:
        """The default cold-start guess for the Atmodeller solve (all-NaN).

        dtype=jnp.float64 (x64 is enabled process-wide, see isofate/__init__.py) matters here:
        without it, jnp.full(..., jnp.nan) is weakly-typed, while a real solved `solution` array is
        not - eqx.filter_jit treats weak_type as part of the trace signature, so a weak cold guess
        and a strong warm guess would each force their own compile of update_solve_extract instead
        of sharing one.

        Returns:
            All-NaN solution array
        """
        return jnp.full(
            (
                self.equilibrium_model.parameters.batch_size,
                self.equilibrium_model.parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
            dtype=jnp.float64,
        )

    def mantle_melt_fraction(self, temperature: ArrayLike) -> Array:
        """Mantle melt fraction at the given surface temperature.

        Delegates to `Planet.mantle_melt_fraction` (the planet's cached melt-fraction grid
        interpolator, or its supplied override). The temperature is clipped to ``10``-``16000`` K,
        as the original IsoFATE coupler did, so it stays within the grid.

        Args:
            temperature: Surface temperature [K]

        Returns:
            Melt fraction of the silicate layer [ndim]
        """
        return self.parameters.system.planet.mantle_melt_fraction(jnp.clip(temperature, 10, 16000))

    @eqx.filter_jit
    def update_solve_extract(
        self,
        mantle_melt_fraction: Array,
        temperature: Array,
        mass_constraints: dict,
        base_solution_array: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Updates the model, solves, and extracts only what feeds back into the time
        integration, in one jitted call.

        The coupling only reads the element ``number_moles`` in the gas and the silicate melt for
        the six coupling elements, which need only ``log_number_moles`` and the (static) formula
        matrix - no equation of state, no activities - so the full ``to_dict`` output (activities,
        fractions, pressures, ...) is never built here. The complete output is built separately
        after the run (see `batched_output`).

        ``planet_mass``/``surface_radius`` are not updated here: both are fixed for the whole run
        and already set by ``construct_equilibrium_model``. ``update_state``/``update_constraints``
        traverse the whole model pytree, so fusing them with the solve in one jitted call pays
        that cost once, at trace time (atmodeller's own idiom, see its tests/test_retracing.py).
        An all-NaN `base_solution_array` is a cold start.

        Returns:
            `(gas_moles, melt_moles, solution, success)`: the moles of each coupling element
            (ordered per `_COUPLING_ELEMENTS`) in the gas and the melt, the solution array and
            whether the solve converged
        """
        model = self.equilibrium_model.update_state(
            mantle_melt_fraction=mantle_melt_fraction, temperature=temperature
        ).update_constraints(mass_constraints=mass_constraints)
        output = model.solve(base_solution_array)

        out_dict = OutputNamedArraysDict(output.parameters, output.multi_attempt_solution)
        gas_output = out_dict.get_phase_output_from_index(GAS_PHASE_INDEX)
        melt_output = out_dict.get_phase_output_from_index(SILICATE_MELT_PHASE_INDEX)

        gas_moles: list[Array] = []
        melt_moles: list[Array] = []
        for element in _COUPLING_ELEMENTS:
            gas_idx = gas_output.phase.species.get_element_index(element)
            melt_idx = melt_output.phase.species.get_element_index(element)
            if gas_idx == -1 or melt_idx == -1:
                raise ValueError(
                    f"Element {element!r} missing from the gas or silicate_melt phase species; "
                    "the extraction assumes both phases carry all six coupling elements."
                )
            gas_moles.append(gas_output.element_number_moles[0, gas_idx])
            melt_moles.append(melt_output.element_number_moles[0, melt_idx])

        return (
            jnp.stack(gas_moles),
            jnp.stack(melt_moles),
            output.solution,
            jnp.all(output.multi_attempt_solution.success),
        )

    def run(
        self,
        Rp,
        mu,
        y: ArrayLike,
        y_int: ArrayLike,
        initial_guess=None,
    ) -> AtmodellerResult:
        """Re-equilibrates the atmosphere and interior with Atmodeller.

        Args:
            Rp: Planet radius at the top of the atmosphere [m]
            mu: Mean atmospheric particle mass [kg]
            y: Atmospheric abundances [atoms], ordered per `isofate.species.SYMBOLS` (H, He, D, O,
                C, N, S)
            y_int: Interior (dissolved) abundances [atoms], ordered as `y`
            initial_guess: Previous solution to warm-start from. Defaults to ``None`` (cold start).

        Returns:
            The re-equilibrated atmospheric and interior abundances, with the surface
            temperatures, the Atmodeller inputs and the solution array for warm-starting the next
            call
        """
        # Fixed for the whole run; the planet values are the ones construct_equilibrium_model used
        Teq: Array = self.parameters.system.equilibrium_temperature
        Mp: Array = self.parameters.system.planet.mass
        rocky_radius: Array = self.parameters.system.planet.rocky_radius

        # TODO: P_surface is not extracted here? Required for self-consistency with Atmodeller?
        # FIXME: With no atmospheric thickness (Rp equal to the rocky radius, e.g.
        # IsocalcOptions.rad_evol=False), the descent from Rp to the surface has zero length and the
        # Atmodeller solve does not converge, so the coupling can't be used with a fixed radius.
        _, T_surface, _ = self.parameters.atmosphere.make_atmosphere_descent(
            Teq, mu, Rp, Mp, rocky_radius
        )

        surface_temperature = jnp.minimum(self.max_surface_temperature, T_surface)
        # Prescribed on the Planet (`Planet.mantle_melt_fraction`), or interpolated from the
        # uncapped surface temperature (as in the original IsoFATE coupler)
        mantle_melt_fraction: Array = self.mantle_melt_fraction(T_surface)

        # Total atoms per coupling element (atmosphere + interior), with D folded into H:
        # Atmodeller has no separate D reservoir, so D atoms count as H atoms (atom-conserving, as
        # in the original IsoFATE coupler)
        total: Array = jnp.asarray(y) + jnp.asarray(y_int)
        element_atoms: Array = self.aggregate_D_into_H(total)
        # D/H ratio of the whole inventory, D / (H + D), assuming the same D/H in the atmosphere
        # and interior; zero if there is no hydrogen
        X_DH: Array = safe_divide(total[_D_INDEX], total[_H_INDEX] + total[_D_INDEX])
        # Mass per atom [kg] of each coupling element, from the IsoFATE species
        # (jnp.asarray first: the species' masses are a NumPy array, which can't be indexed by a
        # JAX array under jit)
        atomic_masses: Array = jnp.asarray(self.parameters.isofate_species.atomic_masses)[
            jnp.array(_COUPLING_ELEMENT_INDICES)
        ]
        # Total mass [kg] of each element, floored so the solve stays well-posed for trace
        # inventories
        MASS_CUTOFF: float = 1e9
        element_total_mass: Array = jnp.maximum(MASS_CUTOFF, element_atoms * atomic_masses)
        mass_constraints: dict[str, Array] = self.mass_constraints_dict(element_total_mass)

        # eqx.filter_jit only treats actual jax/numpy arrays as traced (dynamic) inputs; plain
        # Python floats are treated as *static* arguments (hashed by value), which would force a
        # full retrace on every call since these values change every timestep. Casting to jnp
        # arrays here keeps update_solve_extract's compiled trace reused across calls.
        mantle_melt_fraction = jnp.asarray(mantle_melt_fraction)
        surface_temperature = jnp.asarray(surface_temperature)
        mass_constraints = {key: jnp.asarray(value) for key, value in mass_constraints.items()}

        cold_guess: Array = self.cold_guess()

        def solve(guess: Array):
            return self.update_solve_extract(
                mantle_melt_fraction, surface_temperature, mass_constraints, guess
            )

        if initial_guess is None:
            # Cold start: retrying from the same cold guess could not help, so solve once
            gas_moles, melt_moles, solution, success = solve(cold_guess)
        else:
            # Warm-starting from the previous call's converged solution (a tiny physical
            # perturbation away) avoids re-solving the full nonlinear equilibrium system from
            # scratch every call; fall back to the cold guess if the warm start fails to converge.
            # jax.lax.cond rather than a Python `if` on `success`, which would fail on traced
            # arrays.
            warm = solve(initial_guess)
            gas_moles, melt_moles, solution, success = jax.lax.cond(
                warm[3], lambda warm: warm, lambda warm: solve(cold_guess), warm
            )

        # Atoms of each coupling element in the gas and the melt, split back into H and D
        y_new: Array = self.disaggregate_H_into_D(gas_moles * const.avogadro, X_DH)
        y_int_new: Array = self.disaggregate_H_into_D(melt_moles * const.avogadro, X_DH)

        return AtmodellerResult(
            y=y_new,
            y_int=y_int_new,
            atmosphere_mass=self.parameters.atmosphere_mass(y_new),
            surface_temperature=T_surface,
            surface_temperature_atmodeller=surface_temperature,
            mantle_melt_fraction=mantle_melt_fraction,
            mass_constraints=element_total_mass,
            solution=solution,
            success=success,
        )

    def initial_state(self) -> CouplerState:
        """The state before the first re-equilibration: an empty interior and a cold guess.

        Returns:
            Initial coupler state
        """
        return CouplerState(
            y_int=jnp.zeros(len(SYMBOLS)),
            solution=self.cold_guess(),
            surface_temperature=jnp.array(0.0),
            surface_temperature_atmodeller=jnp.array(0.0),
            mantle_melt_fraction=jnp.array(0.0),
            mass_constraints=jnp.zeros(len(_COUPLING_ELEMENTS)),
        )

    def reequilibrate(
        self, carry: CouplerState, t: ArrayLike, y: Array
    ) -> tuple[CouplerState, Array]:
        """Re-equilibrates the atmosphere and interior at the fixed time `t` - the `on_mass_lost`
        hook for `isofate.integrators.Integrator`.

        The escape radius and mean particle mass are derived from `(t, y)`, the interior reservoir
        and warm start come from `carry`. An all-NaN `carry.solution` (see `initial_state`) is
        equivalent to a cold start. Pure JAX, so it can run inside the jitted segment loop.

        Args:
            carry: Coupler state from the previous re-equilibration
            t: Time [s]
            y: Atmospheric abundances [atoms], ordered per `isofate.species.SYMBOLS`

        Returns:
            `(carry, y_new)`: the updated coupler state and the re-equilibrated atmospheric
            abundances
        """
        result: AtmodellerResult = self.run(
            escape_radius(self.parameters, y, t),
            self.parameters.atmosphere_mean_mu(y),
            y,
            carry.y_int,
            initial_guess=carry.solution,
        )
        # Record the first failed solve, to raise once outside jit (see check_converged)
        first_failure_time = jnp.where(
            jnp.isinf(carry.first_failure_time) & ~result.success,
            jnp.asarray(t, dtype=float),
            carry.first_failure_time,
        )
        new_carry = CouplerState(
            y_int=result.y_int,
            solution=result.solution,
            surface_temperature=jnp.asarray(result.surface_temperature, dtype=float),
            surface_temperature_atmodeller=jnp.asarray(
                result.surface_temperature_atmodeller, dtype=float
            ),
            mantle_melt_fraction=jnp.asarray(result.mantle_melt_fraction, dtype=float),
            mass_constraints=result.mass_constraints,
            first_failure_time=first_failure_time,
        )

        return new_carry, result.y

    @staticmethod
    def check_converged(state: CouplerState) -> None:
        """Raises if any re-equilibration recorded in `state` failed to converge. Call outside jit.

        Args:
            state: Coupler state after the re-equilibrations to check

        Raises:
            RuntimeError: If a solve failed to converge
        """
        t_fail = float(state.first_failure_time)
        if np.isfinite(t_fail):
            raise RuntimeError(
                f"Atmodeller failed to converge (warm start and cold fallback) at "
                f"t = {t_fail * const.s2yr:.6g} yr; its result does not conserve the elements, "
                "so the run is stopped"
            )
