# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from atmodeller import ChemicalSpecies, EquilibriumModel, Planet, ReservoirSpecies
from atmodeller.constants import GAS_PHASE_INDEX, SILICATE_MELT_PHASE_INDEX
from atmodeller.output_base import OutputNamedArraysDict
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from jax.typing import ArrayLike

from isofate.constants import const
from isofate.isofunks import *
from isofate.isojax import make_atmosphere_descent_jax
from isofate.mantle_iron import MantleIronState
from isofate.orbit_params import *
from isofate.species import DEFAULT_SPECIES, SYMBOLS

solubility_models = get_solubility_models()

_COUPLING_ELEMENTS: tuple[str, ...] = ("H", "He", "O", "C", "N", "S")
"""Elements AtmodellerCoupler's `results` dict reports; must match its mass_constraints keys."""

# Precomputed once: which isofate.species.SYMBOLS index feeds each _COUPLING_ELEMENTS slot, and
# where "D" and "H" themselves sit in each ordering - used by aggregate_D_into_H below.
_COUPLING_ELEMENT_INDICES: tuple[int, ...] = tuple(
    SYMBOLS.index(element) for element in _COUPLING_ELEMENTS
)
_D_INDEX: int = SYMBOLS.index("D")
_H_POSITION: int = _COUPLING_ELEMENTS.index("H")


@dataclass(frozen=True)
class TrackedGasSpecies:
    """One isofate-tracked gas-phase species, derived from atmodeller's own species lists."""

    gas_name: str
    """Atmodeller canonical gas-phase name, e.g. "H2_g", "O2S_g"."""
    label: str
    """Isofate's human-readable output label, e.g. "H2", "SO2"."""
    melt_name: str | None
    """Atmodeller canonical melt-phase name, or None if this species has no melt reservoir."""


def get_tracked_gas_species(
    interior_atmosphere: EquilibriumModel,
) -> tuple[TrackedGasSpecies, ...]:
    """Ordered set of gas-phase species isofate tracks as molecular output - every gas species
    except He, which isofate tracks separately as one of its own 7 core H/He/D/O/C/N/S species.

    Derived directly from interior_atmosphere's own species objects, which already carry both
    the original formula (used as isofate's output label, e.g. "SO2") and the Hill-canonicalized
    name atmodeller actually indexes its solve output by (e.g. "O2S_g" - atmodeller internally
    derives this from molmass, see ChemicalSpeciesData) - so no separate Hill-notation
    conversion or per-species override table is needed on isofate's side.
    """
    gas_species = interior_atmosphere.parameters.reaction_system.phase_system.gas.species.species
    melt_names = interior_atmosphere.parameters.reaction_system.phase_system.phases[
        SILICATE_MELT_PHASE_INDEX
    ].species_names
    melt_by_stem = {name.removesuffix("_d"): name for name in melt_names}

    tracked = []
    for sp in gas_species:
        if sp.data.formula == "He":
            continue
        tracked.append(
            TrackedGasSpecies(
                sp.data.name, sp.data.formula, melt_by_stem.get(sp.data.hill_formula)
            )
        )
    return tuple(tracked)


def aggregate_D_into_H(values: ArrayLike) -> np.ndarray:
    """Maps an isocalc per-species array (ordered per isofate.species.SYMBOLS: H, He, D, O, C,
    N, S) onto the layout AtmodellerCoupler expects (`_COUPLING_ELEMENTS`: H, He, O, C, N, S).

    Atmodeller has no separate D reservoir, so D's abundance is folded into H's - this is the one
    place isocalc's per-species arrays (`y`, `N_X_int`) and AtmodellerCoupler's per-element
    arguments disagree on how many species there are.

    Args:
        values: length-7 array/sequence ordered per isofate.species.SYMBOLS

    Returns:
        length-6 array ordered per `_COUPLING_ELEMENTS`, meant to be unpacked directly into
        AtmodellerCoupler's N_<element>_atm/N_<element>_int positional arguments, e.g.
        ``AtmodellerCoupler(..., *aggregate_D_into_H(y), *aggregate_D_into_H(N_X_int), ...)``.
    """
    values = np.asarray(values, dtype=float)
    coupled = values[list(_COUPLING_ELEMENT_INDICES)]
    coupled[_H_POSITION] += values[_D_INDEX]
    return coupled


# Module-level so the compiled trace is cached and reused across AtmodellerCoupler calls, same
# reasoning as _update_solve_extract below. make_atmosphere_descent_jax (isojax.py) integrates
# the same ODE as the original (now-removed) isofunks.make_atmosphere_descent with diffrax
# instead of a plain-Python/NumPy loop; validated to agree with the original to ~1e-12 to 1e-14
# relative before that original was deleted as dead code.
#
# eqx.filter_jit was tried here instead but measured slower: its default filter treats a plain
# Python float as static, so it only avoids retracing every call if every arg is pre-cast to a
# jnp array at the call site - and that cast plus eqx.filter_jit's own partition/combine overhead
# together cost more (~0.07 ms/call) than plain jax.jit accepting raw Python floats directly
# (~0.02 ms/call).
_make_atmosphere_descent_jit = jax.jit(make_atmosphere_descent_jax)


@eqx.filter_jit
def _update_solve_extract(
    interior_atmosphere: EquilibriumModel,
    mantle_melt_fraction,
    temperature,
    mass_constraints: dict,
    base_solution_array,
):
    """Fuses state/constraint updates, the solve, and output extraction into one jitted call.

    ``planet_mass``/``surface_radius`` are deliberately not updated here: both are fixed for the
    whole isocalc run and already set on ``interior_atmosphere`` at construction time
    (``build_atmodeller``), so leaving them out of ``update_state`` (which defaults each omitted
    field to "no change") avoids re-asserting a value that never differs from what the model
    already holds.

    ``update_state``/``update_constraints`` are built on ``eqx.tree_at``, which does several full
    Python-level traversals of the whole model pytree per call (O(num_leaves); ~3479 leaves here)
    - calling them un-jitted, once per AtmodellerCoupler call, cost ~40ms/call regardless of how
    much the actual solve needed to change. Wrapping the whole update+solve+extract sequence in
    one eqx.filter_jit call instead pays that Python-level tree rebuilding cost once, at trace
    time; every later call just replays the compiled XLA graph (~9ms/call: solve + extraction).
    This mirrors atmodeller's own tested idiom for this exact situation (see
    atmodeller/tests/test_retracing.py's `call_solver`/`workflow` pattern).

    `solve_with_default()` is exactly `solve(all-NaN array of this shape)` (see
    atmodeller/classes.py), so passing an all-NaN `base_solution_array` reproduces the old
    cold-start path without needing a separate branch inside this jitted function.
    """
    model = interior_atmosphere.update_state(
        mantle_melt_fraction=mantle_melt_fraction,
        temperature=temperature,
    ).update_constraints(mass_constraints=mass_constraints)
    output = model.solve(base_solution_array)
    sol = output.to_dict(output_format="elements_species", to_numpy=False)

    return sol, output.solution, output.multi_attempt_solution.success


@eqx.filter_jit
def _update_solve_extract_narrow(
    interior_atmosphere: EquilibriumModel,
    mantle_melt_fraction,
    temperature,
    mass_constraints: dict,
    base_solution_array,
):
    """Like ``_update_solve_extract``, but skips every diagnostic AtmodellerCoupler never reads.

    ``to_dict("elements_species")`` computes, for *every* species in *every* phase: activity (a
    real-gas EOS volume root-find for non-ideal species), mass/mole fractions, partial pressure,
    phase volume, log10dIW, and metallicity. AtmodellerCoupler's ``results`` dict - the thing the
    isocalc per-timestep loop actually consumes - only ever reads element ``number_moles`` (gas +
    silicate_melt) for the six mass-constrained elements, the gas phase's total mass, and O2_g's
    number_moles (for the iron-buffer branches). All of those need only ``log_number_moles`` and
    the (static) formula matrix - no EOS, no activity. Building only these keeps the traced graph
    much smaller without changing any value AtmodellerCoupler actually uses from this path.

    Not a substitute for ``_update_solve_extract`` when species-level diagnostics are wanted
    (``save_molecules=True``, or the end-of-run full snapshot) - those still need the full
    ``to_dict`` output.
    """
    model = interior_atmosphere.update_state(
        mantle_melt_fraction=mantle_melt_fraction,
        temperature=temperature,
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
                f"Element {element!r} missing from the gas or silicate_melt phase species; the "
                "narrow extraction assumes both phases carry all six coupling elements."
            )
        sol[element] = {
            "gas": {"number_moles": gas_output.element_number_moles[..., gas_idx : gas_idx + 1]},
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


# NOTE: Rebuilding the whole Atmodeller on a new class atmodeller_coupler_new.py
def build_atmodeller(
    planet_mass: ArrayLike,
    *,
    core_mass_fraction=earth.core_mass_fraction,
    surface_radius: ArrayLike = earth.radius,
    temperature=2000,
) -> EquilibriumModel:
    """Builds an Atmodeller model for the interior-atmosphere coupling.

    Args:
        planet_mass (ArrayLike): Mass of the planet in kg
        core_mass_fraction: Fraction of the planet's mass that is in the core. Defaults to Earth's
            core mass fraction.
        surface_radius (ArrayLike): Radius of the planet's surface in m. Defaults to Earth's
            radius.
        temperature: Temperature of the planet in K. Defaults to 2000 K.

    Returns:
        EquilibriumModel: An Atmodeller equilibrium model for the interior-atmosphere coupling
    """
    # Gas-phase species (v2preview drops per-species solubility; dissolution is now modeled via a
    # separate ReservoirSpecies tied to the silicate melt phase, matched by formula below)
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

    # Constructed once. planet_mass and surface_radius are fixed for the whole isocalc run and
    # never updated afterward; per-timestep state (temperature, melt fraction) and mass
    # constraints are applied via .update_state()/.update_constraints() in AtmodellerCoupler
    planet: Planet = Planet.from_species(
        gas_species,
        planet_mass=planet_mass,
        core_mass_fraction=core_mass_fraction,
        surface_radius=surface_radius,
        temperature=temperature,
        silicate_melt_species=melt_species,
    )
    model: EquilibriumModel = EquilibriumModel.from_state(planet)

    return model


def _passthrough_o(sol: dict, results: dict) -> None:
    """No Fe-O2 reaction this call (mass_Fe2 depleted, or feature disabled) - O just passes
    straight from the equilibrium solve to results, unmodified."""
    results["N_O_atm"] = sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_O_int"] = sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro


def _react_mantle_fe2_with_o2(mass_Fe2: float, sol: dict, results: dict) -> float:
    """Oxidizes Fe2+ -> Fe3+ against atmospheric O2, mutating sol's/results' O-bearing entries in
    place, and returns the mass_Fe2 remaining after reaction (>= 0)."""
    n_Fe2 = mass_Fe2 / const.M_Fe - 4 * sol["O2_g"]["gas"]["number_moles"][0][0]
    n_O2_atm = sol["O2_g"]["gas"]["number_moles"][0][0] - 0.25 * (mass_Fe2 / const.M_Fe - n_Fe2)
    delta_n_O2_atm = 0.25 * (mass_Fe2 / const.M_Fe - n_Fe2)
    sol["O2_g"]["gas"]["number_moles"] = np.array([[np.max([n_O2_atm, 0])]])
    results["N_O_atm"] = (
        sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
        - 2 * delta_n_O2_atm * const.avogadro
    )
    sol["O"]["gas"]["number_moles"] = np.array([[results["N_O_atm"] / const.avogadro]])
    results["N_O_int"] = (
        sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
        + 2 * delta_n_O2_atm * const.avogadro
    )
    sol["O"]["silicate_melt"]["number_moles"] = np.array([[results["N_O_int"] / const.avogadro]])
    return np.max([n_Fe2 * const.M_Fe, 0])


def AtmodellerCoupler(
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
    """
    Args:
        full_output: When ``False``, uses the narrow extraction path (element number_moles,
            gas mass, O2_g number_moles only) instead of the full ``elements_species`` output.
            The periodic per-timestep calls in ``isocalc`` never need more than that unless
            ``save_molecules=True``; the end-of-run full snapshot always needs the full path.
            Defaults to ``True`` to preserve existing behavior for other callers.
    """

    results = {}
    gamma = 7 / 5
    # Planet mass and rocky-component radius are fixed for the whole isocalc run and already set
    # on interior_atmosphere at construction time (build_atmodeller) - read them back here rather
    # than threading them through every call as separate arguments.
    Mp = float(interior_atmosphere.parameters.state.background_planet_mass)
    radius_rocky = float(interior_atmosphere.parameters.state.surface_radius)
    # Converted back to plain Python floats immediately: everything downstream in this function
    # (and the isocalc loop calling it) is plain Python/NumPy, not JAX. Matm is discarded here -
    # AtmodellerCoupler computes M_atm from the equilibrium solve's own output instead.
    _, T_surface_j, P_surface_j = _make_atmosphere_descent_jit(
        Teq, mu, Rp, Mp, gamma, radius_rocky
    )
    T_surface, P_surface = float(T_surface_j), float(P_surface_j)
    surface_temperature: float = np.min([6000, T_surface])  # K
    if melt_fraction != False:
        mantle_melt_fraction: float = melt_fraction
    elif melt_fraction == False:
        mantle_melt_fraction: float = MeltFraction(Mp, np.clip(T_surface, 10, 16000))

    # element masses
    mass_H: float = (N_H_atm + N_H_int) * const.mu_H
    mass_O: float = (N_O_atm + N_O_int) * const.mu_O
    mass_C: float = (N_C_atm + N_C_int) * const.mu_C
    mass_He: float = (N_He_atm + N_He_int) * const.mu_He
    mass_N: float = (N_N_atm + N_N_int) * const.mu_N
    mass_S: float = (N_S_atm + N_S_int) * const.mu_S
    mass_constraints = {
        "H": mass_H,
        "O": mass_O,
        "C": mass_C,
        "He": mass_He,
        "N": mass_N,
        "S": mass_S,
    }

    MASS_CUTOFF: float = 10**9
    mass_constraints = {
        "H": max(MASS_CUTOFF, mass_H),
        "O": max(MASS_CUTOFF, mass_O),
        "C": max(MASS_CUTOFF, mass_C),
        "He": max(MASS_CUTOFF, mass_He),
        "N": max(MASS_CUTOFF, mass_N),
        "S": max(MASS_CUTOFF, mass_S),
    }

    # Warm-starting from the previous call's converged solution (a tiny physical perturbation
    # away) avoids re-solving the full nonlinear equilibrium system from scratch every call; fall
    # back to the default cold-start (all-NaN) guess if that warm start fails to converge.
    if initial_guess is None:
        # dtype=jnp.float64 (x64 is enabled process-wide, see isofate/__init__.py) matters here:
        # without it, jnp.full(..., jnp.nan) is weakly-typed, while a real solved `solution`
        # array (returned below) is not - eqx.filter_jit treats weak_type as part of the trace
        # signature, so a weak first guess and a strong later guess would each force their own
        # ~2s compile of _update_solve_extract instead of sharing one.
        base_solution_array = jnp.full(
            (
                interior_atmosphere.parameters.batch_size,
                interior_atmosphere.parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
            dtype=jnp.float64,
        )
    else:
        base_solution_array = initial_guess

    # eqx.filter_jit only treats actual jax/numpy arrays as traced (dynamic) inputs; plain Python
    # floats are treated as *static* arguments (hashed by value), which would force a full
    # retrace on every call since these values change every timestep. Casting to jnp arrays here
    # keeps _update_solve_extract's compiled trace reused across calls.
    mantle_melt_fraction_j = jnp.asarray(mantle_melt_fraction)
    surface_temperature_j = jnp.asarray(surface_temperature)
    mass_constraints_j = {key: jnp.asarray(value) for key, value in mass_constraints.items()}

    extract_fn = _update_solve_extract if full_output else _update_solve_extract_narrow

    sol_jax, solution, success = extract_fn(
        interior_atmosphere,
        mantle_melt_fraction_j,
        surface_temperature_j,
        mass_constraints_j,
        base_solution_array,
    )
    if not np.all(np.asarray(success)):
        sol_jax, solution, success = extract_fn(
            interior_atmosphere,
            mantle_melt_fraction_j,
            surface_temperature_j,
            mass_constraints_j,
            jnp.full_like(base_solution_array, jnp.nan),
        )
    # Entries that need in-place-style updates below are replaced wholesale (new array of the
    # same (1, 1) shape) rather than item-assigned; jax.device_get gives plain (writable) numpy
    # arrays, same as the to_numpy=True the jitted helper skips (jit can't emit numpy directly).
    sol = jax.device_get(sol_jax)
    results["N_H_atm"] = sol["H"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_H_int"] = sol["H"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_He_atm"] = sol["He"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_He_int"] = sol["He"]["silicate_melt"]["number_moles"][0][0] * const.avogadro

    if mantle_iron_state is None or mantle_iron_state.mass_Fe2 == 0:
        _passthrough_o(sol, results)
    elif mantle_iron_state.config.reaction_type == "dynamic":
        # Read the OLD Fe2+ fraction before mass_Fe is overwritten below - mass_Fe is recomputed
        # fresh every call from the current melt fraction, but the fraction of it that's still
        # Fe2+ (as opposed to already-oxidized) must persist across calls.
        old_x_Fe2 = mantle_iron_state.x_Fe2
        new_mass_Fe = mantle_melt_fraction * Mp * mantle_iron_state.config.fe_mass_fraction
        if new_mass_Fe > 0:
            final_mass_Fe2 = _react_mantle_fe2_with_o2(old_x_Fe2 * new_mass_Fe, sol, results)
            mantle_iron_state = dataclasses.replace(
                mantle_iron_state, mass_Fe=new_mass_Fe, mass_Fe2=final_mass_Fe2
            )
        else:
            _passthrough_o(sol, results)
            mantle_iron_state = dataclasses.replace(mantle_iron_state, mass_Fe=new_mass_Fe)
    else:  # "static": mass_Fe is fixed for the whole run, only mass_Fe2 evolves
        final_mass_Fe2 = _react_mantle_fe2_with_o2(mantle_iron_state.mass_Fe2, sol, results)
        mantle_iron_state = dataclasses.replace(mantle_iron_state, mass_Fe2=final_mass_Fe2)
    results["N_C_atm"] = sol["C"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_C_int"] = sol["C"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_N_atm"] = sol["N"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_N_int"] = sol["N"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_S_atm"] = sol["S"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_S_int"] = sol["S"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["M_atm"] = sol["gas"]["phase"]["mass"][0][0]
    results["T_surface"] = T_surface
    results["T_surface_atmod"] = surface_temperature

    return results, sol, mantle_iron_state, solution


@dataclass(frozen=True)
class AtmodellerStepResult:
    """Result of one per-timestep AtmodellerCoupler call, already unpacked into isocalc's y/
    isofate_species_abund_int layout (ordered per isofate.species.SYMBOLS: H, He, D, O, C, N, S).
    """

    y: np.ndarray
    isofate_species_abund_int: np.ndarray
    M_atm: float
    T_surf_analytic: float
    T_surf_atmod: float
    mantle_iron_state: MantleIronState | None
    atmod_initial_guess: ArrayLike | None
    atmod_full: dict


def run_atmodeller_step(
    T,
    radius_p,
    mu,
    melt_fraction_override,
    mantle_iron_state: MantleIronState | None,
    y: np.ndarray,
    isofate_species_abund_int: np.ndarray,
    X_DH: float,
    interior_atmosphere: EquilibriumModel,
    atmod_initial_guess,
    full_output: bool,
) -> AtmodellerStepResult:
    """Runs one AtmodellerCoupler equilibrium solve and unpacks its results into isocalc's
    per-species y/isofate_species_abund_int layout - the same unpacking previously inlined in
    isocalc's `n % n_atmodeller == 0` branch. Does not touch T_surf_analytic/T_surf_atmod's
    carry-forward-when-not-run behavior - that stays in isocalc, which only calls this when it has
    already decided to run atmodeller this step.
    """
    atmod_results, atmod_full, mantle_iron_state, atmod_initial_guess = AtmodellerCoupler(
        T,
        radius_p,
        mu,
        melt_fraction_override,
        mantle_iron_state,
        *aggregate_D_into_H(y),
        *aggregate_D_into_H(isofate_species_abund_int),
        interior_atmosphere,
        initial_guess=atmod_initial_guess,
        full_output=full_output,
    )

    isofate_species_abund_int = isofate_species_abund_int.copy()
    isofate_species_abund_int[0] = atmod_results["N_H_int"] * (1 - X_DH)
    isofate_species_abund_int[2] = atmod_results["N_H_int"] * X_DH
    isofate_species_abund_int[1] = atmod_results["N_He_int"]
    isofate_species_abund_int[3] = atmod_results["N_O_int"]
    isofate_species_abund_int[4] = atmod_results["N_C_int"]
    isofate_species_abund_int[5] = atmod_results["N_N_int"]
    isofate_species_abund_int[6] = atmod_results["N_S_int"]

    y = y.copy()
    if atmod_results["N_H_atm"] == 0:
        y[0] = 0
        y[2] = 0
    else:
        Y3 = X_DH * atmod_results["N_H_atm"]
        Y1 = (1 - X_DH) * atmod_results["N_H_atm"]
        y[0] = Y1
        y[2] = Y3
    y[1] = atmod_results["N_He_atm"]
    y[3] = atmod_results["N_O_atm"]
    y[4] = atmod_results["N_C_atm"]
    y[5] = atmod_results["N_N_atm"]
    y[6] = atmod_results["N_S_atm"]

    # Derived from y (mass-conservation identity: the gas phase's total mass is exactly the sum
    # of its constituent atoms' masses) rather than atmodeller's own `sol["gas"]["phase"]["mass"]`
    # - that value is computed before the H/D disaggregation above, so it treats the aggregated
    # H+D count as pure H (one atomic mass) instead of the correct per-isotope split. The
    # difference is the D/H mass fraction (~2e-5, protosolar) - negligible, but this version is
    # exact rather than an approximation.
    M_atm = float(np.dot(y, DEFAULT_SPECIES.atomic_masses))

    return AtmodellerStepResult(
        y=y,
        isofate_species_abund_int=isofate_species_abund_int,
        M_atm=M_atm,
        T_surf_analytic=atmod_results["T_surface"],
        T_surf_atmod=atmod_results["T_surface_atmod"],
        mantle_iron_state=mantle_iron_state,
        atmod_initial_guess=atmod_initial_guess,
        atmod_full=atmod_full,
    )


# Keys of isocalc's "atmodeller_final" full-output snapshot - single source of truth shared by
# extract_full_output (the real, equilibrium-solve-derived values) and nan_full_output (the
# early-exit placeholder, used when isocalc terminates before ever reaching a real snapshot), so
# the two can never drift out of sync.
ATMOD_FULL_OUTPUT_KEYS: tuple[str, ...] = (
    "H2O_atm",
    "H2O_mantle",
    "H2_atm",
    "H2_mantle",
    "O2_atm",
    "O2_mantle",
    "CO_atm",
    "CO_mantle",
    "CO2_atm",
    "CO2_mantle",
    "CH4_atm",
    "CH4_mantle",
    "N2_atm",
    "N2_mantle",
    "S2_atm",
    "S2_mantle",
    "H2O4S_atm",
    "H2O4S_mantle",
    "SO2_atm",
    "He_mantle",
    "O2_fugacity",
    "log10dIW_1_bar",
)


def extract_full_output(atmod_sol: dict) -> dict[str, float]:
    """Extracts the full per-molecule/fugacity diagnostic snapshot from one AtmodellerCoupler
    solve's raw `sol` output (see AtmodellerCoupler's `full_output=True` path), keyed per
    ATMOD_FULL_OUTPUT_KEYS.
    """
    extractors: dict[str, Callable[[dict], float]] = {
        "H2O_atm": lambda sol: sol["H2O_g"]["gas"]["number_moles"][0][0],
        "H2O_mantle": lambda sol: sol["H2O_d"]["silicate_melt"]["number_moles"][0][0],
        "H2_atm": lambda sol: sol["H2_g"]["gas"]["number_moles"][0][0],
        "H2_mantle": lambda sol: sol["H2_d"]["silicate_melt"]["number_moles"][0][0],
        "O2_atm": lambda sol: sol["O2_g"]["gas"]["number_moles"][0][0],
        "O2_mantle": lambda sol: 0.0,  # O2 has no solubility model / melt reservoir
        "CO_atm": lambda sol: sol["CO_g"]["gas"]["number_moles"][0][0],
        "CO_mantle": lambda sol: sol["CO_d"]["silicate_melt"]["number_moles"][0][0],
        "CO2_atm": lambda sol: sol["CO2_g"]["gas"]["number_moles"][0][0],
        "CO2_mantle": lambda sol: sol["CO2_d"]["silicate_melt"]["number_moles"][0][0],
        "CH4_atm": lambda sol: sol["CH4_g"]["gas"]["number_moles"][0][0],
        "CH4_mantle": lambda sol: sol["CH4_d"]["silicate_melt"]["number_moles"][0][0],
        "N2_atm": lambda sol: sol["N2_g"]["gas"]["number_moles"][0][0],
        "N2_mantle": lambda sol: sol["N2_d"]["silicate_melt"]["number_moles"][0][0],
        "S2_atm": lambda sol: sol["S2_g"]["gas"]["number_moles"][0][0],
        "S2_mantle": lambda sol: sol["S2_d"]["silicate_melt"]["number_moles"][0][0],
        "H2O4S_atm": lambda sol: sol["H2O4S_g"]["gas"]["number_moles"][0][0],
        "H2O4S_mantle": lambda sol: 0.0,  # H2O4S has no solubility model / melt reservoir
        "SO2_atm": lambda sol: sol["O2S_g"]["gas"]["number_moles"][0][0],
        "He_mantle": lambda sol: sol["He_d"]["silicate_melt"]["number_moles"][0][0],
        "O2_fugacity": lambda sol: sol["O2_g"]["gas"]["activity"][0][0],
        "log10dIW_1_bar": lambda sol: sol["gas"]["phase"]["log10dIW_1_bar"][0][0],
    }
    return {key: extractors[key](atmod_sol) for key in ATMOD_FULL_OUTPUT_KEYS}


def nan_full_output() -> dict[str, float]:
    """NaN-filled placeholder for isocalc's "atmodeller_final" full-output snapshot, used when a
    run terminates early (e.g. the entire atmosphere is lost) before ever reaching a real
    equilibrium-solve snapshot. Shares ATMOD_FULL_OUTPUT_KEYS with extract_full_output so the two
    can never drift out of sync.
    """
    return {key: np.nan for key in ATMOD_FULL_OUTPUT_KEYS}
