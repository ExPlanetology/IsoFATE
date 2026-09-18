# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE."""

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
from isofate.orbit_params import *
from isofate.species import SYMBOLS

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
    planet_mass,
    mantle_melt_fraction,
    surface_radius,
    temperature,
    mass_constraints: dict,
    base_solution_array,
):
    """Fuses state/constraint updates, the solve, and output extraction into one jitted call.

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
        planet_mass=planet_mass,
        mantle_melt_fraction=mantle_melt_fraction,
        surface_radius=surface_radius,
        temperature=temperature,
    ).update_constraints(mass_constraints=mass_constraints)
    output = model.solve(base_solution_array)
    sol = output.to_dict(output_format="elements_species", to_numpy=False)
    return sol, output.solution, output.multi_attempt_solution.success


@eqx.filter_jit
def _update_solve_extract_narrow(
    interior_atmosphere: EquilibriumModel,
    planet_mass,
    mantle_melt_fraction,
    surface_radius,
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
        planet_mass=planet_mass,
        mantle_melt_fraction=mantle_melt_fraction,
        surface_radius=surface_radius,
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
    CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
    CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
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
    CO_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO", solubility=solubility_models["CO_basalt_yoshioka19"]
    )
    CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO2", solubility=solubility_models["CO2_basalt_dixon95"]
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

    # Constructed once; per-timestep state (temperature, melt fraction, radius) and mass
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


def AtmodellerCoupler(
    Teq,
    Mp,
    Rp,
    mu,
    melt_fraction,
    mantle_iron_dict,
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
    radius_rocky: ArrayLike,
    initial_guess=None,
    full_output: bool = True,
):
    """
    Args:
        radius_rocky: Planet rocky-component radius [m] (``Planet.rocky_radius``), the single
            source of truth for this quantity - passed in rather than recomputed here so callers
            only ever compute it once.
        full_output: When ``False``, uses the narrow extraction path (element number_moles,
            gas mass, O2_g number_moles only) instead of the full ``elements_species`` output.
            The periodic per-timestep calls in ``isocalc`` never need more than that unless
            ``save_molecules=True``; the end-of-run full snapshot always needs the full path.
            Defaults to ``True`` to preserve existing behavior for other callers.
    """

    results = {}
    gamma = 7 / 5
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
    planet_mass: float = Mp
    surface_radius: float = radius_rocky

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
    planet_mass_j = jnp.asarray(planet_mass)
    mantle_melt_fraction_j = jnp.asarray(mantle_melt_fraction)
    surface_radius_j = jnp.asarray(surface_radius)
    surface_temperature_j = jnp.asarray(surface_temperature)
    mass_constraints_j = {key: jnp.asarray(value) for key, value in mass_constraints.items()}

    extract_fn = _update_solve_extract if full_output else _update_solve_extract_narrow

    sol_jax, solution, success = extract_fn(
        interior_atmosphere,
        planet_mass_j,
        mantle_melt_fraction_j,
        surface_radius_j,
        surface_temperature_j,
        mass_constraints_j,
        base_solution_array,
    )
    if not np.all(np.asarray(success)):
        sol_jax, solution, success = extract_fn(
            interior_atmosphere,
            planet_mass_j,
            mantle_melt_fraction_j,
            surface_radius_j,
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

    if mantle_iron_dict:
        if mantle_iron_dict["type"] == "dynamic":
            if mantle_iron_dict["mass_Fe2"] == 0:
                results["N_O_atm"] = sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
                results["N_O_int"] = (
                    sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
                )
            else:
                mantle_iron_dict["mass_Fe"] = (
                    mantle_melt_fraction * Mp * mantle_iron_dict["Fe_mass_fraction"]
                )
                if mantle_iron_dict["mass_Fe"] > 0:
                    mantle_iron_dict["mass_Fe2"] = (
                        mantle_iron_dict["X_Fe2"] * mantle_iron_dict["mass_Fe"]
                    )
                    n_Fe2 = (
                        mantle_iron_dict["mass_Fe2"] / const.M_Fe
                        - 4 * sol["O2_g"]["gas"]["number_moles"][0][0]
                    )  # oxidize Fe2+ to Fe3+
                    n_O2_atm = sol["O2_g"]["gas"]["number_moles"][0][0] - 0.25 * (
                        mantle_iron_dict["mass_Fe2"] / const.M_Fe - n_Fe2
                    )
                    delta_n_O2_atm = 0.25 * (mantle_iron_dict["mass_Fe2"] / const.M_Fe - n_Fe2)
                    sol["O2_g"]["gas"]["number_moles"] = np.array([[np.max([n_O2_atm, 0])]])
                    results["N_O_atm"] = (
                        sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
                        - 2 * delta_n_O2_atm * const.avogadro
                    )
                    sol["O"]["gas"]["number_moles"] = np.array(
                        [[results["N_O_atm"] / const.avogadro]]
                    )
                    results["N_O_int"] = (
                        sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
                        + 2 * delta_n_O2_atm * const.avogadro
                    )
                    sol["O"]["silicate_melt"]["number_moles"] = np.array(
                        [[results["N_O_int"] / const.avogadro]]
                    )
                    mantle_iron_dict["mass_Fe2"] = np.max(
                        [n_Fe2 * const.M_Fe, 0]
                    )  # update remaining Fe2 mass
                    mantle_iron_dict["X_Fe2"] = (
                        mantle_iron_dict["mass_Fe2"] / mantle_iron_dict["mass_Fe"]
                    )
                else:
                    results["N_O_atm"] = sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
                    results["N_O_int"] = (
                        sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
                    )
        elif mantle_iron_dict["type"] == "static":
            if mantle_iron_dict["mass_Fe2"] == 0:
                results["N_O_atm"] = sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
                results["N_O_int"] = (
                    sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
                )
            else:
                n_Fe2 = (
                    mantle_iron_dict["mass_Fe2"] / const.M_Fe
                    - 4 * sol["O2_g"]["gas"]["number_moles"][0][0]
                )  # oxidize Fe2+ to Fe3+; this goes neg. when n_O2 > n_Fe2/4
                n_O2_atm = sol["O2_g"]["gas"]["number_moles"][0][0] - 0.25 * (
                    mantle_iron_dict["mass_Fe2"] / const.M_Fe - n_Fe2
                )  # goes to zero when when n_O2 >= n_Fe2/4 (won't go neg.)
                delta_n_O2_atm = 0.25 * (
                    mantle_iron_dict["mass_Fe2"] / const.M_Fe - n_Fe2
                )  # calculates exactly the n_O2 reacted away
                sol["O2_g"]["gas"]["number_moles"] = np.array(
                    [[np.max([n_O2_atm, 0])]]
                )  # goes to zero even when n_Fe2 goes neg., which should put O2 back in atm. Confirmed: doesn't matter b/c of line results['N_O_atm'] = ...
                results["N_O_atm"] = (
                    sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
                    - 2 * delta_n_O2_atm * const.avogadro
                )  # takes away only O2 reacted from total O_atm inventory
                sol["O"]["gas"]["number_moles"] = np.array([[results["N_O_atm"] / const.avogadro]])
                results["N_O_int"] = (
                    sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
                    + 2 * delta_n_O2_atm * const.avogadro
                )
                sol["O"]["silicate_melt"]["number_moles"] = np.array(
                    [[results["N_O_int"] / const.avogadro]]
                )
                mantle_iron_dict["mass_Fe2"] = np.max(
                    [n_Fe2 * const.M_Fe, 0]
                )  # update remaining Fe2 mass
    else:
        results["N_O_atm"] = sol["O"]["gas"]["number_moles"][0][0] * const.avogadro
        results["N_O_int"] = sol["O"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_C_atm"] = sol["C"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_C_int"] = sol["C"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_N_atm"] = sol["N"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_N_int"] = sol["N"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["N_S_atm"] = sol["S"]["gas"]["number_moles"][0][0] * const.avogadro
    results["N_S_int"] = sol["S"]["silicate_melt"]["number_moles"][0][0] * const.avogadro
    results["M_atm"] = sol["gas"]["phase"]["mass"][0][0]
    results["T_surface"] = T_surface
    results["T_surface_atmod"] = surface_temperature

    return results, sol, mantle_iron_dict, solution
