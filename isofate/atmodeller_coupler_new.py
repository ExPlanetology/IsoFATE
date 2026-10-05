# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmodeller coupler for IsoFATE."""

import diffrax
import equinox as eqx
import jax.numpy as jnp
from atmodeller import ChemicalSpecies, EquilibriumModel, ReservoirSpecies
from atmodeller import Planet as AtmodellerPlanet
from atmodeller.constants import SILICATE_MELT_PHASE_INDEX
from atmodeller.solubility import get_solubility_models
from jax.typing import ArrayLike
from jaxtyping import Array

from isofate.constants import const
from isofate.mantle_iron import MantleIronState
from isofate.parameters import Parameters
from isofate.species import SYMBOLS
from isofate.system import Planet
from isofate.utils import gravitational_acceleration

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


class AtmosphereModel(eqx.Module):
    """Atmosphere model

    Args:
        planet: Planet
        gamma: Adiabatic index. Defaults to ``7/5`` (diatomic ideal gas).
        pem: Pressure at the emission level/top of atmosphere (Pa). Defaults to ``0.2e5`` Pa
            (0.2 bar).
    """

    planet: Planet
    gamma: ArrayLike = 7 / 5
    pem: float = 0.2e5

    def _atmosphere_descent_vector_field(
        self,
        r: ArrayLike,
        y: tuple[ArrayLike, ArrayLike],
        args: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    ) -> tuple[ArrayLike, Array]:
        """RHS of the atmosphere-descent ODE system.

        Integrated downward in `r` (diffrax's independent/"time" variable here) from the planetary
        radius `rplanet` to the rocky-component radius `r_rocky`, hydrostatically building up the
        atmospheric column on a dry adiabat.

        Args:
            r: Radial coordinate (m) - the integration variable, decreasing from `rplanet` to
                `r_rocky`.
            y: `(p, matm)` - pressure at `r` (Pa) and the atmospheric mass (kg) accumulated so far,
                integrated from the top of the atmosphere down to `r`.
            args: `(Tem, mu, Mc, K)` - emission temperature (K), mean molecular mass (kg),
                planet core mass (kg), and adiabatic exponent `(gamma-1)/gamma` (dimensionless).

        Returns:
            `(dp_dr, dmatm_dr)`: the hydrostatic pressure gradient and the rate of atmospheric mass
            accumulation, both with respect to `r`.
        """
        p, _ = y
        Tem, mu, Mc, K = args

        # dry adiabat: T is algebraic in p, not itself integrated
        T: ArrayLike = Tem * (p / self.pem) ** K

        # Below is intentionally kept from the original code (but was always commented out), to
        # presumably test a deep isothermal layer.  It requires rewriting since it is not JAX
        # compliant.
        #
        # include a deep isothermal layer?
        # if p[i-1]>100*1e5:
        #    T[i-1]=T[i]
        # else:
        #    T[i-1]=T[i]/(p[i]/p[i-1])**K

        rho: ArrayLike = p * mu / (const.kb * T)
        g: ArrayLike = gravitational_acceleration(Mc, r)

        dp_dr: ArrayLike = -1 * g * rho
        dmatm_dr: Array = -4 * jnp.pi * jnp.square(r) * rho

        return dp_dr, dmatm_dr

    def make_atmosphere_descent(
        self,
        Tem: ArrayLike,
        mu: ArrayLike,
        rplanet: ArrayLike,
        Mc: ArrayLike,
        r_rocky: ArrayLike,
    ) -> tuple[ArrayLike, Array, ArrayLike]:
        """Atmosphere-descent integration: returns `M_atm`, `T_surf`, `P_surf`.

        The original NumPy implementation's three per-step array updates reduce to a 2-state ODE in
        `r`: `T` is not actually integrated - it's read off the dry adiabat algebraically from `p`
        (`T = Tem*(p/pem)**K`), and `rho` is a pure function of `p` too, so the only genuine ODE
        states are `p(r)` and the accumulated `M_atm(r)`.

        This integrates with `diffrax.Tsit5()` (5th-order explicit Runge-Kutta) and adaptive step
        sizing (`PIDController`).

        All three outputs are always computed and returned together. Callers that only need a subset
        should just discard the rest (e.g. `_, T_surf, P_surf = make_atmosphere_descent_jax(...)`).

        Args:
            Tem: Emission temperature (K)
            mu: Atomic mass (kg)
            rplanet: Planetary radius (m)
            Mc: Planet core mass (kg)
            r_rocky: Planet rocky-component radius (m) - the integration's lower bound in `r`.

        Returns:
            `(M_atm, T_surf, P_surf)`
        """
        K: ArrayLike = (self.gamma - 1) / self.gamma

        gem: ArrayLike = gravitational_acceleration(Mc, rplanet)
        matm0: ArrayLike = 4 * jnp.pi * rplanet**2 * self.pem / gem

        term = diffrax.ODETerm(self._atmosphere_descent_vector_field)
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=rplanet,  # pyright: ignore[reportArgumentType]
            t1=r_rocky,  # pyright: ignore[reportArgumentType]
            dt0=None,  # let the PIDController choose an initial step adaptively
            y0=(self.pem, matm0),
            args=(Tem, mu, Mc, K),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
            saveat=diffrax.SaveAt(t1=True),
        )
        P_surf: ArrayLike = sol.ys[0][-1]  # pyright: ignore[reportOptionalSubscript]
        M_atm: ArrayLike = sol.ys[1][-1]  # pyright: ignore[reportOptionalSubscript]
        T_surf: Array = jnp.power((P_surf / self.pem), K) * Tem

        return M_atm, T_surf, P_surf


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
