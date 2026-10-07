# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main IsoFATE script for coupled model."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from isofate.atmodeller_coupler import (
    AtmodellerCoupler,
    TrackedGasSpecies,
    aggregate_D_into_H,
    build_atmodeller,
    extract_full_output,
    get_tracked_gas_species,
    nan_full_output,
    run_atmodeller_step,
)
from isofate.atmodeller_coupler_new import AtmodellerCoupler as EventAtmodellerCoupler
from isofate.atmodeller_coupler_new import CouplerState
from isofate.constants import const
from isofate.engine import (
    bondi_radius,
    escape_radius,
    tidal_gravitational_potential,
)
from isofate.escape.fractionation import Phi_1_2, Phi_minor_species
from isofate.escape.mechanisms import EscapeState
from isofate.integrators import (
    EXHAUSTION_FRACTION,
    IntegrationStop,
    IsocalcIntegrator,
    integrate_segments,
    integrate_segments_euler,
)
from isofate.isofunks import R_atm, R_env
from isofate.mantle_iron import MantleIronState
from isofate.parameters import Parameters
from isofate.species import DEFAULT_SPECIES, SYMBOLS
from isofate.system import Planet
from isofate.utils import gravitational_acceleration


def isocalc(
    parameters: Parameters,
    t_end=5e9,
    isofate_species_abund: ArrayLike = (0, 0, 0, 0, 0, 0, 0),
):
    """
    This is a test

    Description

    Args:
        a (array): a test array
        b (array): a test array

    Returns:
        array: a test array

    Returns:
        array: a test array
    """

    # '''
    # Computes species abundances in ternary mixture of H, D, and He
    # via time-integrated numerical simulation of atmospheric escape.
    # Note: species 1 is H, species 2 is He, species 3 is D

    # Inputs:
    #  - f_atm: atmospheric mass fraction(s), must be in the form of an array [ndim]
    #  - Mp: planet mass [kg]
    #  - Mstar: stellar mass [kg]
    #  - F0: initial incident XUV flux [W/m2]
    #  - Fp: incident bolometric flux [W/m2]
    #  - T: planet equilibrium temperature [K]
    #  - d: orbtial distance [m]
    #  - t_end: simulation end time, i.e. system age at the end of the run; scalar [yr]
    #  - isofate_species_abund: initial abundance [atoms] for each tracked species, ordered as in
    #  isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
    #  - options: mode switches and tuning constants unrelated to escape mechanism, fixed for
    #  the whole run - see IsocalcOptions for the full list (rad_evol, melt_fraction_override,
    #  mu, n_steps, t_start, thermal, n_atmodeller, save_molecules, mantle_iron, dynamic_phi)
    #  - escape: escape-mechanism instance (isofate.escape.mechanisms.EscapeMechanism) controlling the
    #  atmospheric mass-flux calculation each timestep; defaults to XUVEscape(), equivalent to
    #  today's default mechanism="XUV", RR=True. See isofate.escape.mechanisms for XUVEscape, CPMLEscape,
    #  PhiKillEscape, CombinedEscape.

    # Output: Dictionary of 2-D arrays [len(f_atm) x n_steps] with keys,
    #  - 'time': simulation time array [s]
    #  - 'Rp': total planet radius [m]
    #  - 'Ratm': convective atm depth [m]
    #  - 'Matm': atmospheric mass [kg]
    #  - 'Vpot': gravitational potential at outer layer [J/kg]
    #  - 'fatm': total atmospheric mass fraction [ndim]
    #  - 'Mloss': atm mass loss per time step [kg]
    #  - 'phi': atm mass flux [kg/m2/s]
    #  - 'phic': critical mass flux for species 2 escape [kg/m2/s]
    #  - 'N_H': H number [atoms]
    #  - 'N_He': He number [atoms]
    #  - 'N_D': D number [atoms]
    #  - 'x1': H molar concentration [ndim]
    #  - 'x2': He molar concentration [ndim]
    #  - 'Phi_H': H number flux [atoms/s/m2]
    #  - 'Phi_He': He number flux [atoms/s/m2]
    #  - 'Phi_D': D number flux [atoms/s/m2]
    # '''

    # `parameters` bundles the config that's fixed for the whole run (System, escape mechanism,
    # EscapeNumberFlux, IsocalcOptions) - unpacked once here so the rest of this function (largely
    # unchanged from when these were separate arguments) can keep referring to them by these same
    # local names, matching isocalc_jax. Every IsocalcOptions field is read-only for the whole run
    # and referenced directly as `options.<field>` below. `mu`/R_B are derived fresh from the
    # evolving y every iteration below (no bootstrap `IsocalcOptions.mu` field exists anymore),
    # matching `IsocalcIntegrator.algebraic` (see engine.py). (`options.mantle_iron` similarly
    # seeds a local `mantle_iron_state` below, once `interior_atmosphere` is available.)
    system = parameters.system
    options = parameters.isocalc_options
    escape = parameters.escape_mechanism
    escape_number_flux = parameters.escape_number_flux

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, N_He, N_D, N_O, N_C, N_N, N_S = isofate_species_abund

    # T and Fp are confirmed fixed for the whole run (never reassigned anywhere below), so
    # they're read from `system` once, here. `system.star.mass` and the orbital distance are not
    # cached separately - `tidal_reduction_factor(system, Rp)` and `EscapeState(system=...)` read
    # them directly (see below).
    planet: Planet = system.planet
    T: ArrayLike = system.equilibrium_temperature
    Fp: ArrayLike = system.insolation

    # Only planet.mass is ever read here (Mp never changes over the run) - planet.f_atm is not
    # read at all: M_atm/f_atm are derived fresh from y every iteration below (see the loop body),
    # not seeded from it, so the initial atmosphere mass is exactly
    # dot(isofate_species_abund, atomic_masses), matching isocalc_jax.
    Mp = planet.mass

    ###_____Initialize physical values_____###
    R_H = system.hill_radius  # Hill radius [m]

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps  # timesteps
    t0_seconds = options.t_start / const.s2yr  # simulation start time [s]
    t = t_end / const.s2yr - t0_seconds  # simulation duration [s]
    delta_t = t / n_tot  # timestep [s]

    ###_____Set initial values____###

    atomic_masses = DEFAULT_SPECIES.atomic_masses
    species_names = SYMBOLS

    ### atmodeller interior
    # Ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as
    # isofate_species_abund above.
    isofate_species_abund_int = np.zeros(7)
    if options.n_atmodeller == 0:
        T_surf_analytic = 0
        T_surf_atmod = 0

    # build atmodeller model for interior-atmosphere coupling
    interior_atmosphere = build_atmodeller(Mp, surface_radius=planet.rocky_radius)
    # Ordered per interior_atmosphere's own gas-phase species (SpeciesCollection), not hand-typed
    # - see the molecule-array declarations/writes below, which are driven by this tuple so they
    # can never drift out of sync with atmodeller's actual species set/order.
    tracked_species: tuple[TrackedGasSpecies, ...] = get_tracked_gas_species(interior_atmosphere)
    # Warm-starts each AtmodellerCoupler call from the previous call's converged solution instead
    # of solving cold every time - consecutive calls are a tiny physical perturbation apart, so
    # this drastically cuts the number of Newton iterations needed. None on the first call.
    atmod_initial_guess = None

    atmod_full_output = {}  # dictionary to store atmodeller full output

    # background_mantle_mass reads atmodeller's own mantle-mass partitioning (from the
    # core_mass_fraction build_atmodeller passed to Planet.from_species) directly, rather than
    # re-deriving/duplicating it here.
    mantle_iron_state: MantleIronState | None = None
    if options.mantle_iron is not None:
        mantle_mass = float(interior_atmosphere.parameters.state.background_mantle_mass)
        mantle_iron_state = MantleIronState.initial(options.mantle_iron, mantle_mass)

    ### atmosphere
    # Atmospheric number of atoms per species [atoms], ordered per
    # isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as isofate_species_abund above.
    # M_atm/f_atm are not seeded here - they're derived fresh from y at the top of every loop
    # iteration below (see the loop body).
    y = np.array(isofate_species_abund, dtype=float)
    ###_____Initialize arrays_____###

    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a = t0_seconds + delta_t * np.arange(1, n_tot + 1)

    phi_a = np.zeros(n_tot)  # mass flux array [kg/s/m2]
    phic_a = np.zeros(n_tot)  # critical mass flux array [kg/s/m2]
    Rp_a = np.zeros(n_tot)  # total radius, diagnostic [m]
    Renv_a = np.zeros(n_tot)  # envelope radius [m]
    Matm_a = np.zeros(n_tot)  # atmospheric mass [kg]
    fatm_a = np.zeros(n_tot)  # atm mass fraction [ndim]
    Vpot_a = np.zeros(n_tot)  # grav potential, diagnostic [J/kg]
    Mloss_a = np.zeros(n_tot)  # mass lost per timestep [kg]

    # Atmospheric/mantle number-of-atoms history, ordered per isofate.species.SYMBOLS (H, He, D,
    # O, C, N, S) - same order as y/isofate_species_abund_int - one column per species, one row per
    # timestep.
    y_a = np.zeros((n_tot, 7))  # atmospheric number array [atoms]
    y_a_int = np.zeros((n_tot, 7))  # mantle number array [atoms]
    # Atmospheric/mantle molecule number history, ordered per interior_atmosphere's own gas-phase
    # SpeciesCollection (see tracked_species above), keyed by isofate's human-readable label
    # (e.g. "SO2", not atmodeller's canonical "O2S_g").
    gas_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    melt_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    fO2_a = np.zeros(n_tot)  # fugacity array [bar]
    # Molar concentration history [ndim], ordered per isofate.species.SYMBOLS (H, He, D, O, C, N,
    # S) - same order as y/x - one column per species, one row per timestep.
    x_a = np.zeros((n_tot, 7))
    # Number flux history [atoms/s/m2], ordered per isofate.species.SYMBOLS (H, He, D, O, C, N,
    # S) - same convention as y_a/x_a above.
    Phi_a = np.zeros((n_tot, 7))
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot)  # atmodeller surface temperature array (capped at 6000 K) [K]

    def geometry(y: np.ndarray, age: float) -> tuple[float, float, float, float]:
        """Mean particle mass, envelope thickness, escape radius and atmosphere mass fraction of
        the atmosphere `y` at the given age [s]."""
        f_atm = parameters.atmosphere_mass_fraction(y)
        mu = parameters.atmosphere_mean_mu(y)
        if options.rad_evol == False:
            return mu, 0, planet.rocky_radius, f_atm
        R_B = bondi_radius(parameters, y)  # recomputed from the current mu, not a fixed bootstrap
        radius_env = R_env(Mp, f_atm, Fp, age, options.thermal)
        radius_atm = R_atm(T, Mp, planet.rocky_radius, radius_env, mu)
        radius_p = planet.rocky_radius + radius_atm + radius_env
        # limits Rp to the min of Bondi/Hill/Lopez+Fortney radius; plain min() avoids numpy's
        # array-construction/dispatch overhead on a 3-scalar comparison run every timestep
        return mu, radius_env, min(R_B, R_H, radius_p), f_atm

    def fill_exhausted(n: int) -> None:
        """Fills the outputs from row `n` onwards once the entire atmosphere is lost."""
        Matm_a[n:] = 0
        fatm_a[n:] = 0
        Renv_a[n:] = 0
        Rp_a[n:] = planet.rocky_radius
        Vpot_a[n:] = tidal_gravitational_potential(system, planet.rocky_radius)
        phi_a[n:] = 0
        Mloss_a[n:] = 0
        y_a[n:] = 0
        y_a_int[n:] = 0
        for label in gas_num_a:
            gas_num_a[label][n:] = 0
            melt_num_a[label][n:] = 0
        x_a[n:] = x_a[n - 1]  # carry the last molar concentration forward
        Phi_a[n:] = 0
        fO2_a[n:] = 0

    def is_exhausted(y: np.ndarray) -> bool:
        return np.dot(y, atomic_masses) <= 0 or np.sum(y) <= 0

    ###_____Loop through the states_____###

    # Schedule (shared with isocalc_jax3): state n is the state after n steps, at time
    # t0 + n*delta_t; output row j holds state j + 1 (at t_a[j]), so the last row is the state at
    # t_end. Atmodeller equilibrates state 0 before the loop, then re-equilibrates every
    # n_atmodeller steps, i.e. at states n_atmodeller, 2*n_atmodeller, ..., including the final
    # state at t_end when n_steps is a multiple of n_atmodeller; a row at an equilibration holds
    # the re-equilibrated state. Each row's diagnostics (radius, fluxes, ...) are evaluated from
    # the state it holds, and are the ones used for the following Euler step.

    # Molecule numbers and O2 activity from the latest equilibration, recorded in every row
    gas_num: dict[str, float] = {sp.label: 0.0 for sp in tracked_species}
    melt_num: dict[str, float] = {sp.label: 0.0 for sp in tracked_species}
    fO2: float = 0.0

    def equilibrate(t_now: float) -> None:
        """Re-equilibrates the atmosphere `y` with the interior using Atmodeller at time `t_now`
        [s], updating the loop state in place.

        This runs *before* a state's escape fluxes are computed. Previously the fluxes were
        computed from the pre-equilibration atmosphere and then subtracted from the
        post-equilibration one: the pre-equilibration atmosphere still holds the inventory that
        Atmodeller then dissolves (e.g. N, S), so the fluxes were far too large for what was left
        in the gas and, with clipping at zero, emptied it in a single step - removing atoms that
        never escaped, at every equilibration. Atmodeller's inputs (escape radius, mean particle
        mass) come from the pre-equilibration atmosphere, as in isocalc_jax3's
        AtmodellerCoupler.reequilibrate.
        """
        nonlocal y, isofate_species_abund_int, T_surf_analytic, T_surf_atmod
        nonlocal mantle_iron_state, atmod_initial_guess, fO2
        mu, _, radius_p, _ = geometry(y, t_now)
        # D/H fraction of the whole inventory (atmosphere + interior) at this solve, assuming the
        # same D/H in both - computed from the current state, as in isocalc_jax3, so the H/D split
        # after the solve conserves D exactly (zero if there is no hydrogen)
        total_HD = y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
        X_DH = (y[2] + isofate_species_abund_int[2]) / total_HD if total_HD != 0 else 0.0
        # Species-level diagnostics (step.atmod_full["H2_g"]["gas"][...], O2 activity, etc.) are
        # only read below when save_molecules is True; otherwise the narrow extraction (element
        # number_moles + gas mass only) is all this loop needs.
        step = run_atmodeller_step(
            T,
            radius_p,
            mu,
            options.melt_fraction_override,
            mantle_iron_state,
            y,
            isofate_species_abund_int,
            X_DH,
            interior_atmosphere,
            atmod_initial_guess,
            full_output=options.save_molecules,
        )
        y = step.y
        isofate_species_abund_int = step.isofate_species_abund_int
        T_surf_analytic = step.T_surf_analytic
        T_surf_atmod = step.T_surf_atmod
        mantle_iron_state = step.mantle_iron_state
        atmod_initial_guess = step.atmod_initial_guess
        if options.save_molecules == True:
            for sp in tracked_species:
                gas_num[sp.label] = step.atmod_full[sp.gas_name]["gas"]["number_moles"][0][0]
                melt_num[sp.label] = (
                    step.atmod_full[sp.melt_name]["silicate_melt"]["number_moles"][0][0]
                    if sp.melt_name is not None
                    else 0.0  # no solubility model / melt reservoir
                )
            fO2 = step.atmod_full["O2_g"]["gas"]["activity"][0][0]

    # Initial equilibration of the given inventory, before the time integration starts. Like
    # isocalc_jax3, the initial state is not recorded, so the outputs start from the equilibrated
    # atmosphere (minus one step of escape) rather than showing the initial drop as it dissolves.
    if options.n_atmodeller != 0 and not is_exhausted(y):
        equilibrate(t0_seconds)

    for n in range(n_tot + 1):
        # Physical time of state n. Used as the age in all age-dependent physics (envelope
        # contraction in R_env, the XUV flux history). Previously the physics was evaluated at the
        # old output label t_a[n] (= t0 + (n+1)*delta_t*n_tot/(n_tot-1)), about one step ahead,
        # which shifted the escape radius, and so the surface temperature and Atmodeller's
        # gas/melt partition, at every equilibration.
        t_now = t0_seconds + n * delta_t
        row = n - 1  # output row holding state n (state 0, the initial state, is not recorded)

        ### Stop simulation when entire atmosphere is lost
        if is_exhausted(y):
            fill_exhausted(max(row, 0))
            if options.n_atmodeller != 0:
                atmod_full_output = nan_full_output()  # atmodeller full output for monte carlo runs
            break

        ##### run atmodeller ######
        # Scheduled re-equilibration (state 0 was equilibrated before the loop), *before* this
        # state's escape fluxes are computed - see `equilibrate`
        if options.n_atmodeller != 0 and n > 0 and n % options.n_atmodeller == 0:
            equilibrate(t_now)
            # The equilibration may have dissolved the entire atmosphere
            if is_exhausted(y):
                fill_exhausted(row)
                atmod_full_output = nan_full_output()
                break

        # Derived fresh from the (possibly re-equilibrated) y every iteration (mass-conservation
        # identity: the atmosphere's total mass is exactly the sum of its constituent atoms'
        # masses), rather than tracked as separately-updated state - matches isocalc_jax's design
        # (see integrators.py's IsocalcIntegrator.algebraic).
        M_atm = np.dot(y, atomic_masses)
        N_tot = np.sum(y)
        mu, radius_env, radius_p, f_atm = geometry(y, t_now)

        Vpot = tidal_gravitational_potential(system, radius_p)
        A = 4 * np.pi * radius_p**2

        # sets mass flux [kg/m2/s]
        state = EscapeState(
            system=system,
            escape_radius=radius_p,
            gravitational_potential=Vpot,
            mean_mu=mu,
            convective_envelope_thickness=radius_env,
            atmosphere_mass_fraction=f_atm,
            t_current=t_now,
        )
        phi = escape.compute_mass_flux(state)

        mass_loss = phi * A * delta_t
        g = gravitational_acceleration(Mp, radius_p)

        x = y / N_tot  # molar concentration per species [ndim]

        if options.dynamic_phi == False:
            Phi, phi_c = escape_number_flux.get_number_flux(y, T, g, phi)

        elif options.dynamic_phi == True:
            N_values = y  # ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
            abundances_with_idx = [(i, N_values[i]) for i in range(7)]
            abundances_with_idx.sort(key=lambda item: item[1], reverse=True)

            most_abundant_idx = abundances_with_idx[0][0]
            second_most_abundant_idx = abundances_with_idx[1][0]

            # Get masses and scale heights for the two most abundant species
            scale_heights = DEFAULT_SPECIES.scale_heights(T, g)

            mass_most = atomic_masses[most_abundant_idx]
            mass_second = atomic_masses[second_most_abundant_idx]
            H_most = scale_heights[most_abundant_idx]
            H_second = scale_heights[second_most_abundant_idx]

            # Determine which is lighter (species 1) and heavier (species 2) by MASS
            if mass_most <= mass_second:
                light_dominant_idx = most_abundant_idx  # species 1 (lighter)
                heavy_dominant_idx = second_most_abundant_idx  # species 2 (heavier)
                mass_1 = mass_most
                mass_2 = mass_second
                H_1 = H_most
                H_2 = H_second
            else:
                light_dominant_idx = second_most_abundant_idx  # species 1 (lighter)
                heavy_dominant_idx = most_abundant_idx  # species 2 (heavier)
                mass_1 = mass_second
                mass_2 = mass_most
                H_1 = H_second
                H_2 = H_most

            # Calculate molar fractions for the two dominant species
            N1 = N_values[light_dominant_idx]  # lightest dominant (species 1)
            N2 = N_values[heavy_dominant_idx]  # heaviest dominant (species 2)
            N_tot_binary = N1 + N2

            X1 = N1 / N_tot_binary  # molar fraction of species 1 in binary mixture
            X2 = N2 / N_tot_binary  # molar fraction of species 2 in binary mixture
            MU = X1 * mass_1 + X2 * mass_2

            # Calculate binary diffusion coefficient between the two dominant species
            light_name = species_names[light_dominant_idx]  # species 1
            heavy_name = species_names[heavy_dominant_idx]  # species 2

            b = escape_number_flux.binary_diffusion.get(light_name, heavy_name, T)

            # Calculate escape fluxes for the two dominant species
            Phi_1_calc, Phi_2_calc, phi_c = Phi_1_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)

            # Assign fluxes to correct species based on light/heavy dominant indices - ordered
            # per isofate.species.SYMBOLS (H, He, D, O, C, N, S)
            Phi = np.zeros(7)
            Phi[light_dominant_idx] = Phi_1_calc
            Phi[heavy_dominant_idx] = Phi_2_calc

            # Calculate fluxes for remaining species using corrected generalized function
            for i in range(7):
                if i != light_dominant_idx and i != heavy_dominant_idx:
                    Phi[i] = Phi_minor_species(
                        Phi_1_calc,
                        Phi_2_calc,
                        H_1,
                        H_2,
                        scale_heights[i],
                        N_values,
                        T,
                        i,
                        light_dominant_idx,
                        heavy_dominant_idx,
                    )

        # record state n in its output row
        if row >= 0:
            Matm_a[row] = M_atm
            fatm_a[row] = f_atm
            # this will still change even with Rp limited to min(R_Bondi, R_Hill)
            Renv_a[row] = radius_env
            Rp_a[row] = radius_p
            Vpot_a[row] = Vpot
            phi_a[row] = phi
            phic_a[row] = phi_c
            Mloss_a[row] = mass_loss

            y_a[row] = y
            y_a_int[row] = isofate_species_abund_int
            x_a[row] = x
            Phi_a[row] = Phi
            T_surf_analytic_a[row] = T_surf_analytic
            T_surf_atmod_a[row] = T_surf_atmod
            for label in gas_num_a:
                gas_num_a[label][row] = gas_num[label]
                melt_num_a[label][row] = melt_num[label]
            fO2_a[row] = fO2

        if n == n_tot:
            # save final molecular abundances (read-only diagnostic of the final state)
            if options.n_atmodeller != 0:
                atmod_sol = AtmodellerCoupler(
                    T,
                    radius_p,
                    mu,
                    options.melt_fraction_override,
                    mantle_iron_state,
                    *aggregate_D_into_H(y),
                    *aggregate_D_into_H(isofate_species_abund_int),
                    interior_atmosphere,
                    initial_guess=atmod_initial_guess,
                )[1]
                atmod_full_output = extract_full_output(atmod_sol)
            break

        # advance to the next state (forward Euler, clipped at zero)
        y_loss = Phi * A * delta_t
        y -= y_loss
        y = np.maximum(y, 0)

    # save results
    solutions = {
        "time": t_a,
        "Rp": Rp_a,
        "Ratm": Renv_a,
        "Matm": Matm_a,
        "Vpot": Vpot_a,
        "fatm": fatm_a,
        "Mloss": Mloss_a,
        "phi": phi_a,
        "phic": phic_a,
        # y_a/y_a_int columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S);
        # .copy() keeps each output array independent, matching the pre-array-refactor behavior.
        "N_H": y_a[:, 0].copy(),
        "N_He": y_a[:, 1].copy(),
        "N_D": y_a[:, 2].copy(),
        "N_O": y_a[:, 3].copy(),
        "N_C": y_a[:, 4].copy(),
        "N_N": y_a[:, 5].copy(),
        "N_S": y_a[:, 6].copy(),
        "N_H_int": y_a_int[:, 0].copy(),
        "N_He_int": y_a_int[:, 1].copy(),
        "N_D_int": y_a_int[:, 2].copy(),
        "N_O_int": y_a_int[:, 3].copy(),
        "N_C_int": y_a_int[:, 4].copy(),
        "N_N_int": y_a_int[:, 5].copy(),
        "N_S_int": y_a_int[:, 6].copy(),
        "x1": x_a[:, 0].copy(),
        "x2": x_a[:, 1].copy(),
        "x3": x_a[:, 2].copy(),
        "x4": x_a[:, 3].copy(),
        "x5": x_a[:, 4].copy(),
        "x6": x_a[:, 5].copy(),
        "x7": x_a[:, 6].copy(),
        # Phi_a columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S); .copy()
        # keeps each output array independent, matching the pre-array-refactor behavior.
        "Phi_H": Phi_a[:, 0].copy(),
        "Phi_He": Phi_a[:, 1].copy(),
        "Phi_D": Phi_a[:, 2].copy(),
        "Phi_O": Phi_a[:, 3].copy(),
        "Phi_C": Phi_a[:, 4].copy(),
        "Phi_N": Phi_a[:, 5].copy(),
        "Phi_S": Phi_a[:, 6].copy(),
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
    }
    if options.save_molecules == True:
        for label, arr in gas_num_a.items():
            solutions[f"n_{label}_a"] = arr
        for label, arr in melt_num_a.items():
            solutions[f"n_{label}_a_int"] = arr
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions


def isocalc_jax(
    parameters: Parameters,
    t_end=5e9,
    isofate_species_abund: Array = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
):
    """
    This is a test

    Description

    Args:
        a (array): a test array
        b (array): a test array

    Returns:
        array: a test array

    Returns:
        array: a test array
    """

    # '''
    # Computes species abundances in ternary mixture of H, D, and He
    # via time-integrated numerical simulation of atmospheric escape.
    # Note: species 1 is H, species 2 is He, species 3 is D

    # Inputs:
    #  - f_atm: atmospheric mass fraction(s), must be in the form of an array [ndim]
    #  - Mp: planet mass [kg]
    #  - Mstar: stellar mass [kg]
    #  - F0: initial incident XUV flux [W/m2]
    #  - Fp: incident bolometric flux [W/m2]
    #  - T: planet equilibrium temperature [K]
    #  - d: orbtial distance [m]
    #  - t_end: simulation end time, i.e. system age at the end of the run; scalar [yr]
    #  - isofate_species_abund: initial abundance [atoms] for each tracked species, ordered as in
    #  isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
    #  - options: mode switches and tuning constants unrelated to escape mechanism, fixed for
    #  the whole run - see IsocalcOptions for the full list (rad_evol, melt_fraction_override,
    #  mu, n_steps, t_start, thermal, n_atmodeller, save_molecules, mantle_iron, dynamic_phi)
    #  - escape: escape-mechanism instance (isofate.escape.mechanisms.EscapeMechanism) controlling the
    #  atmospheric mass-flux calculation each timestep; defaults to XUVEscape(), equivalent to
    #  today's default mechanism="XUV", RR=True. See isofate.escape.mechanisms for XUVEscape, CPMLEscape,
    #  PhiKillEscape, CombinedEscape.

    # Output: Dictionary of 2-D arrays [len(f_atm) x n_steps] with keys,
    #  - 'time': simulation time array [s]
    #  - 'Rp': total planet radius [m]
    #  - 'Ratm': convective atm depth [m]
    #  - 'Matm': atmospheric mass [kg]
    #  - 'Vpot': gravitational potential at outer layer [J/kg]
    #  - 'fatm': total atmospheric mass fraction [ndim]
    #  - 'Mloss': atm mass loss per time step [kg]
    #  - 'phi': atm mass flux [kg/m2/s]
    #  - 'phic': critical mass flux for species 2 escape [kg/m2/s]
    #  - 'N_H': H number [atoms]
    #  - 'N_He': He number [atoms]
    #  - 'N_D': D number [atoms]
    #  - 'x1': H molar concentration [ndim]
    #  - 'x2': He molar concentration [ndim]
    #  - 'Phi_H': H number flux [atoms/s/m2]
    #  - 'Phi_He': He number flux [atoms/s/m2]
    #  - 'Phi_D': D number flux [atoms/s/m2]
    # '''

    # `system`/`options` are unpacked here since they're also read directly below (`system.planet`,
    # `options.<field>`); `escape`/`escape_number_flux` are not - `parameters` itself is passed
    # straight through to `IsocalcIntegrator.integrate`, which pulls them off internally.
    # `planet.f_atm` is NOT read here (unlike isocalc): `IsocalcIntegrator.integrate` derives
    # mu/M_atm/f_atm/R_B purely from the evolving y - see that function's docstring for why. Every
    # other IsocalcOptions field is read-only for the whole run and referenced directly as
    # `options.<field>` below. (`options.mantle_iron` similarly seeds a local `mantle_iron_state`
    # below, once `interior_atmosphere` is available.)
    system = parameters.system
    options = parameters.isocalc_options

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, _, N_D, _, _, _, _ = isofate_species_abund

    # Only planet.mass is read here, to seed build_atmodeller below (Mp never changes over the
    # run). `IsocalcIntegrator.integrate` re-derives T/d/Fp/Mp from `system` itself (via its
    # `parameters` field), so they don't need to be threaded through separately.
    # `system.star.mass` is no longer cached separately - `tidal_reduction_factor(system, Rp)`
    # reads it directly (see below).
    planet: Planet = system.planet
    Mp = planet.mass

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps  # timesteps
    t_start_seconds = options.t_start / const.s2yr  # simulation start time [s]
    t_end_seconds = t_end / const.s2yr  # simulation end time [s]
    duration = t_end_seconds - t_start_seconds  # simulation duration [s]
    delta_t = duration / n_tot  # timestep [s]

    ###_____Set initial values____###

    ### atmodeller interior
    # Ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as
    # isofate_species_abund above.
    isofate_species_abund_int = np.zeros(7)
    if options.n_atmodeller == 0:
        T_surf_analytic = 0
        T_surf_atmod = 0
    if N_H != 0 and N_D != 0:  # needed to allow D and H to outgas from mantle
        X_DH = N_D / (N_H + N_D)  # ignores D in mantle

    # build atmodeller model for interior-atmosphere coupling
    interior_atmosphere = build_atmodeller(Mp, surface_radius=planet.rocky_radius)
    # Ordered per interior_atmosphere's own gas-phase species (SpeciesCollection), not hand-typed
    # - see the molecule-array declarations/writes below, which are driven by this tuple so they
    # can never drift out of sync with atmodeller's actual species set/order.
    tracked_species: tuple[TrackedGasSpecies, ...] = get_tracked_gas_species(interior_atmosphere)
    # Warm-starts each AtmodellerCoupler call from the previous call's converged solution instead
    # of solving cold every time - consecutive calls are a tiny physical perturbation apart, so
    # this drastically cuts the number of Newton iterations needed. None on the first call.
    atmod_initial_guess = None

    atmod_full_output = {}  # dictionary to store atmodeller full output

    # TODO: Will add back eventually, once JAX refactor is working for the simpler case
    # background_mantle_mass reads atmodeller's own mantle-mass partitioning (from the
    # core_mass_fraction build_atmodeller passed to Planet.from_species) directly, rather than
    # re-deriving/duplicating it here.
    # mantle_iron_state: MantleIronState | None = None
    # if options.mantle_iron is not None:
    #     mantle_mass = float(interior_atmosphere.parameters.state.background_mantle_mass)
    #     mantle_iron_state = MantleIronState.initial(options.mantle_iron, mantle_mass)

    ### atmosphere
    # Atmospheric number of atoms per species [atoms], ordered per
    # isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as isofate_species_abund above.
    # The initial atmospheric mass is not seeded from planet.f_atm here (unlike isocalc) - see
    # IsocalcIntegrator's docstring for why - it's derived from this y instead.
    # y = np.array(isofate_species_abund, dtype=float)
    ###_____Initialize arrays_____###

    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a = t_start_seconds + delta_t * np.arange(1, n_tot + 1)

    y_a_int = np.zeros((n_tot, 7))  # mantle number array [atoms] - atmodeller disabled, stays 0
    # Atmospheric/mantle molecule number history, ordered per interior_atmosphere's own gas-phase
    # SpeciesCollection (see tracked_species above), keyed by isofate's human-readable label
    # (e.g. "SO2", not atmodeller's canonical "O2S_g"). Atmodeller is disabled in this simplified
    # case (see TODOs below), so these stay all-zero - no per-timestep population needed.
    gas_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    melt_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    fO2_a = np.zeros(n_tot)  # fugacity array [bar]
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot)  # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Integrate_____###

    integrator = IsocalcIntegrator(parameters)
    sol = integrator.integrate(
        jnp.asarray(t_start_seconds),
        jnp.asarray(t_a),
        jnp.asarray(isofate_species_abund),
    )
    y_a = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
    alg_a = integrator.diagnostics(jnp.asarray(t_a), y_a)
    Matm_a = alg_a.atmosphere_mass
    fatm_a = alg_a.atmosphere_mass_fraction
    Renv_a = alg_a.convective_envelope_thickness
    Rp_a = alg_a.escape_radius
    Vpot_a = alg_a.gravitational_potential
    phi_a = alg_a.mass_flux
    phic_a = alg_a.critical_mass_flux
    Phi_a = alg_a.number_flux
    x_a = alg_a.atom_fractions
    # Matches isocalc's per-output-step estimate (instantaneous rate * the output grid's fixed
    # delta_t) - not literally "mass lost between saved points" (which the adaptive trajectory
    # could give more precisely via diff(Matm_a), but that's a different definition; keeping this
    # one for now, matching the original quantity).
    Mloss_a = phi_a * alg_a.area * delta_t

    # TODO: For a later refactor, decide on the post-exhaustion convention: the rows after the
    # exhaustion event are currently inf (as diffrax leaves them, see IsocalcIntegrator.integrate),
    # as opposed to isocalc's original per-field zero-vs-hold choices (e.g. Rp_a held at
    # planet.rocky_radius).

    # TODO: Will add back eventually, once JAX refactor is working for the simpler case
    ##### run atmodeller ######
    # if options.n_atmodeller != 0:  # save final molecular abundances on last time step
    #     if n == options.n_steps - 1:
    #         atmod_sol = AtmodellerCoupler(
    #             T,
    #             escape_radius,
    #             mu,
    #             options.melt_fraction_override,
    #             mantle_iron_state,
    #             *aggregate_D_into_H(y),
    #             *aggregate_D_into_H(isofate_species_abund_int),
    #             interior_atmosphere,
    #             initial_guess=atmod_initial_guess,
    #         )[1]
    #         atmod_full_output = extract_full_output(atmod_sol)
    #     if n % options.n_atmodeller == 0:  # run atmodeller every n_atmodeller steps.
    #         # Species-level diagnostics (step.atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
    #         # are only read below when save_molecules is True; otherwise the narrow extraction
    #         # (element number_moles + gas mass only) is all this loop needs.
    #         step = run_atmodeller_step(
    #             T,
    #             escape_radius,
    #             mu,
    #             options.melt_fraction_override,
    #             mantle_iron_state,
    #             y,
    #             isofate_species_abund_int,
    #             X_DH,
    #             interior_atmosphere,
    #             atmod_initial_guess,
    #             full_output=options.save_molecules,
    #         )
    #         y = step.y
    #         isofate_species_abund_int = step.isofate_species_abund_int
    #         M_atm = step.M_atm
    #         T_surf_analytic = step.T_surf_analytic
    #         T_surf_atmod = step.T_surf_atmod
    #         mantle_iron_state = step.mantle_iron_state
    #         atmod_initial_guess = step.atmod_initial_guess
    #         if options.save_molecules == True:
    #             for sp in tracked_species:
    #                 gas_num_a[sp.label][n] = step.atmod_full[sp.gas_name]["gas"][
    #                     "number_moles"
    #                 ][0][0]
    #                 if sp.melt_name is not None:
    #                     melt_num_a[sp.label][n] = step.atmod_full[sp.melt_name][
    #                         "silicate_melt"
    #                     ]["number_moles"][0][0]
    #                 else:
    #                     melt_num_a[sp.label][n] = 0.0  # no solubility model / melt reservoir
    #             fO2_a[n] = step.atmod_full["O2_g"]["gas"]["activity"][0][0]
    #     else:
    #         for label in gas_num_a:
    #             gas_num_a[label][n] = gas_num_a[label][n - 1]
    #             melt_num_a[label][n] = melt_num_a[label][n - 1]
    #         fO2_a[n] = fO2_a[n - 1]

    # needed to allow D and H to outgas from mantle - X_DH stays inert while atmodeller is
    # disabled (isofate_species_abund_int never becomes nonzero), kept here for when atmodeller
    # is reintegrated.
    if y_a[-1, 0] + y_a[-1, 2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
        X_DH = (y_a[-1, 2] + isofate_species_abund_int[2]) / (
            y_a[-1, 0] + y_a[-1, 2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
        )  # assumes D/H is in equilibrium between interior and atmosphere

    # save results
    solutions = {
        "time": t_a,
        "Rp": Rp_a,
        "Ratm": Renv_a,
        "Matm": Matm_a,
        "Vpot": Vpot_a,
        "fatm": fatm_a,
        "Mloss": Mloss_a,
        "phi": phi_a,
        "phic": phic_a,
        # y_a/y_a_int columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S);
        # .copy() keeps each output array independent, matching the pre-array-refactor behavior.
        "N_H": y_a[:, 0].copy(),
        "N_He": y_a[:, 1].copy(),
        "N_D": y_a[:, 2].copy(),
        "N_O": y_a[:, 3].copy(),
        "N_C": y_a[:, 4].copy(),
        "N_N": y_a[:, 5].copy(),
        "N_S": y_a[:, 6].copy(),
        "N_H_int": y_a_int[:, 0].copy(),
        "N_He_int": y_a_int[:, 1].copy(),
        "N_D_int": y_a_int[:, 2].copy(),
        "N_O_int": y_a_int[:, 3].copy(),
        "N_C_int": y_a_int[:, 4].copy(),
        "N_N_int": y_a_int[:, 5].copy(),
        "N_S_int": y_a_int[:, 6].copy(),
        "x1": x_a[:, 0].copy(),
        "x2": x_a[:, 1].copy(),
        "x3": x_a[:, 2].copy(),
        "x4": x_a[:, 3].copy(),
        "x5": x_a[:, 4].copy(),
        "x6": x_a[:, 5].copy(),
        "x7": x_a[:, 6].copy(),
        # Phi_a columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S); .copy()
        # keeps each output array independent, matching the pre-array-refactor behavior.
        "Phi_H": Phi_a[:, 0].copy(),
        "Phi_He": Phi_a[:, 1].copy(),
        "Phi_D": Phi_a[:, 2].copy(),
        "Phi_O": Phi_a[:, 3].copy(),
        "Phi_C": Phi_a[:, 4].copy(),
        "Phi_N": Phi_a[:, 5].copy(),
        "Phi_S": Phi_a[:, 6].copy(),
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
    }
    if options.save_molecules == True:
        for label, arr in gas_num_a.items():
            solutions[f"n_{label}_a"] = arr
        for label, arr in melt_num_a.items():
            solutions[f"n_{label}_a_int"] = arr
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions


def isocalc_jax2(
    parameters: Parameters,
    t_end=5e9,
    isofate_species_abund: Array = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
):
    """
    This is a test

    Description

    Args:
        a (array): a test array
        b (array): a test array

    Returns:
        array: a test array

    Returns:
        array: a test array
    """

    # Duplicate of isocalc_jax (kept untouched as a reference for the no-Atmodeller path) with
    # Atmodeller coupling re-added: the integration is split into segments at Atmodeller-call
    # boundaries (spaced options.n_atmodeller output points apart, exactly like isocalc's discrete
    # cadence), each segment run as a normal adaptive IsocalcIntegrator.integrate call, with
    # Atmodeller called from Python between segments exactly as isocalc's discrete loop calls it.
    # See the "options.n_atmodeller != 0" branch below.
    #
    # Known, deliberate one-index difference from isocalc: isocalc records isofate_species_abund
    # at output index n *before* iteration n's own Atmodeller call is applied (the call's effect
    # only appears starting at index n+1; index 0 is always isocalc's untouched raw initial
    # condition) - an artifact of its discrete per-timestep recording order, invisible for smooth
    # quantities and only visible right at call boundaries. Here, each boundary's Atmodeller call
    # result is instead recorded starting exactly at that boundary's own output index, one index
    # earlier than isocalc's convention. This does not affect final-value comparisons (the last
    # index of both agree) - only per-index comparisons taken right at an Atmodeller-call boundary
    # would see it.

    # `system`/`options` are unpacked here since they're also read directly below (`system.planet`,
    # `options.<field>`); `escape`/`escape_number_flux` are not - `parameters` itself is passed
    # straight through to `IsocalcIntegrator.integrate`, which pulls them off internally.
    # `planet.f_atm` is NOT read here (unlike isocalc): `IsocalcIntegrator.integrate` derives
    # mu/M_atm/f_atm/R_B purely from the evolving y - see that function's docstring for why. Every
    # other IsocalcOptions field is read-only for the whole run and referenced directly as
    # `options.<field>` below. (`options.mantle_iron` similarly seeds a local `mantle_iron_state`
    # below, once `interior_atmosphere` is available.)
    system = parameters.system
    options = parameters.isocalc_options

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, _, N_D, _, _, _, _ = isofate_species_abund

    # Only planet.mass is read here, to seed build_atmodeller below (Mp never changes over the
    # run). `IsocalcIntegrator.integrate` re-derives T/d/Fp/Mp from `system` itself (via its
    # `parameters` field), so they don't need to be threaded through separately.
    # `system.star.mass` is no longer cached separately - `tidal_reduction_factor(system, Rp)`
    # reads it directly (see below).
    planet: Planet = system.planet
    Mp = planet.mass
    T = system.equilibrium_temperature  # fixed for the whole run

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps  # timesteps
    # np.asarray (not a bare Python float): the segment loop below passes this as `t0_chunk` on
    # its first iteration and `t_a[b - 1]` (a numpy-array-derived scalar) on every later one - a
    # bare Python float there is "weak-typed" once jnp.asarray'd, while an array-derived scalar
    # isn't, and IsocalcIntegrator.integrate (eqx.filter_jit) treats those as different abstract
    # types, forcing a second, otherwise-unnecessary trace/compile for the first segment alone.
    # Matching the type here up front keeps every segment on the one compiled program.
    t_start_seconds = np.asarray(options.t_start / const.s2yr)  # simulation start time [s]
    t_end_seconds = t_end / const.s2yr  # simulation end time [s]
    duration = t_end_seconds - t_start_seconds  # simulation duration [s]
    delta_t = duration / n_tot  # timestep [s]

    ###_____Set initial values____###

    ### atmodeller interior
    # Ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as
    # isofate_species_abund above.
    isofate_species_abund_int = np.zeros(7)
    if options.n_atmodeller == 0:
        T_surf_analytic = 0
        T_surf_atmod = 0
    if N_H != 0 and N_D != 0:  # needed to allow D and H to outgas from mantle
        X_DH = N_D / (N_H + N_D)  # ignores D in mantle

    # build atmodeller model for interior-atmosphere coupling
    interior_atmosphere = build_atmodeller(Mp, surface_radius=planet.rocky_radius)
    # Ordered per interior_atmosphere's own gas-phase species (SpeciesCollection), not hand-typed
    # - see the molecule-array declarations/writes below, which are driven by this tuple so they
    # can never drift out of sync with atmodeller's actual species set/order.
    tracked_species: tuple[TrackedGasSpecies, ...] = get_tracked_gas_species(interior_atmosphere)
    # Warm-starts each AtmodellerCoupler call from the previous call's converged solution instead
    # of solving cold every time - consecutive calls are a tiny physical perturbation apart, so
    # this drastically cuts the number of Newton iterations needed. None on the first call.
    atmod_initial_guess = None

    atmod_full_output = {}  # dictionary to store atmodeller full output

    # background_mantle_mass reads atmodeller's own mantle-mass partitioning (from the
    # core_mass_fraction build_atmodeller passed to Planet.from_species) directly, rather than
    # re-deriving/duplicating it here.
    mantle_iron_state: MantleIronState | None = None
    if options.mantle_iron is not None:
        mantle_mass = float(interior_atmosphere.parameters.state.background_mantle_mass)
        mantle_iron_state = MantleIronState.initial(options.mantle_iron, mantle_mass)

    ###_____Initialize arrays_____###

    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a = t_start_seconds + delta_t * np.arange(1, n_tot + 1)

    # Atmospheric/mantle molecule number history, ordered per interior_atmosphere's own gas-phase
    # SpeciesCollection (see tracked_species above), keyed by isofate's human-readable label
    # (e.g. "SO2", not atmodeller's canonical "O2S_g"). Stay all-zero when n_atmodeller == 0.
    y_a_int = np.zeros((n_tot, 7))  # mantle number array [atoms]
    gas_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    melt_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    fO2_a = np.zeros(n_tot)  # fugacity array [bar]
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot)  # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Integrate_____###

    integrator = IsocalcIntegrator(parameters)

    if options.n_atmodeller == 0:
        sol = integrator.integrate(
            jnp.asarray(t_start_seconds),
            jnp.asarray(t_a),
            jnp.asarray(isofate_species_abund),
        )
        y_a = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
        alg_a = integrator.diagnostics(jnp.asarray(t_a), y_a)
        Matm_a = alg_a.atmosphere_mass
        fatm_a = alg_a.atmosphere_mass_fraction
        Renv_a = alg_a.convective_envelope_thickness
        Rp_a = alg_a.escape_radius
        Vpot_a = alg_a.gravitational_potential
        phi_a = alg_a.mass_flux
        phic_a = alg_a.critical_mass_flux
        Phi_a = alg_a.number_flux
        x_a = alg_a.atom_fractions
        # Matches isocalc's per-output-step estimate (instantaneous rate * the output grid's fixed
        # delta_t) - not literally "mass lost between saved points".
        Mloss_a = phi_a * alg_a.area * delta_t
    else:
        atomic_masses = DEFAULT_SPECIES.atomic_masses
        y = np.asarray(isofate_species_abund, dtype=float)

        Matm_a = np.zeros(n_tot)
        fatm_a = np.zeros(n_tot)
        Renv_a = np.zeros(n_tot)
        Rp_a = np.zeros(n_tot)
        Vpot_a = np.zeros(n_tot)
        phi_a = np.zeros(n_tot)
        phic_a = np.zeros(n_tot)
        Mloss_a = np.zeros(n_tot)
        Phi_a = np.zeros((n_tot, 7))
        x_a = np.zeros((n_tot, 7))
        y_a = np.zeros((n_tot, 7))

        # EXHAUSTION_FRACTION (integrators.py) applied here against the run's true initial mass -
        # IsocalcIntegrator.integrate's internal event is relative to each segment's own
        # (shrinking) y0 and would otherwise under-detect exhaustion late in a run.
        M_atm0_run = float(np.dot(y, atomic_masses))

        def _fill_exhausted(start: int) -> None:
            """Fills [start:n_tot) with isocalc's exact post-exhaustion convention (see
            isocalc's `if M_atm <= 0 or sum(y) <= 0` block)."""
            Matm_a[start:] = 0
            fatm_a[start:] = 0
            Renv_a[start:] = 0
            Rp_a[start:] = planet.rocky_radius
            Vpot_a[start:] = tidal_gravitational_potential(system, planet.rocky_radius)
            phi_a[start:] = 0
            Mloss_a[start:] = 0
            y_a[start:] = 0
            y_a_int[start:] = 0
            for label in gas_num_a:
                gas_num_a[label][start:] = 0
                melt_num_a[label][start:] = 0
            x_a[start:] = x_a[start - 1]  # carry the last molar concentration forward
            Phi_a[start:] = 0
            if options.save_molecules == True:
                fO2_a[start:] = 0

        exhausted = False
        boundaries = list(range(0, n_tot, options.n_atmodeller))
        for i, b in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n_tot

            if exhausted:
                continue

            ### Stop simulation when entire atmosphere is lost
            M_atm_current = float(np.dot(y, atomic_masses))
            if M_atm_current <= EXHAUSTION_FRACTION * M_atm0_run:
                _fill_exhausted(b)
                atmod_full_output = nan_full_output()
                exhausted = True
                continue

            # needed to allow D and H to outgas from mantle
            if y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
                X_DH = (y[2] + isofate_species_abund_int[2]) / (
                    y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
                )  # assumes D/H is in equilibrium between interior and atmosphere

            # T/mu/escape_radius at the segment boundary, via the same shared physics chain
            # IsocalcIntegrator.vector_field uses during integration (its algebraic method) - avoids
            # reimplementing the mu/escape_radius formulas here. Evaluated at t_a[b] - matching
            # isocalc's own iteration-b physics (R_env(..., t_a[n], ...)) - not at this
            # segment's integration-start time (t0_chunk, below): R_env's age-dependent thermal
            # contraction term is sensitive to this at early times (t_a[0] vs t_start_seconds
            # differ by a large relative amount when t_start_seconds is itself small), even though
            # the two converge as the run progresses and one delta_t becomes negligible next to the
            # accumulated age.
            t0_chunk = t_start_seconds if i == 0 else t_a[b - 1]
            alg_boundary = integrator.algebraic(jnp.asarray(t_a[b]), jnp.asarray(y))
            mu = float(alg_boundary.mean_mu)
            escape_radius = float(alg_boundary.escape_radius)

            ##### run atmodeller ######
            # Species-level diagnostics (step.atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
            # are only read below when save_molecules is True; otherwise the narrow extraction
            # (element number_moles + gas mass only) is all this loop needs.
            step = run_atmodeller_step(
                T,
                escape_radius,
                mu,
                options.melt_fraction_override,
                mantle_iron_state,
                y,
                isofate_species_abund_int,
                X_DH,
                interior_atmosphere,
                atmod_initial_guess,
                full_output=options.save_molecules,
            )
            y = step.y
            isofate_species_abund_int = step.isofate_species_abund_int
            T_surf_analytic = step.T_surf_analytic
            T_surf_atmod = step.T_surf_atmod
            mantle_iron_state = step.mantle_iron_state
            atmod_initial_guess = step.atmod_initial_guess

            # Constant-until-the-next-call diagnostics, broadcast across this whole chunk - the
            # direct translation of isocalc's per-timestep carry-forward into the segmented
            # design, where there is no per-output-point loop to carry a value forward in.
            y_a_int[b:end] = isofate_species_abund_int
            T_surf_analytic_a[b:end] = T_surf_analytic
            T_surf_atmod_a[b:end] = T_surf_atmod
            if options.save_molecules == True:
                for sp in tracked_species:
                    gas_num_a[sp.label][b:end] = step.atmod_full[sp.gas_name]["gas"][
                        "number_moles"
                    ][0][0]
                    if sp.melt_name is not None:
                        melt_num_a[sp.label][b:end] = step.atmod_full[sp.melt_name][
                            "silicate_melt"
                        ]["number_moles"][0][0]
                    else:
                        melt_num_a[sp.label][b:end] = 0.0  # no solubility model / melt reservoir
                fO2_a[b:end] = step.atmod_full["O2_g"]["gas"]["activity"][0][0]

            sol = integrator.integrate(
                jnp.asarray(t0_chunk),
                jnp.asarray(t_a[b:end]),
                jnp.asarray(y),
            )
            y_a_chunk = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
            alg_a_chunk = integrator.diagnostics(jnp.asarray(t_a[b:end]), y_a_chunk)
            stop = IntegrationStop.from_solution(sol)

            # Exhausted from the first output row after the integrator's own (segment-relative)
            # exhaustion event, whose rows are inf, or from the first finite row below the
            # run-level floor, whichever comes first.
            M_atm_chunk = np.asarray(alg_a_chunk.atmosphere_mass)
            after_event = bool(stop.mass_lost) & (t_a[b:end] > float(stop.t))
            exhausted_mask = after_event | (M_atm_chunk <= EXHAUSTION_FRACTION * M_atm0_run)

            if exhausted_mask.any():
                k_rel = int(np.argmax(exhausted_mask))
                k = b + k_rel
                Matm_a[b:k] = M_atm_chunk[:k_rel]
                fatm_a[b:k] = np.asarray(alg_a_chunk.atmosphere_mass_fraction)[:k_rel]
                Renv_a[b:k] = np.asarray(alg_a_chunk.convective_envelope_thickness)[:k_rel]
                Rp_a[b:k] = np.asarray(alg_a_chunk.escape_radius)[:k_rel]
                Vpot_a[b:k] = np.asarray(alg_a_chunk.gravitational_potential)[:k_rel]
                phi_a[b:k] = np.asarray(alg_a_chunk.mass_flux)[:k_rel]
                phic_a[b:k] = np.asarray(alg_a_chunk.critical_mass_flux)[:k_rel]
                Phi_a[b:k] = np.asarray(alg_a_chunk.number_flux)[:k_rel]
                x_a[b:k] = np.asarray(alg_a_chunk.atom_fractions)[:k_rel]
                Mloss_a[b:k] = (
                    np.asarray(alg_a_chunk.mass_flux)[:k_rel]
                    * np.asarray(alg_a_chunk.area)[:k_rel]
                    * delta_t
                )
                y_a[b:k] = np.asarray(y_a_chunk)[:k_rel]

                _fill_exhausted(k)
                atmod_full_output = nan_full_output()
                exhausted = True
            else:
                Matm_a[b:end] = M_atm_chunk
                fatm_a[b:end] = np.asarray(alg_a_chunk.atmosphere_mass_fraction)
                Renv_a[b:end] = np.asarray(alg_a_chunk.convective_envelope_thickness)
                Rp_a[b:end] = np.asarray(alg_a_chunk.escape_radius)
                Vpot_a[b:end] = np.asarray(alg_a_chunk.gravitational_potential)
                phi_a[b:end] = np.asarray(alg_a_chunk.mass_flux)
                phic_a[b:end] = np.asarray(alg_a_chunk.critical_mass_flux)
                Phi_a[b:end] = np.asarray(alg_a_chunk.number_flux)
                x_a[b:end] = np.asarray(alg_a_chunk.atom_fractions)
                Mloss_a[b:end] = (
                    np.asarray(alg_a_chunk.mass_flux) * np.asarray(alg_a_chunk.area) * delta_t
                )
                y_a[b:end] = np.asarray(y_a_chunk)
                y = np.asarray(y_a_chunk[-1])

        # save final molecular abundances - read-only diagnostic snapshot, does not feed back
        # into the trajectory (mirrors isocalc's `if n == options.n_steps - 1: ...` call).
        if not exhausted:
            t_final = float(t_a[-1])
            alg_final = integrator.algebraic(jnp.asarray(t_final), jnp.asarray(y))
            mu = float(alg_final.mean_mu)
            escape_radius = float(alg_final.escape_radius)
            atmod_sol = AtmodellerCoupler(
                T,
                escape_radius,
                mu,
                options.melt_fraction_override,
                mantle_iron_state,
                *aggregate_D_into_H(y),
                *aggregate_D_into_H(isofate_species_abund_int),
                interior_atmosphere,
                initial_guess=atmod_initial_guess,
            )[1]
            atmod_full_output = extract_full_output(atmod_sol)

    # needed to allow D and H to outgas from mantle - X_DH stays inert while atmodeller is
    # disabled (isofate_species_abund_int never becomes nonzero) i.e. the n_atmodeller == 0
    # branch above.
    if y_a[-1, 0] + y_a[-1, 2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
        X_DH = (y_a[-1, 2] + isofate_species_abund_int[2]) / (
            y_a[-1, 0] + y_a[-1, 2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
        )  # assumes D/H is in equilibrium between interior and atmosphere

    # save results
    solutions = {
        "time": t_a,
        "Rp": Rp_a,
        "Ratm": Renv_a,
        "Matm": Matm_a,
        "Vpot": Vpot_a,
        "fatm": fatm_a,
        "Mloss": Mloss_a,
        "phi": phi_a,
        "phic": phic_a,
        # y_a/y_a_int columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S);
        # .copy() keeps each output array independent, matching the pre-array-refactor behavior.
        "N_H": y_a[:, 0].copy(),
        "N_He": y_a[:, 1].copy(),
        "N_D": y_a[:, 2].copy(),
        "N_O": y_a[:, 3].copy(),
        "N_C": y_a[:, 4].copy(),
        "N_N": y_a[:, 5].copy(),
        "N_S": y_a[:, 6].copy(),
        "N_H_int": y_a_int[:, 0].copy(),
        "N_He_int": y_a_int[:, 1].copy(),
        "N_D_int": y_a_int[:, 2].copy(),
        "N_O_int": y_a_int[:, 3].copy(),
        "N_C_int": y_a_int[:, 4].copy(),
        "N_N_int": y_a_int[:, 5].copy(),
        "N_S_int": y_a_int[:, 6].copy(),
        "x1": x_a[:, 0].copy(),
        "x2": x_a[:, 1].copy(),
        "x3": x_a[:, 2].copy(),
        "x4": x_a[:, 3].copy(),
        "x5": x_a[:, 4].copy(),
        "x6": x_a[:, 5].copy(),
        "x7": x_a[:, 6].copy(),
        # Phi_a columns are ordered per isofate.species.SYMBOLS (H, He, D, O, C, N, S); .copy()
        # keeps each output array independent, matching the pre-array-refactor behavior.
        "Phi_H": Phi_a[:, 0].copy(),
        "Phi_He": Phi_a[:, 1].copy(),
        "Phi_D": Phi_a[:, 2].copy(),
        "Phi_O": Phi_a[:, 3].copy(),
        "Phi_C": Phi_a[:, 4].copy(),
        "Phi_N": Phi_a[:, 5].copy(),
        "Phi_S": Phi_a[:, 6].copy(),
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
    }
    if options.save_molecules == True:
        for label, arr in gas_num_a.items():
            solutions[f"n_{label}_a"] = arr
        for label, arr in melt_num_a.items():
            solutions[f"n_{label}_a_int"] = arr
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions


def isocalc_jax3(
    parameters: Parameters,
    t_end=5e9,
    isofate_species_abund: Array = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
    euler: bool = False,
):
    """Atmospheric escape with event-triggered Atmodeller coupling.

    Replaces `isocalc_jax2`'s loop of separate solves: the integration runs through
    `isofate.integrators.integrate_segments`, which stops on an event, re-equilibrates the atmosphere
    and interior with Atmodeller (`AtmodellerCoupler.reequilibrate`), and restarts. The atmosphere
    and interior are first equilibrated once at the start time.

    The events are:
        - Elapsed time: every `IsocalcOptions.n_atmodeller` output steps, i.e. at exactly
          ``t_start + k * n_atmodeller * delta_t`` - `isocalc`'s cadence. `n_atmodeller = 0`
          switches the coupling off.
        - Mass loss: `IsocalcOptions.mass_loss_fraction` of the atmospheric mass lost since the
          last re-equilibration (by default only at exhaustion), and optionally
          `IsocalcOptions.species_loss_fraction` of any species.

    The mantle iron reaction is ignored for now
    (`IsocalcOptions.mantle_iron` is not read). Output rows after the atmosphere is exhausted are
    ``inf``, as `integrate_segments` leaves them.

    The reference for comparison is `isocalc` (manual fixed time step, re-equilibrating every
    `n_atmodeller` steps).

    Args:
        parameters: Parameters
        t_end: Simulation end time, i.e. system age at the end of the run [yr]. Defaults to
            ``5e9``.
        isofate_species_abund: Initial atmospheric abundances [atoms], ordered per
            `isofate.species.SYMBOLS` (H, He, D, O, C, N, S)
        euler: Integrate with isocalc's fixed-step forward Euler scheme
            (`isofate.integrators.integrate_segments_euler`: re-equilibrating every `n_atmodeller`
            steps, no events) instead of the adaptive solver with events. Defaults to ``False``.

    Returns:
        Dictionary with the same keys as `isocalc` (plus ``t_atmodeller``, the re-equilibration
        times [s])
    """
    options = parameters.isocalc_options

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps
    t_start_seconds = np.asarray(options.t_start / const.s2yr)  # simulation start time [s]
    t_end_seconds = t_end / const.s2yr  # simulation end time [s]
    duration = t_end_seconds - t_start_seconds  # simulation duration [s]
    delta_t = duration / n_tot  # timestep [s]
    # Output times [s]: row j is the state after j + 1 steps, so the last row is at t_end
    t_a = t_start_seconds + delta_t * np.arange(1, n_tot + 1)

    # Re-equilibrate every n_atmodeller output steps, like isocalc (a time event)
    integrator = IsocalcIntegrator(
        parameters,
        reequilibration_interval=(
            float(options.n_atmodeller * delta_t) if options.n_atmodeller != 0 else None
        ),
    )
    # Built once per run, outside jit (it builds the Atmodeller model). Also built when the
    # coupling is off, for the molecule output keys
    coupler = EventAtmodellerCoupler(parameters)
    tracked_species = coupler.tracked_species
    y_raw = jnp.asarray(isofate_species_abund, dtype=float)

    # TODO: temporary output assembly to match isocalc's keys - clean up
    y_a_int = np.zeros((n_tot, 7))  # interior number array [atoms]
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from the atmosphere descent [K]
    T_surf_atmod_a = np.zeros(n_tot)  # surface temperature used by Atmodeller [K]
    gas_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    melt_num_a: dict[str, np.ndarray] = {sp.label: np.zeros(n_tot) for sp in tracked_species}
    fO2_a = np.zeros(n_tot)  # O2 activity
    t_atmodeller = np.zeros(0)  # re-equilibration times [s]
    atmod_full_output: dict = {}

    ###_____Integrate_____###

    if options.n_atmodeller == 0:
        if euler:
            y_a, alg_a, _, _ = integrate_segments_euler(
                integrator, jnp.asarray(t_start_seconds), jnp.asarray(t_a), y_raw, n_tot
            )
        else:
            sol = integrator.integrate(jnp.asarray(t_start_seconds), jnp.asarray(t_a), y_raw)
            y_a = sol.ys[0]  # pyright: ignore[reportOptionalSubscript]
            alg_a = integrator.diagnostics(jnp.asarray(t_a), y_a)
    else:
        t_start_j = jnp.asarray(t_start_seconds)
        state0, y0 = coupler.reequilibrate(coupler.initial_state(), t_start_j, y_raw)
        if euler:
            y_a, alg_a, restarts, state = integrate_segments_euler(
                integrator,
                t_start_j,
                jnp.asarray(t_a),
                y0,
                options.n_atmodeller,
                on_mass_lost=coupler.reequilibrate,
                carry=state0,
            )
        else:
            y_a, alg_a, restarts, state = integrate_segments(
                integrator,
                t_start_j,
                jnp.asarray(t_a),
                y0,
                on_mass_lost=coupler.reequilibrate,
                carry=state0,
            )
        count = int(restarts.count)

        # TODO: temporary output assembly to match isocalc's keys - clean up.
        # One entry per equilibration: the initial one, then each restart.
        def stacked(initial, per_restart):
            return np.concatenate([np.asarray(initial)[None], np.asarray(per_restart)[:count]])

        carries: CouplerState = restarts.carry
        eq_t = stacked(t_start_seconds, restarts.t)
        eq_y_before = stacked(y_raw, restarts.y_before)
        y_int_states = stacked(state0.y_int, carries.y_int)
        T_surf_states = stacked(state0.surface_temperature, carries.surface_temperature)
        T_atmod_states = stacked(
            state0.surface_temperature_atmodeller, carries.surface_temperature_atmodeller
        )

        # Re-equilibrate once more at the end time when it is on the schedule (n_steps a multiple
        # of n_atmodeller), as isocalc does - integrate_segments never restarts at the last output
        # time. The last row then holds the re-equilibrated state.
        y_a = np.array(y_a)
        floor = EXHAUSTION_FRACTION * float(parameters.atmosphere_mass(y0))
        if (
            n_tot % options.n_atmodeller == 0
            and np.all(np.isfinite(y_a[-1]))
            and float(parameters.atmosphere_mass(jnp.asarray(y_a[-1]))) > floor
        ):
            state, y_end = coupler.reequilibrate(state, jnp.asarray(t_a[-1]), jnp.asarray(y_a[-1]))
            eq_t = np.append(eq_t, t_a[-1])
            eq_y_before = np.concatenate([eq_y_before, y_a[-1][None]])
            y_int_states = np.concatenate([y_int_states, np.asarray(state.y_int)[None]])
            T_surf_states = np.append(T_surf_states, float(state.surface_temperature))
            T_atmod_states = np.append(T_atmod_states, float(state.surface_temperature_atmodeller))
            y_a[-1] = np.asarray(y_end)
            alg_a = integrator.diagnostics(jnp.asarray(t_a), jnp.asarray(y_a))
        t_atmodeller = eq_t

        # The interior and surface temperatures are piecewise constant: each output time takes the
        # latest equilibration at or before it (within half an output spacing, since equilibration
        # times coincide with output times only to the last bit)
        half_gap = 0.5 * float(np.min(np.diff(t_a)))
        which = np.searchsorted(eq_t[1:] - half_gap, t_a, side="right")
        y_a_int = y_int_states[which]
        T_surf_analytic_a = T_surf_states[which]
        T_surf_atmod_a = T_atmod_states[which]

        if options.save_molecules:
            # Re-run each equilibration with full output, with exactly the inputs the hook had
            # (the pre-equilibration atmosphere and the previous interior), so it reproduces the
            # same equilibrium
            y_int_before_states = np.concatenate([np.zeros((1, 7)), y_int_states[:-1]])
            for j in range(len(eq_t)):
                y_before = jnp.asarray(eq_y_before[j])
                t_j = jnp.asarray(eq_t[j])
                output = jax.device_get(
                    coupler.run(
                        escape_radius(parameters, y_before, t_j),
                        parameters.atmosphere_mean_mu(y_before),
                        None,
                        y_before,
                        jnp.asarray(y_int_before_states[j]),
                        full_output=True,
                    ).output
                )
                rows = which == j
                for sp in tracked_species:
                    gas_num_a[sp.label][rows] = output[sp.gas_name]["gas"]["number_moles"][0][0]
                    if sp.melt_name is not None:
                        melt_num_a[sp.label][rows] = output[sp.melt_name]["silicate_melt"][
                            "number_moles"
                        ][0][0]
                fO2_a[rows] = output["O2_g"]["gas"]["activity"][0][0]

        # Final full snapshot (read-only diagnostic), as isocalc's "atmodeller_final"
        y_final = jnp.asarray(y_a[-1])
        if bool(jnp.all(jnp.isfinite(y_final))):
            t_final = jnp.asarray(t_a[-1])
            final = coupler.run(
                escape_radius(parameters, y_final, t_final),
                parameters.atmosphere_mean_mu(y_final),
                None,
                y_final,
                state.y_int,
                initial_guess=state.solution,
                full_output=True,
            )
            atmod_full_output = extract_full_output(jax.device_get(final.output))
        else:
            atmod_full_output = nan_full_output()

    y_a = np.asarray(y_a)
    Matm_a = np.asarray(alg_a.atmosphere_mass)
    fatm_a = np.asarray(alg_a.atmosphere_mass_fraction)
    Renv_a = np.asarray(alg_a.convective_envelope_thickness)
    Rp_a = np.asarray(alg_a.escape_radius)
    Vpot_a = np.asarray(alg_a.gravitational_potential)
    phi_a = np.asarray(alg_a.mass_flux)
    phic_a = np.asarray(alg_a.critical_mass_flux)
    Phi_a = np.asarray(alg_a.number_flux)
    x_a = np.asarray(alg_a.atom_fractions)
    Mloss_a = phi_a * np.asarray(alg_a.area) * delta_t

    # save results
    solutions = {
        "time": t_a,
        "Rp": Rp_a,
        "Ratm": Renv_a,
        "Matm": Matm_a,
        "Vpot": Vpot_a,
        "fatm": fatm_a,
        "Mloss": Mloss_a,
        "phi": phi_a,
        "phic": phic_a,
        **{f"N_{symbol}": y_a[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        **{f"N_{symbol}_int": y_a_int[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        **{f"x{i + 1}": x_a[:, i].copy() for i in range(len(SYMBOLS))},
        **{f"Phi_{symbol}": Phi_a[:, i].copy() for i, symbol in enumerate(SYMBOLS)},
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
        "t_atmodeller": t_atmodeller,
    }
    if options.save_molecules:
        for label, arr in gas_num_a.items():
            solutions[f"n_{label}_a"] = arr
        for label, arr in melt_num_a.items():
            solutions[f"n_{label}_a_int"] = arr
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions

