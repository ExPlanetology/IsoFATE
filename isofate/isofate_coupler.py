# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main IsoFATE script for coupled model."""

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
from isofate.constants import const
from isofate.engine import _algebraic, _integrate_isocalc_jax
from isofate.escape import EscapeState
from isofate.escape_fractionation import Phi_1_2, Phi_minor_species
from isofate.isofunks import R_atm, R_env
from isofate.mantle_iron import MantleIronState
from isofate.parameters import Parameters
from isofate.species import DEFAULT_SPECIES, SYMBOLS
from isofate.system import Planet
from isofate.utils import gravitational_acceleration


def isocalc(
    parameters: Parameters,
    time=5e9,
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
    #  - time: total simulation time; scalar [yr]
    #  - isofate_species_abund: initial abundance [atoms] for each tracked species, ordered as in
    #  isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
    #  - options: mode switches and tuning constants unrelated to escape mechanism, fixed for
    #  the whole run - see IsocalcOptions for the full list (rad_evol, melt_fraction_override,
    #  mu, n_steps, t0, thermal, n_atmodeller, save_molecules, mantle_iron, dynamic_phi)
    #  - escape: escape-mechanism instance (isofate.escape.EscapeMechanism) controlling the
    #  atmospheric mass-flux calculation each timestep; defaults to XUVEscape(), equivalent to
    #  today's default mechanism="XUV", RR=True. See isofate.escape for XUVEscape, CPMLEscape,
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
    # matching isocalc_jax's `_algebraic` (see engine.py). (`options.mantle_iron` similarly seeds a
    # local `mantle_iron_state` below, once `interior_atmosphere` is available.)
    system = parameters.system
    options = parameters.isocalc_options
    escape = parameters.escape_mechanism
    escape_number_flux = parameters.escape_number_flux

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, N_He, N_D, N_O, N_C, N_N, N_S = isofate_species_abund

    # d, T, and Fp are confirmed fixed for the whole run (never reassigned anywhere below), so
    # they're read from `system` once, here. `system.star.mass` is no longer cached separately -
    # `system.tidal_reduction_factor(Rp)` reads it directly (see below).
    planet: Planet = system.planet
    d: ArrayLike = system.semi_major_axis
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
    t0_seconds = options.t0 / const.s2yr  # simulation start time [s]
    t = time / const.s2yr - t0_seconds  # total simulation time [s]
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

    ### atmosphere
    # Atmospheric number of atoms per species [atoms], ordered per
    # isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as isofate_species_abund above.
    # M_atm/f_atm are not seeded here - they're derived fresh from y at the top of every loop
    # iteration below (see the loop body).
    y = np.array(isofate_species_abund, dtype=float)
    ###_____Initialize arrays_____###

    t_a = delta_t * np.linspace(1, n_tot + 1, n_tot) + t0_seconds  # time array [s]

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

    ###_____Loop through timesteps_____###

    for n in range(n_tot):
        # Derived fresh from y every iteration (mass-conservation identity: the atmosphere's
        # total mass is exactly the sum of its constituent atoms' masses), rather than tracked as
        # separately-updated state - matches isocalc_jax's design (see engine.py's _algebraic).
        M_atm = np.dot(y, atomic_masses)
        f_atm = M_atm / Mp

        ### Stop simulation when entire atmosphere is lost
        if M_atm <= 0 or np.sum(y) <= 0:
            Matm_a[n:] = 0  # M_atm #Matm_a[n-1]
            fatm_a[n:] = 0  # f_atm #fatm_a[n-1]
            Renv_a[n:] = 0  # radius_env #Renv_a[n-1]
            Rp_a[n:] = planet.rocky_radius
            Vpot_a[n:] = system.gravitational_potential()

            phi_a[n:] = 0
            Mloss_a[n:] = 0

            y_a[n:] = 0
            y_a_int[n:] = 0
            for label in gas_num_a:
                gas_num_a[label][n:] = 0
                melt_num_a[label][n:] = 0
            x_a[n:] = x_a[n - 1]  # carry the last molar concentration forward
            Phi_a[n:] = 0

            # atmodeller full ouput for monte carlo runs
            if options.n_atmodeller != 0:
                atmod_full_output = nan_full_output()
                if options.save_molecules == True:
                    for label in gas_num_a:
                        gas_num_a[label][n:] = 0
                        melt_num_a[label][n:] = 0
                    fO2_a[n:] = 0

            break

        # time-variable average atomic mass
        N_tot = np.sum(y)
        mu = escape_number_flux.species.atmosphere_mean_mu(y)
        R_B = system.bondi_radius(mu, T)  # recomputed from the current mu, not a fixed bootstrap

        if options.rad_evol == False:
            radius_env = 0
            radius_atm = 0
            radius_p = planet.rocky_radius
        else:
            radius_env = R_env(Mp, f_atm, Fp, t_a[n], options.thermal)
            radius_atm = R_atm(T, Mp, planet.rocky_radius, radius_env, mu)
            radius_p = planet.rocky_radius + radius_atm + radius_env
            # limits Rp to the min of Bondi/Hill/Lopez+Fortney radius; plain min() avoids numpy's
            # array-construction/dispatch overhead on a 3-scalar comparison run every timestep
            radius_p = min(R_B, R_H, radius_p)

        Vpot = system.gravitational_potential(radius_p)
        A = 4 * np.pi * radius_p**2

        # sets mass flux [kg/m2/s]
        state = EscapeState(
            radius_p=radius_p,
            Mp=Mp,
            T=T,
            Vpot=Vpot,
            d=d,
            A=A,
            mu=mu,
            radius_env=radius_env,
            f_atm=f_atm,
            t_now=t_a[n],
            t_total=t,
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

        # record values
        Matm_a[n] = M_atm
        fatm_a[n] = f_atm
        Renv_a[n] = (
            radius_env  # this will still change even with Rp limited to min(R_Bondi, R_Hill)
        )
        Rp_a[n] = radius_p
        Vpot_a[n] = Vpot
        phi_a[n] = phi
        phic_a[n] = phi_c
        Mloss_a[n] = mass_loss

        y_a[n] = y
        y_a_int[n] = isofate_species_abund_int
        x_a[n] = x
        Phi_a[n] = Phi

        ##### run atmodeller ######
        if options.n_atmodeller != 0:  # save final molecular abundances on last time step
            if n == options.n_steps - 1:
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
            if n % options.n_atmodeller == 0:  # run atmodeller every n_atmodeller steps.
                # Species-level diagnostics (step.atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
                # are only read below when save_molecules is True; otherwise the narrow extraction
                # (element number_moles + gas mass only) is all this loop needs.
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
                # Not read into M_atm here (unlike before): M_atm is derived fresh from y at the
                # top of every iteration (see above), and step.M_atm is exactly
                # dot(step.y, atomic_masses) anyway (see atmodeller_coupler.py's
                # run_atmodeller_step), so next iteration's derived M_atm already reflects this.
                T_surf_analytic = step.T_surf_analytic
                T_surf_atmod = step.T_surf_atmod
                mantle_iron_state = step.mantle_iron_state
                atmod_initial_guess = step.atmod_initial_guess
                if options.save_molecules == True:
                    for sp in tracked_species:
                        gas_num_a[sp.label][n] = step.atmod_full[sp.gas_name]["gas"][
                            "number_moles"
                        ][0][0]
                        if sp.melt_name is not None:
                            melt_num_a[sp.label][n] = step.atmod_full[sp.melt_name][
                                "silicate_melt"
                            ]["number_moles"][0][0]
                        else:
                            melt_num_a[sp.label][n] = 0.0  # no solubility model / melt reservoir
                    fO2_a[n] = step.atmod_full["O2_g"]["gas"]["activity"][0][0]
            else:
                for label in gas_num_a:
                    gas_num_a[label][n] = gas_num_a[label][n - 1]
                    melt_num_a[label][n] = melt_num_a[label][n - 1]
                fO2_a[n] = fO2_a[n - 1]
        T_surf_analytic_a[n] = T_surf_analytic
        T_surf_atmod_a[n] = T_surf_atmod

        # needed to allow D and H to outgas from mantle
        if y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
            X_DH = (y[2] + isofate_species_abund_int[2]) / (
                y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
            )  # assumes D/H is in equilibrium between interior and atmosphere

        # advance to next step - M_atm/f_atm are not updated here: next iteration re-derives them
        # fresh from this same y (see the top of the loop above).
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
    time=5e9,
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
    #  - time: total simulation time; scalar [yr]
    #  - isofate_species_abund: initial abundance [atoms] for each tracked species, ordered as in
    #  isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
    #  - options: mode switches and tuning constants unrelated to escape mechanism, fixed for
    #  the whole run - see IsocalcOptions for the full list (rad_evol, melt_fraction_override,
    #  mu, n_steps, t0, thermal, n_atmodeller, save_molecules, mantle_iron, dynamic_phi)
    #  - escape: escape-mechanism instance (isofate.escape.EscapeMechanism) controlling the
    #  atmospheric mass-flux calculation each timestep; defaults to XUVEscape(), equivalent to
    #  today's default mechanism="XUV", RR=True. See isofate.escape for XUVEscape, CPMLEscape,
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
    # straight through to `_integrate_isocalc_jax`, which pulls them off internally.
    # `planet.f_atm` is NOT read here (unlike isocalc): `_integrate_isocalc_jax` derives
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
    # run). `_integrate_isocalc_jax` re-derives T/d/Fp/Mp from `system` itself (already one of its
    # arguments), so they don't need to be threaded through separately. `system.star.mass` is no
    # longer cached separately - `system.tidal_reduction_factor(Rp)` reads it directly (see
    # below).
    planet: Planet = system.planet
    Mp = planet.mass

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps  # timesteps
    t0_seconds = options.t0 / const.s2yr  # simulation start time [s]
    # Named t_total (not the bare `t` isocalc uses) - `t` here would collide with diffrax's own
    # integration-time parameter name in the vector field/postprocessing closures below.
    t_total = time / const.s2yr - t0_seconds  # total simulation time [s]
    delta_t = t_total / n_tot  # timestep [s]

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
    # _integrate_isocalc_jax's docstring for why - it's derived from this y instead.
    # y = np.array(isofate_species_abund, dtype=float)
    ###_____Initialize arrays_____###

    t_a = delta_t * np.linspace(1, n_tot + 1, n_tot) + t0_seconds  # time array [s]

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

    y_a, alg_a = _integrate_isocalc_jax(
        jnp.asarray(t0_seconds),
        jnp.asarray(t_a),
        jnp.asarray(isofate_species_abund),
        options.thermal,
        jnp.asarray(t_total),
        parameters,
    )
    Matm_a = alg_a["M_atm"]
    fatm_a = alg_a["f_atm"]
    Renv_a = alg_a["radius_env"]
    Rp_a = alg_a["radius_p"]
    Vpot_a = alg_a["Vpot"]
    phi_a = alg_a["phi"]
    phic_a = alg_a["phi_c"]
    Phi_a = alg_a["Phi"]
    x_a = alg_a["x"]
    # Matches isocalc's per-output-step estimate (instantaneous rate * the output grid's fixed
    # delta_t) - not literally "mass lost between saved points" (which the adaptive trajectory
    # could give more precisely via diff(Matm_a), but that's a different definition; keeping this
    # one for now, matching the original quantity).
    Mloss_a = phi_a * alg_a["A"] * delta_t

    # TODO: For a later refactor, decide if holding the last-integrated state constant (done above,
    # via the finite_mask/last_finite_idx handling of diffrax's SaveAt(ts=...)+Event interaction,
    # which otherwise returns inf for entries at/after the event) is the right post-exhaustion
    # behavior to keep long-term, as opposed to isocalc's original per-field zero-vs-hold choices
    # (e.g. Rp_a held at planet.rocky_radius, not the last pre-exhaustion Rp).

    # TODO: Will add back eventually, once JAX refactor is working for the simpler case
    # elif options.dynamic_phi == True:
    #     N_values = y  # ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
    #     abundances_with_idx = [(i, N_values[i]) for i in range(7)]
    #     abundances_with_idx.sort(key=lambda item: item[1], reverse=True)

    #     most_abundant_idx = abundances_with_idx[0][0]
    #     second_most_abundant_idx = abundances_with_idx[1][0]

    #     # Get masses and scale heights for the two most abundant species
    #     scale_heights = DEFAULT_SPECIES.scale_heights(T, g)

    #     mass_most = atomic_masses[most_abundant_idx]
    #     mass_second = atomic_masses[second_most_abundant_idx]
    #     H_most = scale_heights[most_abundant_idx]
    #     H_second = scale_heights[second_most_abundant_idx]

    #     # Determine which is lighter (species 1) and heavier (species 2) by MASS
    #     if mass_most <= mass_second:
    #         light_dominant_idx = most_abundant_idx  # species 1 (lighter)
    #         heavy_dominant_idx = second_most_abundant_idx  # species 2 (heavier)
    #         mass_1 = mass_most
    #         mass_2 = mass_second
    #         H_1 = H_most
    #         H_2 = H_second
    #     else:
    #         light_dominant_idx = second_most_abundant_idx  # species 1 (lighter)
    #         heavy_dominant_idx = most_abundant_idx  # species 2 (heavier)
    #         mass_1 = mass_second
    #         mass_2 = mass_most
    #         H_1 = H_second
    #         H_2 = H_most

    #     # Calculate molar fractions for the two dominant species
    #     N1 = N_values[light_dominant_idx]  # lightest dominant (species 1)
    #     N2 = N_values[heavy_dominant_idx]  # heaviest dominant (species 2)
    #     N_tot_binary = N1 + N2

    #     X1 = N1 / N_tot_binary  # molar fraction of species 1 in binary mixture
    #     X2 = N2 / N_tot_binary  # molar fraction of species 2 in binary mixture
    #     MU = X1 * mass_1 + X2 * mass_2

    #     # Calculate binary diffusion coefficient between the two dominant species
    #     light_name = species_names[light_dominant_idx]  # species 1
    #     heavy_name = species_names[heavy_dominant_idx]  # species 2

    #     b = DEFAULT_SPECIES.binary_diffusion.get(light_name, heavy_name, T)

    #     # Calculate escape fluxes for the two dominant species
    #     Phi_1_calc, Phi_2_calc, phi_c = Phi_1_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)

    #     # Assign fluxes to correct species based on light/heavy dominant indices - ordered
    #     # per isofate.species.SYMBOLS (H, He, D, O, C, N, S)
    #     Phi = np.zeros(7)
    #     Phi[light_dominant_idx] = Phi_1_calc
    #     Phi[heavy_dominant_idx] = Phi_2_calc

    #     # Calculate fluxes for remaining species using corrected generalized function
    #     for i in range(7):
    #         if i != light_dominant_idx and i != heavy_dominant_idx:
    #             Phi[i] = Phi_minor_species(
    #                 Phi_1_calc,
    #                 Phi_2_calc,
    #                 H_1,
    #                 H_2,
    #                 scale_heights[i],
    #                 N_values,
    #                 T,
    #                 i,
    #                 light_dominant_idx,
    #                 heavy_dominant_idx,
    #             )

    # TODO: Will add back eventually, once JAX refactor is working for the simpler case
    ##### run atmodeller ######
    # if options.n_atmodeller != 0:  # save final molecular abundances on last time step
    #     if n == options.n_steps - 1:
    #         atmod_sol = AtmodellerCoupler(
    #             T,
    #             radius_p,
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
    #             radius_p,
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
    time=5e9,
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
    # cadence), each segment run as a normal adaptive _integrate_isocalc_jax call, with Atmodeller
    # called from Python between segments exactly as isocalc's discrete loop calls it. See the
    # "options.n_atmodeller != 0" branch below.
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
    # straight through to `_integrate_isocalc_jax`, which pulls them off internally.
    # `planet.f_atm` is NOT read here (unlike isocalc): `_integrate_isocalc_jax` derives
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
    # run). `_integrate_isocalc_jax` re-derives T/d/Fp/Mp from `system` itself (already one of its
    # arguments), so they don't need to be threaded through separately. `system.star.mass` is no
    # longer cached separately - `system.tidal_reduction_factor(Rp)` reads it directly (see
    # below).
    planet: Planet = system.planet
    Mp = planet.mass
    T = system.equilibrium_temperature  # fixed for the whole run

    ###_____Initialize timesteps_____###

    n_tot = options.n_steps  # timesteps
    t0_seconds = options.t0 / const.s2yr  # simulation start time [s]
    # Named t_total (not the bare `t` isocalc uses) - `t` here would collide with diffrax's own
    # integration-time parameter name in the vector field/postprocessing closures below.
    t_total = time / const.s2yr - t0_seconds  # total simulation time [s]
    delta_t = t_total / n_tot  # timestep [s]

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

    t_a = delta_t * np.linspace(1, n_tot + 1, n_tot) + t0_seconds  # time array [s]

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

    if options.n_atmodeller == 0:
        y_a, alg_a = _integrate_isocalc_jax(
            jnp.asarray(t0_seconds),
            jnp.asarray(t_a),
            jnp.asarray(isofate_species_abund),
            options.thermal,
            jnp.asarray(t_total),
            parameters,
        )
        Matm_a = alg_a["M_atm"]
        fatm_a = alg_a["f_atm"]
        Renv_a = alg_a["radius_env"]
        Rp_a = alg_a["radius_p"]
        Vpot_a = alg_a["Vpot"]
        phi_a = alg_a["phi"]
        phic_a = alg_a["phi_c"]
        Phi_a = alg_a["Phi"]
        x_a = alg_a["x"]
        # Matches isocalc's per-output-step estimate (instantaneous rate * the output grid's fixed
        # delta_t) - not literally "mass lost between saved points".
        Mloss_a = phi_a * alg_a["A"] * delta_t
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

        # 1e-6 matches _exhausted's own floor (engine.py), applied here against the run's true
        # initial mass/count - _integrate_isocalc_jax's internal event is relative to each
        # segment's own (shrinking) y0 and would otherwise under-detect exhaustion late in a run.
        exhaustion_fraction = 1e-6
        M_atm0_run = float(np.dot(y, atomic_masses))
        sum_y0_run = float(np.sum(y))

        def _fill_exhausted(start: int) -> None:
            """Fills [start:n_tot) with isocalc's exact post-exhaustion convention (see
            isocalc's `if M_atm <= 0 or sum(y) <= 0` block)."""
            Matm_a[start:] = 0
            fatm_a[start:] = 0
            Renv_a[start:] = 0
            Rp_a[start:] = planet.rocky_radius
            Vpot_a[start:] = system.gravitational_potential()
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
            sum_y_current = float(np.sum(y))
            if (
                M_atm_current <= exhaustion_fraction * M_atm0_run
                or sum_y_current <= exhaustion_fraction * sum_y0_run
            ):
                _fill_exhausted(b)
                atmod_full_output = nan_full_output()
                exhausted = True
                continue

            # needed to allow D and H to outgas from mantle
            if y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
                X_DH = (y[2] + isofate_species_abund_int[2]) / (
                    y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
                )  # assumes D/H is in equilibrium between interior and atmosphere

            # T/mu/radius_p at the segment boundary, via the same shared physics chain
            # _vector_field uses during integration (engine.py's _algebraic) - avoids
            # reimplementing the mu/radius_p formulas here. Evaluated at t_a[b] - matching
            # isocalc's own iteration-b physics (R_env(..., t_a[n], ...)) - not at this
            # segment's integration-start time (t0_chunk, below): R_env's age-dependent thermal
            # contraction term is sensitive to this at early times (t_a[0] vs t0_seconds differ
            # by a large relative amount when t0_seconds is itself small), even though the two
            # converge as the run progresses and one delta_t becomes negligible next to the
            # accumulated age.
            t0_chunk = t0_seconds if i == 0 else t_a[b - 1]
            alg_boundary = _algebraic(
                jnp.asarray(t_a[b]),
                jnp.asarray(y),
                options.thermal,
                jnp.asarray(t_total),
                parameters,
            )
            mu = float(alg_boundary["mu"])
            radius_p = float(alg_boundary["radius_p"])

            ##### run atmodeller ######
            # Species-level diagnostics (step.atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
            # are only read below when save_molecules is True; otherwise the narrow extraction
            # (element number_moles + gas mass only) is all this loop needs.
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

            y_a_chunk, alg_a_chunk = _integrate_isocalc_jax(
                jnp.asarray(t0_chunk),
                jnp.asarray(t_a[b:end]),
                jnp.asarray(y),
                options.thermal,
                jnp.asarray(t_total),
                parameters,
            )

            # Re-derived from this chunk's own trajectory against the run-level floor above,
            # rather than trusting _integrate_isocalc_jax's internal (segment-relative) event -
            # robust whether or not that internal event fired early (see isocalc_jax2 design
            # notes / the project plan).
            M_atm_chunk = np.asarray(alg_a_chunk["M_atm"])
            sum_y_chunk = np.asarray(jnp.sum(y_a_chunk, axis=1))
            exhausted_mask = (M_atm_chunk <= exhaustion_fraction * M_atm0_run) | (
                sum_y_chunk <= exhaustion_fraction * sum_y0_run
            )

            if exhausted_mask.any():
                k_rel = int(np.argmax(exhausted_mask))
                k = b + k_rel
                Matm_a[b:k] = M_atm_chunk[:k_rel]
                fatm_a[b:k] = np.asarray(alg_a_chunk["f_atm"])[:k_rel]
                Renv_a[b:k] = np.asarray(alg_a_chunk["radius_env"])[:k_rel]
                Rp_a[b:k] = np.asarray(alg_a_chunk["radius_p"])[:k_rel]
                Vpot_a[b:k] = np.asarray(alg_a_chunk["Vpot"])[:k_rel]
                phi_a[b:k] = np.asarray(alg_a_chunk["phi"])[:k_rel]
                phic_a[b:k] = np.asarray(alg_a_chunk["phi_c"])[:k_rel]
                Phi_a[b:k] = np.asarray(alg_a_chunk["Phi"])[:k_rel]
                x_a[b:k] = np.asarray(alg_a_chunk["x"])[:k_rel]
                Mloss_a[b:k] = (
                    np.asarray(alg_a_chunk["phi"])[:k_rel]
                    * np.asarray(alg_a_chunk["A"])[:k_rel]
                    * delta_t
                )
                y_a[b:k] = np.asarray(y_a_chunk)[:k_rel]

                _fill_exhausted(k)
                atmod_full_output = nan_full_output()
                exhausted = True
            else:
                Matm_a[b:end] = M_atm_chunk
                fatm_a[b:end] = np.asarray(alg_a_chunk["f_atm"])
                Renv_a[b:end] = np.asarray(alg_a_chunk["radius_env"])
                Rp_a[b:end] = np.asarray(alg_a_chunk["radius_p"])
                Vpot_a[b:end] = np.asarray(alg_a_chunk["Vpot"])
                phi_a[b:end] = np.asarray(alg_a_chunk["phi"])
                phic_a[b:end] = np.asarray(alg_a_chunk["phi_c"])
                Phi_a[b:end] = np.asarray(alg_a_chunk["Phi"])
                x_a[b:end] = np.asarray(alg_a_chunk["x"])
                Mloss_a[b:end] = (
                    np.asarray(alg_a_chunk["phi"]) * np.asarray(alg_a_chunk["A"]) * delta_t
                )
                y_a[b:end] = np.asarray(y_a_chunk)
                y = np.asarray(y_a_chunk[-1])

        # save final molecular abundances - read-only diagnostic snapshot, does not feed back
        # into the trajectory (mirrors isocalc's `if n == options.n_steps - 1: ...` call).
        if not exhausted:
            t_final = float(t_a[-1])
            alg_final = _algebraic(
                jnp.asarray(t_final),
                jnp.asarray(y),
                options.thermal,
                jnp.asarray(t_total),
                parameters,
            )
            mu = float(alg_final["mu"])
            radius_p = float(alg_final["radius_p"])
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
