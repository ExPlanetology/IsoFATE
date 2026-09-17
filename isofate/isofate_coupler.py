# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main IsoFATE script for coupled model."""

import numpy as np
from jaxtyping import ArrayLike

from isofate.atmodeller_coupler import AtmodellerCoupler, aggregate_D_into_H, build_atmodeller
from isofate.constants import binary_diffusion, const
from isofate.isofunks import (
    Phi_1,
    Phi_2,
    Phi_C_Z90,
    Phi_D_Z90,
    Phi_minor_species,
    Phi_N_Z90,
    Phi_O_Z90,
    Phi_S_Z90,
    R_atm,
    R_env,
    V_reduction,
    phi_E,
    phi_kill,
    phi_RR,
    phiE_CP,
)
from isofate.options import IsocalcOptions
from isofate.species import DEFAULT_SPECIES, SYMBOLS, get_binary_diffusion_coeff
from isofate.system import Planet, Star, System
from isofate.utils import gravitational_acceleration


def isocalc(
    system: System,
    F0,
    time=5e9,
    isofate_species_abund: ArrayLike = (0, 0, 0, 0, 0, 0, 0),
    options: IsocalcOptions = IsocalcOptions(),
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
    #  - options: mode switches and tuning constants, fixed for the whole run - see
    #  IsocalcOptions for the full list (mechanism, rad_evol, melt_fraction_override, mu, eps,
    #  activity, flux_model, stellar_type, Rp_override, t_sat, step_fn, F_final, t_pms,
    #  pms_factor, n_steps, t0, rho_rcb, RR, thermal, beta, n_atmodeller, save_molecules,
    #  mantle_iron_dict, dynamic_phi)

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

    # Every other IsocalcOptions field is read-only for the whole run and referenced directly as
    # `options.<field>` below. `mu` and `mantle_iron_dict` are the two exceptions: both become
    # this loop's own time-evolving local state (like f_atm below) - `options.mu`/
    # `options.mantle_iron_dict` supply only their *initial* values and are never read again.
    mu = options.mu
    mantle_iron_dict = options.mantle_iron_dict

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, N_He, N_D, N_O, N_C, N_N, N_S = isofate_species_abund

    # Mstar, d, T, and Fp are confirmed fixed for the whole run (never reassigned anywhere
    # below), so they're read from `system` once, here. F0 is deliberately NOT derived from
    # System: it's a modeling choice (e.g. F0 = Fp*1e-3 "for M stars"), not a strict derived
    # quantity, so the caller must still supply it directly.
    planet: Planet = system.planet
    Mstar: Star = system.star.mass
    d: ArrayLike = system.semi_major_axis
    T: ArrayLike = system.equilibrium_temperature
    Fp: ArrayLike = system.insolation

    # Only planet.mass and planet.f_atm are ever read, and only here, once, to seed the
    # *initial* conditions: Mp never changes over the run, but f_atm is immediately reassigned
    # to a plain float and becomes this loop's own time-evolving local variable (see M_atm/f_atm
    # below) - it must never be read from `planet` (or `system`) again after this point.
    Mp = planet.mass
    f_atm = planet.f_atm

    ###_____Initialize physical values_____###

    b = binary_diffusion.H_He(T)
    radius_rocky = planet.rocky_radius  # [m]
    R_B = system.bondi_radius(mu, T)  # Bondi radius [m]
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
    interior_atmosphere = build_atmodeller(Mp)
    # Warm-starts each AtmodellerCoupler call from the previous call's converged solution instead
    # of solving cold every time - consecutive calls are a tiny physical perturbation apart, so
    # this drastically cuts the number of Newton iterations needed. None on the first call.
    atmod_initial_guess = None

    atmod_full_output = {}  # dictionary to store atmodeller full output

    if mantle_iron_dict:
        mantle_iron_dict["mantle_mass"] = 0.704665308539034 * Mp  # fraction from atmodeller
        mantle_iron_dict["mass_Fe"] = (
            mantle_iron_dict["mantle_mass"] * mantle_iron_dict["Fe_mass_fraction"]
        )
        mantle_iron_dict["mass_Fe2"] = mantle_iron_dict["mass_Fe"]
        mantle_iron_dict["X_Fe2"] = mantle_iron_dict["mass_Fe2"] / mantle_iron_dict["mass_Fe"]

    ### atmosphere
    M_atm0 = Mp * f_atm  # initial atmospheric mass [kg]
    M_atm = M_atm0
    # Atmospheric number of atoms per species [atoms], ordered per
    # isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as isofate_species_abund above.
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
    H2_a = np.zeros(n_tot)  # atmospheric H2 number array [molecules]
    H2O_a = np.zeros(n_tot)  # atmospheric H2O number array [molecules]
    O2_a = np.zeros(n_tot)  # atmospheric O2 number array [molecules]
    CO2_a = np.zeros(n_tot)  # atmospheric CO2 number array [molecules]
    CO_a = np.zeros(n_tot)  # atmospheric CO number array [molecules]
    CH4_a = np.zeros(n_tot)  # atmospheric CH4 number array [molecules]
    N2_a = np.zeros(n_tot)  # atmospheric N2 number array [molecules]
    S2_a = np.zeros(n_tot)  # atmospheric S2 number array [molecules]
    H2O4S_a = np.zeros(n_tot)  # atmospheric H2O4S gas number array [molecules]
    SO2_a = np.zeros(n_tot)  # atmospheric SO2 number array [molecules]
    fO2_a = np.zeros(n_tot)  # fugacity array [bar]
    H2_a_int = np.zeros(n_tot)  # mantle H2 number array [molecules]
    H2O_a_int = np.zeros(n_tot)  # mantle H2O number array [molecules]
    O2_a_int = np.zeros(n_tot)  # mantle O2 number array [molecules]
    CO2_a_int = np.zeros(n_tot)  # mantle CO2 number array [molecules]
    CO_a_int = np.zeros(n_tot)  # mantle CO number array [molecules]
    CH4_a_int = np.zeros(n_tot)  # mantle CH4 number array [molecules]
    N2_a_int = np.zeros(n_tot)  # mantle N2 number array [molecules]
    S2_a_int = np.zeros(n_tot)  # mantle S2 number array [molecules]
    H2O4S_a_int = np.zeros(n_tot)  # mantle H2O4S gas number array [molecules]
    SO2_a_int = np.zeros(n_tot)  # mantle SO2 number array [molecules]
    # Molar concentration history [ndim], ordered per isofate.species.SYMBOLS (H, He, D, O, C, N,
    # S) - same order as y/x - one column per species, one row per timestep.
    x_a = np.zeros((n_tot, 7))
    Phi_H_a = np.zeros(n_tot)  # H number flux array [atoms/s/m2]
    Phi_He_a = np.zeros(n_tot)  # He number flux array [atoms/s/m2]
    Phi_D_a = np.zeros(n_tot)  # D number flux array [atoms/s/m2]
    Phi_O_a = np.zeros(n_tot)  # O number flux array [atoms/s/m2]
    Phi_C_a = np.zeros(n_tot)  # C number flux array [atoms/s/m2]
    Phi_N_a = np.zeros(n_tot)  # N number flux array [atoms/s/m2]
    Phi_S_a = np.zeros(n_tot)  # S number flux array [atoms/s/m2]
    T_surf_analytic_a = np.zeros(n_tot)  # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot)  # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Loop through timesteps_____###

    for n in range(n_tot):
        ### Stop simulation when entire atmosphere is lost
        if M_atm <= 0 or np.sum(y) <= 0:
            Matm_a[n:] = 0  # M_atm #Matm_a[n-1]
            fatm_a[n:] = 0  # f_atm #fatm_a[n-1]
            Renv_a[n:] = 0  # radius_env #Renv_a[n-1]
            Rp_a[n:] = radius_rocky
            K = np.max(
                [V_reduction(Mp, Mstar, d, radius_rocky), 0.01]
            )  # grav potential reduction factor due to stellar tidal forces
            Vpot_a[n:] = K * const.G * Mp / radius_rocky
            phi_a[n:] = 0
            Mloss_a[n:] = 0

            y_a[n:] = 0
            y_a_int[n:] = 0
            H2_a[n:] = 0  # H2_a[n-1]
            H2O_a[n:] = 0  # H2O_a[n-1]
            O2_a[n:] = 0  # O2_a[n-1]
            CO2_a[n:] = 0  # CO2_a[n-1]
            CO_a[n:] = 0  # CO_a[n-1]
            CH4_a[n:] = 0  # CH4_a[n-1]
            N2_a[n:] = 0  # N2_a[n-1]
            S2_a[n:] = 0  # S2_a[n-1]
            H2_a_int[n:] = 0  # H2_a_int[n-1]
            H2O_a_int[n:] = 0  # H2O_a_int[n-1]
            O2_a_int[n:] = 0  # O2_a_int[n-1]
            CO2_a_int[n:] = 0  # CO2_a_int[n-1]
            CO_a_int[n:] = 0  # CO_a_int[n-1]
            CH4_a_int[n:] = 0  # CH4_a_int[n-1]
            N2_a_int[n:] = 0  # N2_a_int[n-1]
            S2_a_int[n:] = 0  # S2_a_int[n-1]
            H2O4S_a_int[n:] = 0  # H2O4S_a_int[n-1]
            SO2_a_int[n:] = 0  # SO2_a_int[n-1]
            x_a[n:] = x_a[n - 1]  # carry the last molar concentration forward
            Phi_H_a[n:] = 0
            Phi_He_a[n:] = 0
            Phi_D_a[n:] = 0
            Phi_O_a[n:] = 0
            Phi_C_a[n:] = 0
            Phi_N_a[n:] = 0
            Phi_S_a[n:] = 0

            # atmodeller full ouput for monte carlo runs
            if options.n_atmodeller != 0:
                # atmod_full_output = {}
                atmod_full_output["H2O_atm"] = np.nan
                atmod_full_output["H2O_mantle"] = np.nan
                atmod_full_output["H2_atm"] = np.nan
                atmod_full_output["H2_mantle"] = np.nan
                atmod_full_output["O2_atm"] = np.nan
                atmod_full_output["O2_mantle"] = np.nan
                atmod_full_output["CO_atm"] = np.nan
                atmod_full_output["CO_mantle"] = np.nan
                atmod_full_output["CO2_atm"] = np.nan
                atmod_full_output["CO2_mantle"] = np.nan
                atmod_full_output["CH4_atm"] = np.nan
                atmod_full_output["CH4_mantle"] = np.nan
                atmod_full_output["He_mantle"] = np.nan
                atmod_full_output["N2_atm"] = np.nan
                atmod_full_output["N2_mantle"] = np.nan
                atmod_full_output["S2_atm"] = np.nan
                atmod_full_output["S2_mantle"] = np.nan
                atmod_full_output["H2O4S_atm"] = np.nan
                atmod_full_output["H2O4S_mantle"] = np.nan
                atmod_full_output["SO2_atm"] = np.nan
                atmod_full_output["O2_fugacity"] = np.nan
                if options.save_molecules == True:
                    H2_a[n:] = 0
                    H2O_a[n:] = 0
                    O2_a[n:] = 0
                    CO2_a[n:] = 0
                    CO_a[n:] = 0
                    CH4_a[n:] = 0
                    N2_a[n:] = 0
                    S2_a[n:] = 0
                    H2O4S_a[n:] = 0
                    SO2_a[n:] = 0
                    H2_a_int[n:] = 0
                    H2O_a_int[n:] = 0
                    O2_a_int[n:] = 0
                    CO2_a_int[n:] = 0
                    CO_a_int[n:] = 0
                    CH4_a_int[n:] = 0
                    N2_a_int[n:] = 0
                    S2_a_int[n:] = 0
                    H2O4S_a_int[n:] = 0
                    SO2_a_int[n:] = 0
                    fO2_a[n:] = 0

            break

        # time-variable average atomic mass
        N_tot = np.sum(y)
        mu = np.dot(y, atomic_masses) / N_tot

        if options.rad_evol == False:
            radius_env = 0
            radius_atm = 0
            radius_p = radius_rocky
            if options.Rp_override != False:
                radius_rocky = options.Rp_override
                radius_env = 0
                radius_atm = 0
        else:
            radius_env = R_env(Mp, f_atm, Fp, t_a[n], options.thermal)
            radius_atm = R_atm(T, Mp, radius_rocky, radius_env, mu)
            radius_p = radius_rocky + radius_atm + radius_env
            # limits Rp to the min of Bondi/Hill/Lopez+Fortney radius; plain min() avoids numpy's
            # array-construction/dispatch overhead on a 3-scalar comparison run every timestep
            radius_p = min(R_B, R_H, radius_p)

        # grav potential reduction factor due to stellar tidal forces
        K = max(V_reduction(Mp, Mstar, d, radius_p), 0.01)
        Vpot = K * const.G * Mp / radius_p
        A = 4 * np.pi * radius_p**2

        # sets mass flux [kg/m2/s]
        if options.mechanism == "XUV":
            if options.RR == True:
                phi = min(
                    phi_RR(radius_p, Mp, T, t_a[n], F0, options),
                    phi_E(t_a[n], Vpot, d, F0, options),
                )
            else:
                phi = phi_E(t_a[n], Vpot, d, F0, options)
        elif options.mechanism == "CPML":
            phi = phiE_CP(T, Mp, options.rho_rcb, options.eps, Vpot, A, mu, radius_env)
        elif options.mechanism == "phi kill":
            phi = phi_kill(Mp * f_atm, radius_p, t - t_a[n])
        elif options.mechanism == "XUV+CPML":
            if options.RR == True:
                phi_XUV = min(
                    phi_RR(radius_p, Mp, T, t_a[n], F0, options),
                    phi_E(t_a[n], Vpot, d, F0, options),
                )
            else:
                phi_XUV = phi_E(t_a[n], Vpot, d, F0, options)
            phi = phi_XUV + phiE_CP(T, Mp, options.rho_rcb, options.eps, Vpot, A, mu, radius_env)

        mass_loss = phi * A * delta_t
        g = gravitational_acceleration(Mp, radius_p)
        H_H = const.R_gas * T / (const.M_H * g)  # H scale height [m]
        H_He = const.R_gas * T / (const.M_He * g)  # He scale height [m]
        H_D = const.R_gas * T / (const.M_D * g)  # D scale height [m]
        H_O = const.R_gas * T / (const.M_O * g)  # O scale height [m]
        H_C = const.R_gas * T / (const.M_C * g)  # C scale height [m]
        H_N = const.R_gas * T / (const.M_N * g)  # N scale height [m]
        H_S = const.R_gas * T / (const.M_S * g)  # S scale height [m]

        x = y / N_tot  # molar concentration per species [ndim]

        if options.dynamic_phi == False:
            if y[0] + y[1] == 0:
                X1 = 0
                X2 = 0
            else:
                X1 = y[0] / (y[0] + y[1])
                X2 = y[1] / (y[0] + y[1])
            MU = X1 * const.mu_H + X2 * const.mu_He
            Phi_H, phi_c = Phi_1(
                phi, b, H_H, H_He, const.mu_H, const.mu_He, X1, X2, MU, output=1
            )  # H number flux [atoms/s/m2]
            Phi_He = Phi_2(
                phi, b, H_H, H_He, const.mu_H, const.mu_He, X1, X2, MU
            )  # He number flux [atoms/s/m2]
            Phi_D = Phi_D_Z90(
                Phi_H, Phi_He, H_H, H_D, H_He, *y, T
            )  # D number flux [atoms/s/m2]
            Phi_O = Phi_O_Z90(
                Phi_H, Phi_He, H_H, H_O, H_He, *y, T
            )  # O number flux [atoms/s/m2]
            Phi_C = Phi_C_Z90(
                Phi_H, Phi_He, H_H, H_C, H_He, *y, T
            )  # C number flux [atoms/s/m2]
            Phi_N = Phi_N_Z90(
                Phi_H, Phi_He, H_H, H_N, H_He, *y, T
            )  # N number flux [atoms/s/m2]
            Phi_S = Phi_S_Z90(
                Phi_H, Phi_He, H_H, H_S, H_He, *y, T
            )  # S number flux [atoms/s/m2]

        elif options.dynamic_phi == True:
            N_values = y  # ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S)
            abundances_with_idx = [(i, N_values[i]) for i in range(7)]
            abundances_with_idx.sort(key=lambda item: item[1], reverse=True)

            most_abundant_idx = abundances_with_idx[0][0]
            second_most_abundant_idx = abundances_with_idx[1][0]

            # Get masses and scale heights for the two most abundant species
            scale_heights = [H_H, H_He, H_D, H_O, H_C, H_N, H_S]

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

            b = get_binary_diffusion_coeff(light_name, heavy_name, T)

            # Calculate escape fluxes for the two dominant species
            Phi_1_calc, phi_c = Phi_1(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU, output=1)
            Phi_2_calc = Phi_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)

            # Assign fluxes to correct species based on light/heavy dominant indices
            if light_dominant_idx == 0:  # H is species 1 (lighter dominant)
                Phi_H = Phi_1_calc
            elif light_dominant_idx == 1:  # He is species 1
                Phi_He = Phi_1_calc
            elif light_dominant_idx == 2:  # D is species 1
                Phi_D = Phi_1_calc
            elif light_dominant_idx == 3:  # O is species 1
                Phi_O = Phi_1_calc
            elif light_dominant_idx == 4:  # C is species 1
                Phi_C = Phi_1_calc
            elif light_dominant_idx == 5:  # N is species 1
                Phi_N = Phi_1_calc
            elif light_dominant_idx == 6:  # S is species 1
                Phi_S = Phi_1_calc

            if heavy_dominant_idx == 0:  # H is species 2 (heavier dominant)
                Phi_H = Phi_2_calc
            elif heavy_dominant_idx == 1:  # He is species 2
                Phi_He = Phi_2_calc
            elif heavy_dominant_idx == 2:  # D is species 2
                Phi_D = Phi_2_calc
            elif heavy_dominant_idx == 3:  # O is species 2
                Phi_O = Phi_2_calc
            elif heavy_dominant_idx == 4:  # C is species 2
                Phi_C = Phi_2_calc
            elif heavy_dominant_idx == 5:  # N is species 2
                Phi_N = Phi_2_calc
            elif heavy_dominant_idx == 6:  # S is species 2
                Phi_S = Phi_2_calc

            # Calculate fluxes for remaining species using corrected generalized function
            for i in range(7):
                if i != light_dominant_idx and i != heavy_dominant_idx:
                    flux = Phi_minor_species(
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
                    # Assign to correct species
                    if i == 0:  # H
                        Phi_H = flux
                    elif i == 1:  # He
                        Phi_He = flux
                    elif i == 2:  # D
                        Phi_D = flux
                    elif i == 3:  # O
                        Phi_O = flux
                    elif i == 4:  # C
                        Phi_C = flux
                    elif i == 5:  # N
                        Phi_N = flux
                    elif i == 6:  # S
                        Phi_S = flux

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
        Phi_H_a[n] = Phi_H
        Phi_He_a[n] = Phi_He
        Phi_D_a[n] = Phi_D
        Phi_O_a[n] = Phi_O
        Phi_C_a[n] = Phi_C
        Phi_N_a[n] = Phi_N
        Phi_S_a[n] = Phi_S

        ##### run atmodeller ######
        if options.n_atmodeller != 0:  # save final molecular abundances on last time step
            if n == options.n_steps - 1:
                # atmod_full_output = {}
                atmod_sol = AtmodellerCoupler(
                    T,
                    Mp,
                    radius_p,
                    mu,
                    options.melt_fraction_override,
                    mantle_iron_dict,
                    *aggregate_D_into_H(y),
                    *aggregate_D_into_H(isofate_species_abund_int),
                    interior_atmosphere,
                    radius_rocky=radius_rocky,
                    initial_guess=atmod_initial_guess,
                )[1]
                atmod_full_output["H2O_atm"] = atmod_sol["H2O_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["H2O_mantle"] = atmod_sol["H2O_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["H2_atm"] = atmod_sol["H2_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["H2_mantle"] = atmod_sol["H2_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["O2_atm"] = atmod_sol["O2_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["O2_mantle"] = 0.0  # O2 has no solubility model / melt reservoir
                atmod_full_output["CO_atm"] = atmod_sol["CO_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["CO_mantle"] = atmod_sol["CO_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["CO2_atm"] = atmod_sol["CO2_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["CO2_mantle"] = atmod_sol["CO2_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["CH4_atm"] = atmod_sol["CH4_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["CH4_mantle"] = atmod_sol["CH4_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["N2_atm"] = atmod_sol["N2_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["N2_mantle"] = atmod_sol["N2_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["S2_atm"] = atmod_sol["S2_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["S2_mantle"] = atmod_sol["S2_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["H2O4S_atm"] = atmod_sol["H2O4S_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["H2O4S_mantle"] = (
                    0.0  # H2O4S has no solubility model / melt reservoir
                )
                atmod_full_output["SO2_atm"] = atmod_sol["O2S_g"]["gas"]["number_moles"][0][0]
                atmod_full_output["He_mantle"] = atmod_sol["He_d"]["silicate_melt"][
                    "number_moles"
                ][0][0]
                atmod_full_output["O2_fugacity"] = atmod_sol["O2_g"]["gas"]["activity"][0][0]
                atmod_full_output["log10dIW_1_bar"] = atmod_sol["gas"]["phase"]["log10dIW_1_bar"][
                    0
                ][0]
            if n % options.n_atmodeller == 0:  # run atmodeller every n_atmodeller steps.
                atmod_results, atmod_full, mantle_iron_dict, atmod_initial_guess = (
                    AtmodellerCoupler(
                        T,
                        Mp,
                        radius_p,
                        mu,
                        options.melt_fraction_override,
                        mantle_iron_dict,
                        *aggregate_D_into_H(y),
                        *aggregate_D_into_H(isofate_species_abund_int),
                        interior_atmosphere,
                        radius_rocky=radius_rocky,
                        initial_guess=atmod_initial_guess,
                        # Species-level diagnostics (atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
                        # are only read below when save_molecules is True; otherwise the narrow
                        # extraction (element number_moles + gas mass only) is all this loop needs.
                        full_output=options.save_molecules,
                    )
                )
                isofate_species_abund_int[0] = atmod_results["N_H_int"] * (1 - X_DH)
                isofate_species_abund_int[2] = atmod_results["N_H_int"] * X_DH
                isofate_species_abund_int[1] = atmod_results["N_He_int"]
                isofate_species_abund_int[3] = atmod_results["N_O_int"]
                isofate_species_abund_int[4] = atmod_results["N_C_int"]
                isofate_species_abund_int[5] = atmod_results["N_N_int"]
                isofate_species_abund_int[6] = atmod_results["N_S_int"]
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
                M_atm = atmod_results["M_atm"]
                T_surf_analytic = atmod_results["T_surface"]
                T_surf_atmod = atmod_results["T_surface_atmod"]
                if options.save_molecules == True:
                    H2_a[n] = atmod_full["H2_g"]["gas"]["number_moles"][0][0]
                    H2O_a[n] = atmod_full["H2O_g"]["gas"]["number_moles"][0][0]
                    O2_a[n] = atmod_full["O2_g"]["gas"]["number_moles"][0][0]
                    CO2_a[n] = atmod_full["CO2_g"]["gas"]["number_moles"][0][0]
                    CO_a[n] = atmod_full["CO_g"]["gas"]["number_moles"][0][0]
                    CH4_a[n] = atmod_full["CH4_g"]["gas"]["number_moles"][0][0]
                    N2_a[n] = atmod_full["N2_g"]["gas"]["number_moles"][0][0]
                    S2_a[n] = atmod_full["S2_g"]["gas"]["number_moles"][0][0]
                    H2O4S_a[n] = atmod_full["H2O4S_g"]["gas"]["number_moles"][0][0]
                    SO2_a[n] = atmod_full["O2S_g"]["gas"]["number_moles"][0][0]
                    H2_a_int[n] = atmod_full["H2_d"]["silicate_melt"]["number_moles"][0][0]
                    H2O_a_int[n] = atmod_full["H2O_d"]["silicate_melt"]["number_moles"][0][0]
                    O2_a_int[n] = 0.0  # O2 has no solubility model / melt reservoir
                    CO2_a_int[n] = atmod_full["CO2_d"]["silicate_melt"]["number_moles"][0][0]
                    CO_a_int[n] = atmod_full["CO_d"]["silicate_melt"]["number_moles"][0][0]
                    CH4_a_int[n] = atmod_full["CH4_d"]["silicate_melt"]["number_moles"][0][0]
                    N2_a_int[n] = atmod_full["N2_d"]["silicate_melt"]["number_moles"][0][0]
                    S2_a_int[n] = atmod_full["S2_d"]["silicate_melt"]["number_moles"][0][0]
                    H2O4S_a_int[n] = 0.0  # H2O4S has no solubility model / melt reservoir
                    SO2_a_int[n] = 0.0  # SO2 has no solubility model / melt reservoir
                    fO2_a[n] = atmod_full["O2_g"]["gas"]["activity"][0][0]
            else:
                H2_a[n] = H2_a[n - 1]
                H2O_a[n] = H2O_a[n - 1]
                O2_a[n] = O2_a[n - 1]
                CO2_a[n] = CO2_a[n - 1]
                CO_a[n] = CO_a[n - 1]
                CH4_a[n] = CH4_a[n - 1]
                N2_a[n] = N2_a[n - 1]
                S2_a[n] = S2_a[n - 1]
                H2O4S_a[n] = H2O4S_a[n - 1]
                SO2_a[n] = SO2_a[n - 1]
                H2_a_int[n] = H2_a_int[n - 1]
                H2O_a_int[n] = H2O_a_int[n - 1]
                O2_a_int[n] = O2_a_int[n - 1]
                CO2_a_int[n] = CO2_a_int[n - 1]
                CO_a_int[n] = CO_a_int[n - 1]
                CH4_a_int[n] = CH4_a_int[n - 1]
                N2_a_int[n] = N2_a_int[n - 1]
                S2_a_int[n] = S2_a_int[n - 1]
                H2O4S_a_int[n] = H2O4S_a_int[n - 1]
                SO2_a_int[n] = SO2_a_int[n - 1]
                fO2_a[n] = fO2_a[n - 1]
        T_surf_analytic_a[n] = T_surf_analytic
        T_surf_atmod_a[n] = T_surf_atmod

        # needed to allow D and H to outgas from mantle
        if y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2] != 0:
            X_DH = (y[2] + isofate_species_abund_int[2]) / (
                y[0] + y[2] + isofate_species_abund_int[0] + isofate_species_abund_int[2]
            )  # assumes D/H is in equilibrium between interior and atmosphere

        # advance to next step
        Phi = np.array([Phi_H, Phi_He, Phi_D, Phi_O, Phi_C, Phi_N, Phi_S])
        y_loss = Phi * A * delta_t
        # M_atm -= mass_loss # comes from phi*A*delta_t
        M_atm -= np.dot(y_loss, atomic_masses)
        f_atm = M_atm / Mp
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
        "Phi_H": Phi_H_a,
        "Phi_He": Phi_He_a,
        "Phi_D": Phi_D_a,
        "Phi_O": Phi_O_a,
        "Phi_C": Phi_C_a,
        "Phi_N": Phi_N_a,
        "Phi_S": Phi_S_a,
        "T_surf_analytic": T_surf_analytic_a,
        "T_surf_atmod": T_surf_atmod_a,
    }
    if options.save_molecules == True:
        solutions["n_H2_a"] = H2_a
        solutions["n_H2O_a"] = H2O_a
        solutions["n_O2_a"] = O2_a
        solutions["n_CO2_a"] = CO2_a
        solutions["n_CO_a"] = CO_a
        solutions["n_CH4_a"] = CH4_a
        solutions["n_N2_a"] = N2_a
        solutions["n_S2_a"] = S2_a
        solutions["n_H2O4S_a"] = H2O4S_a
        solutions["n_SO2_a"] = SO2_a
        solutions["n_H2_a_int"] = H2_a_int
        solutions["n_H2O_a_int"] = H2O_a_int
        solutions["n_O2_a_int"] = O2_a_int
        solutions["n_CO2_a_int"] = CO2_a_int
        solutions["n_CO_a_int"] = CO_a_int
        solutions["n_CH4_a_int"] = CH4_a_int
        solutions["n_N2_a_int"] = N2_a_int
        solutions["n_S2_a_int"] = S2_a_int
        solutions["n_H2O4S_a_int"] = H2O4S_a_int
        solutions["n_SO2_a_int"] = SO2_a_int
        solutions["fO2_a"] = fO2_a
    if options.n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions
