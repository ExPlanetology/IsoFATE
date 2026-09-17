# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main IsoFATE script for coupled model."""

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from isofate.atmodeller_coupler import AtmodellerCoupler, build_atmodeller
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
    get_binary_diffusion_coeff,
    phi_E,
    phi_kill,
    phi_RR,
    phiE_CP,
)
from isofate.species import ATOMIC_MASSES, SYMBOLS
from isofate.system import Planet, Star, System


class IsocalcOptions(eqx.Module):
    """Mode switches and tuning constants for `isocalc`, fixed for the whole run.

    These are numerical/modeling choices, as opposed to `isocalc`'s other arguments (`system`,
    `F0`, `time`, and the initial `N_x` abundances), which describe the actual physical
    initial-value problem being solved and so stay direct `isocalc` arguments.

    Args:
        mechanism: One of 'XUV', 'XUV+RR', 'CPML', 'XUV+CPML', 'fix phi subcritical',
            'fix phi supercritical', 'phi kill'.
        rad_evol: Set to False to fix the planet radius at the rocky radius.
        melt_fraction_override: Fixed mantle melt fraction; if False, it is instead calculated
            from Mp and T_surface.
        mu: Average atmospheric particle mass [kg]; default is H/He solar composition. Only the
            initial value - `isocalc` recomputes it every step from the evolving abundances.
        eps: Heat transfer efficiency [ndim].
        activity: For the `Fxuv_hazmat` flux model (semi-empirical MUSCLES survey data): 'low'
            (lower quartile), 'medium' (median), or 'high' (upper quartile).
        flux_model: 'power law' for the analytic power law, 'phoenix' for `Fxuv_hazmat`,
            'Johnstone' for `Fxuv_Johnstone`.
        stellar_type: 'M1', 'K5', or 'G5'; only used when `flux_model == 'Johnstone'`.
        Rp_override: Scalar planet radius [m] to manually fix a constant radius (radius will not
            evolve); False to disable.
        t_sat: XUV power-law saturation time [yr]; 5e8 matches semi-empirical MUSCLES data.
        step_fn: Toggles a step-function XUV flux evolution (drops to `F_final` at `t_pms`).
        F_final: Final relative XUV flux level (of F0) once `step_fn` engages.
        t_pms: Pre-main-sequence phase duration [yr].
        pms_factor: XUV enhancement factor applied during the pre-main-sequence phase.
        n_steps: Number of timesteps; convergence occurs at 1e6.
        t0: Simulation start time [yr].
        rho_rcb: Gas density at the RCB in the CPML phi equation [kg/m3].
        RR: Toggles the radiation-recombination effect (Ly-alpha cooling; Murray-Clay et al 2009).
        thermal: Toggles planet radius contraction in the Lopez/Fortney equations (False removes
            the age term).
        beta: Exponent in the Fxuv power-law function; determines the rate of XUV decrease.
            -1.23 is consistent with MUSCLES data.
        n_atmodeller: Interval of timesteps between each Atmodeller call.
        save_molecules: Save molecular abundances at every timestep (True) or only the final
            abundances (False).
        mantle_iron_dict: Allows Fe in the mantle to react with O2. `['type']="dynamic"` reacts
            only molten mantle Fe; `['type']="static"` reacts all mantle Fe; also specify
            `['Fe_mass_fraction']`. False disables this.
        dynamic_phi: Toggle dynamic phi calculation based on the most abundant species (True) or
            static phi calculation (False).
    """

    mechanism: str = "XUV"
    rad_evol: bool = True
    melt_fraction_override: ArrayLike | bool = False
    mu: ArrayLike = const.mu_solar
    eps: ArrayLike = 0.15
    activity: str = "medium"
    flux_model: str = "power law"
    stellar_type: str = "M1"
    Rp_override: ArrayLike | bool = False
    t_sat: ArrayLike = 5e8
    step_fn: bool = False
    F_final: ArrayLike = 0
    t_pms: ArrayLike = 0
    pms_factor: ArrayLike = 1e2
    n_steps: int = int(1e5)
    t0: ArrayLike = 1e6
    rho_rcb: ArrayLike = 1.0
    RR: bool = True
    thermal: bool = True
    beta: ArrayLike = -1.23
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    mantle_iron_dict: dict | bool = False
    dynamic_phi: bool = False


def isocalc(
    system: System,
    F0,
    time=5e9,
    N_H=0,
    N_He=0,
    N_D=0,
    N_O=0,
    N_C=0,
    N_N=0,
    N_S=0,
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
    #  - N_x: initial abundance for species x [atoms]
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

    # Unpacked once into plain local variables so the rest of this (very long) function can keep
    # referring to them by their old bare names; `mu` and `mantle_iron_dict` are then immediately
    # treated as this loop's own time-evolving local state, same as f_atm below - never read back
    # from `options` again after this point.
    mechanism = options.mechanism
    rad_evol = options.rad_evol
    melt_fraction_override = options.melt_fraction_override
    mu = options.mu
    eps = options.eps
    activity = options.activity
    flux_model = options.flux_model
    stellar_type = options.stellar_type
    Rp_override = options.Rp_override
    t_sat = options.t_sat
    step_fn = options.step_fn
    F_final = options.F_final
    t_pms = options.t_pms
    pms_factor = options.pms_factor
    n_steps = options.n_steps
    t0 = options.t0
    rho_rcb = options.rho_rcb
    RR = options.RR
    thermal = options.thermal
    beta = options.beta
    n_atmodeller = options.n_atmodeller
    save_molecules = options.save_molecules
    mantle_iron_dict = options.mantle_iron_dict
    dynamic_phi = options.dynamic_phi

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

    n_tot = n_steps  # timesteps
    t0 = t0 / const.s2yr  # simulation start time [s]
    t = time / const.s2yr - t0  # total simulation time [s]
    delta_t = t / n_tot  # timestep [s]

    ###_____Set initial values____###

    atomic_masses = ATOMIC_MASSES
    species_names = SYMBOLS

    ### atmodeller interior
    N_H_int = 0
    N_He_int = 0
    N_D_int = 0
    N_O_int = 0
    N_C_int = 0
    N_N_int = 0
    N_S_int = 0
    if n_atmodeller == 0:
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
    y1 = 1 * N_H  # H number [atoms]
    y2 = 1 * N_He  # He number [atoms]
    y3 = 1 * N_D  # D number [atoms]
    y4 = 1 * N_O  # O number [atoms]
    y5 = 1 * N_C  # O number [atoms]
    y6 = 1 * N_N  # N number [atoms]
    y7 = 1 * N_S  # S number [atoms]
    ###_____Initialize arrays_____###

    t_a = delta_t * np.linspace(1, n_tot + 1, n_tot) + t0  # time array [s]
    phi_a = np.zeros(n_tot)  # mass flux array [kg/s/m2]
    phic_a = np.zeros(n_tot)  # critical mass flux array [kg/s/m2]
    Rp_a = np.zeros(n_tot)  # total radius, diagnostic [m]
    Renv_a = np.zeros(n_tot)  # envelope radius [m]
    Matm_a = np.zeros(n_tot)  # atmospheric mass [kg]
    fatm_a = np.zeros(n_tot)  # atm mass fraction [ndim]
    Vpot_a = np.zeros(n_tot)  # grav potential, diagnostic [J/kg]
    Mloss_a = np.zeros(n_tot)  # mass lost per timestep [kg]

    y1_a = np.zeros(n_tot)  # H number array [atoms]
    y2_a = np.zeros(n_tot)  # He number array [atoms]
    y3_a = np.zeros(n_tot)  # D number array [atoms]
    y4_a = np.zeros(n_tot)  # O number array [atoms]
    y5_a = np.zeros(n_tot)  # C number array [atoms]
    y6_a = np.zeros(n_tot)  # N number array [atoms]
    y7_a = np.zeros(n_tot)  # S number array [atoms]
    y1_a_int = np.zeros(n_tot)  # mantle H number array [atoms]
    y2_a_int = np.zeros(n_tot)  # mantle He number array [atoms]
    y3_a_int = np.zeros(n_tot)  # mantle D number array [atoms]
    y4_a_int = np.zeros(n_tot)  # mantle O number array [atoms]
    y5_a_int = np.zeros(n_tot)  # mantle C number array [atoms]
    y6_a_int = np.zeros(n_tot)  # mantle N number array [atoms]
    y7_a_int = np.zeros(n_tot)  # mantle S number array [atoms]
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
    x1_a = np.zeros(n_tot)  # H molar concentration array [ndim]
    x2_a = np.zeros(n_tot)  # He molar concentration array [ndim]
    x3_a = np.zeros(n_tot)  # D molar concentration array [ndim]
    x4_a = np.zeros(n_tot)  # O molar concentration array [ndim]
    x5_a = np.zeros(n_tot)  # C molar concentration array [ndim]
    x6_a = np.zeros(n_tot)  # N molar concentration array [ndim]
    x7_a = np.zeros(n_tot)  # S molar concentration array [ndim]
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
        if M_atm <= 0 or y1 + y2 + y3 + y4 + y5 + y6 + y7 <= 0:
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

            y1_a[n:] = 0  # y1 #y1_a[n-1]
            y2_a[n:] = 0  # y2 #y2_a[n-1]
            y3_a[n:] = 0  # y3 #y3_a[n-1]
            y4_a[n:] = 0  # y4 #y4_a[n-1]
            y5_a[n:] = 0  # y5 #y5_a[n-1]
            y6_a[n:] = 0  # y6 #y6_a[n-1]
            y7_a[n:] = 0  # y7 #y7_a[n-1]
            y1_a_int[n:] = 0  # y1_int #y1_a_int[n-1]
            y2_a_int[n:] = 0  # y2_int #y2_a_int[n-1]
            y3_a_int[n:] = 0  # y3_int #y3_a_int[n-1]
            y4_a_int[n:] = 0  # y4_int #y4_a_int[n-1]
            y5_a_int[n:] = 0  # y5_int #y5_a_int[n-1]
            y6_a_int[n:] = 0  # y6_int #y6_a_int[n-1]
            y7_a_int[n:] = 0  # y7_int #y7_a_int[n-1]
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
            x1_a[n:] = x1_a[
                n - 1
            ]  # x1_a[max(np.nonzero(x1_a)[0])] # get last non-zero value in array
            x2_a[n:] = x2_a[n - 1]  # x2_a[max(np.nonzero(x2_a)[0])]
            x3_a[n:] = x3_a[n - 1]  # x3_a[max(np.nonzero(x3_a)[0])]
            x4_a[n:] = x4_a[n - 1]  # x4_a[max(np.nonzero(x4_a)[0])]
            x5_a[n:] = x5_a[n - 1]  # x5_a[max(np.nonzero(x5_a)[0])]
            x6_a[n:] = x6_a[n - 1]  # x6_a[max(np.nonzero(x6_a)[0])]
            x7_a[n:] = x7_a[n - 1]  # x7_a[max(np.nonzero(x7_a)[0])]
            Phi_H_a[n:] = 0
            Phi_He_a[n:] = 0
            Phi_D_a[n:] = 0
            Phi_O_a[n:] = 0
            Phi_C_a[n:] = 0
            Phi_N_a[n:] = 0
            Phi_S_a[n:] = 0

            # atmodeller full ouput for monte carlo runs
            if n_atmodeller != 0:
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
                if save_molecules == True:
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
        N_tot = y1 + y2 + y3 + y4 + y5 + y6 + y7
        mu = (
            y1 * const.mu_H
            + y2 * const.mu_He
            + y3 * const.mu_D
            + y4 * const.mu_O
            + y5 * const.mu_C
            + y6 * const.mu_N
            + y7 * const.mu_S
        ) / N_tot

        if rad_evol == False:
            radius_env = 0
            radius_atm = 0
            radius_p = radius_rocky
            if Rp_override != False:
                radius_rocky = Rp_override
                radius_env = 0
                radius_atm = 0
        else:
            radius_env = R_env(Mp, f_atm, Fp, t_a[n], thermal)
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
        if mechanism == "XUV":
            if RR == True:
                phi = min(
                    phi_RR(
                        radius_p,
                        Mp,
                        T,
                        t_a[n],
                        F0,
                        t0 * const.s2yr,
                        t_sat,
                        beta,
                        step_fn,
                        F_final,
                        t_pms,
                        pms_factor,
                    ),
                    phi_E(
                        t_a[n],
                        eps,
                        Vpot,
                        d,
                        F0,
                        t0 * const.s2yr,
                        t_sat,
                        beta,
                        activity,
                        flux_model,
                        stellar_type,
                        step_fn,
                        F_final,
                        t_pms,
                        pms_factor,
                    ),
                )
            else:
                phi = phi_E(
                    t_a[n],
                    eps,
                    Vpot,
                    d,
                    F0,
                    t0 * const.s2yr,
                    t_sat,
                    beta,
                    activity,
                    flux_model,
                    stellar_type,
                    step_fn,
                    F_final,
                    t_pms,
                    pms_factor,
                )
        elif mechanism == "CPML":
            phi = phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
        elif mechanism == "phi kill":
            phi = phi_kill(Mp * f_atm, radius_p, t - t_a[n])
        elif mechanism == "XUV+CPML":
            if RR == True:
                phi_XUV = min(
                    phi_RR(
                        radius_p,
                        Mp,
                        T,
                        t_a[n],
                        F0,
                        t0 * const.s2yr,
                        t_sat,
                        beta,
                        step_fn,
                        F_final,
                        t_pms,
                        pms_factor,
                    ),
                    phi_E(
                        t_a[n],
                        eps,
                        Vpot,
                        d,
                        F0,
                        t0 * const.s2yr,
                        t_sat,
                        beta,
                        activity,
                        flux_model,
                        stellar_type,
                        step_fn,
                        F_final,
                        t_pms,
                        pms_factor,
                    ),
                )
            else:
                phi_XUV = phi_E(
                    t_a[n],
                    eps,
                    Vpot,
                    d,
                    F0,
                    t0 * const.s2yr,
                    t_sat,
                    beta,
                    activity,
                    flux_model,
                    stellar_type,
                    step_fn,
                    F_final,
                    t_pms,
                    pms_factor,
                )
            phi = phi_XUV + phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)

        mass_loss = phi * A * delta_t
        g = const.G * Mp / radius_p**2
        H_H = const.R_gas * T / (const.M_H * g)  # H scale height [m]
        H_He = const.R_gas * T / (const.M_He * g)  # He scale height [m]
        H_D = const.R_gas * T / (const.M_D * g)  # D scale height [m]
        H_O = const.R_gas * T / (const.M_O * g)  # O scale height [m]
        H_C = const.R_gas * T / (const.M_C * g)  # C scale height [m]
        H_N = const.R_gas * T / (const.M_N * g)  # N scale height [m]
        H_S = const.R_gas * T / (const.M_S * g)  # S scale height [m]

        x1 = y1 / N_tot
        x2 = y2 / N_tot
        x3 = y3 / N_tot
        x4 = y4 / N_tot
        x5 = y5 / N_tot
        x6 = y6 / N_tot
        x7 = y7 / N_tot

        if dynamic_phi == False:
            if y1 + y2 == 0:
                X1 = 0
                X2 = 0
            else:
                X1 = y1 / (y1 + y2)
                X2 = y2 / (y1 + y2)
            MU = X1 * const.mu_H + X2 * const.mu_He
            Phi_H, phi_c = Phi_1(
                phi, b, H_H, H_He, const.mu_H, const.mu_He, X1, X2, MU, output=1
            )  # H number flux [atoms/s/m2]
            Phi_He = Phi_2(
                phi, b, H_H, H_He, const.mu_H, const.mu_He, X1, X2, MU
            )  # He number flux [atoms/s/m2]
            Phi_D = Phi_D_Z90(
                Phi_H, Phi_He, H_H, H_D, H_He, y1, y2, y3, y4, y5, y6, y7, T
            )  # D number flux [atoms/s/m2]
            Phi_O = Phi_O_Z90(
                Phi_H, Phi_He, H_H, H_O, H_He, y1, y2, y3, y4, y5, y6, y7, T
            )  # O number flux [atoms/s/m2]
            Phi_C = Phi_C_Z90(
                Phi_H, Phi_He, H_H, H_C, H_He, y1, y2, y3, y4, y5, y6, y7, T
            )  # C number flux [atoms/s/m2]
            Phi_N = Phi_N_Z90(
                Phi_H, Phi_He, H_H, H_N, H_He, y1, y2, y3, y4, y5, y6, y7, T
            )  # N number flux [atoms/s/m2]
            Phi_S = Phi_S_Z90(
                Phi_H, Phi_He, H_H, H_S, H_He, y1, y2, y3, y4, y5, y6, y7, T
            )  # S number flux [atoms/s/m2]

        elif dynamic_phi == True:
            N_values = [y1, y2, y3, y4, y5, y6, y7]  # [H, He, D, O, C, N, S]
            abundances_with_idx = [(i, N_values[i]) for i in range(7)]
            abundances_with_idx.sort(key=lambda x: x[1], reverse=True)

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

        y1_a[n] = y1
        y2_a[n] = y2
        y3_a[n] = y3
        y4_a[n] = y4
        y5_a[n] = y5
        y6_a[n] = y6
        y7_a[n] = y7
        y1_a_int[n] = N_H_int
        y2_a_int[n] = N_He_int
        y3_a_int[n] = N_D_int
        y4_a_int[n] = N_O_int
        y5_a_int[n] = N_C_int
        y6_a_int[n] = N_N_int
        y7_a_int[n] = N_S_int
        x1_a[n] = x1
        x2_a[n] = x2
        x3_a[n] = x3
        x4_a[n] = x4
        x5_a[n] = x5
        x6_a[n] = x6
        x7_a[n] = x7
        Phi_H_a[n] = Phi_H
        Phi_He_a[n] = Phi_He
        Phi_D_a[n] = Phi_D
        Phi_O_a[n] = Phi_O
        Phi_C_a[n] = Phi_C
        Phi_N_a[n] = Phi_N
        Phi_S_a[n] = Phi_S

        ##### run atmodeller ######
        if n_atmodeller != 0:  # save final molecular abundances on last time step
            if n == n_steps - 1:
                # atmod_full_output = {}
                atmod_sol = AtmodellerCoupler(
                    T,
                    Mp,
                    radius_p,
                    mu,
                    melt_fraction_override,
                    mantle_iron_dict,
                    y1 + y3,
                    y2,
                    y4,
                    y5,
                    y6,
                    y7,
                    N_H_int + N_D_int,
                    N_He_int,
                    N_O_int,
                    N_C_int,
                    N_N_int,
                    N_S_int,
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
            if n % n_atmodeller == 0:  # run atmodeller every n_atmodeller steps.
                atmod_results, atmod_full, mantle_iron_dict, atmod_initial_guess = (
                    AtmodellerCoupler(
                        T,
                        Mp,
                        radius_p,
                        mu,
                        melt_fraction_override,
                        mantle_iron_dict,
                        y1 + y3,
                        y2,
                        y4,
                        y5,
                        y6,
                        y7,
                        N_H_int + N_D_int,
                        N_He_int,
                        N_O_int,
                        N_C_int,
                        N_N_int,
                        N_S_int,
                        interior_atmosphere,
                        radius_rocky=radius_rocky,
                        initial_guess=atmod_initial_guess,
                        # Species-level diagnostics (atmod_full["H2_g"]["gas"][...], O2 activity, etc.)
                        # are only read below when save_molecules is True; otherwise the narrow
                        # extraction (element number_moles + gas mass only) is all this loop needs.
                        full_output=save_molecules,
                    )
                )
                N_H_int = atmod_results["N_H_int"] * (1 - X_DH)
                N_D_int = atmod_results["N_H_int"] * X_DH
                N_He_int = atmod_results["N_He_int"]
                N_O_int = atmod_results["N_O_int"]
                N_C_int = atmod_results["N_C_int"]
                N_N_int = atmod_results["N_N_int"]
                N_S_int = atmod_results["N_S_int"]
                if atmod_results["N_H_atm"] == 0:
                    y1 = 0
                    y3 = 0
                else:
                    Y3 = X_DH * atmod_results["N_H_atm"]
                    Y1 = (1 - X_DH) * atmod_results["N_H_atm"]
                    y1 = Y1
                    y3 = Y3
                y2 = atmod_results["N_He_atm"]
                y4 = atmod_results["N_O_atm"]
                y5 = atmod_results["N_C_atm"]
                y6 = atmod_results["N_N_atm"]
                y7 = atmod_results["N_S_atm"]
                M_atm = atmod_results["M_atm"]
                T_surf_analytic = atmod_results["T_surface"]
                T_surf_atmod = atmod_results["T_surface_atmod"]
                if save_molecules == True:
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

        if y1 + y3 + N_H_int + N_D_int != 0:  # needed to allow D and H to outgas from mantle
            X_DH = (y3 + N_D_int) / (
                y1 + y3 + N_H_int + N_D_int
            )  # assumes D/H is in equilibrium between interior and atmosphere

        # advance to next step
        y1_loss = Phi_H * A * delta_t
        y2_loss = Phi_He * A * delta_t
        y3_loss = Phi_D * A * delta_t
        y4_loss = Phi_O * A * delta_t
        y5_loss = Phi_C * A * delta_t
        y6_loss = Phi_N * A * delta_t
        y7_loss = Phi_S * A * delta_t
        # M_atm -= mass_loss # comes from phi*A*delta_t
        M_atm -= (
            y1_loss * const.mu_H
            + y2_loss * const.mu_He
            + y3_loss * const.mu_D
            + y4_loss * const.mu_O
            + y5_loss * const.mu_C
            + y6_loss * const.mu_N
            + y7_loss * const.mu_S
        )
        f_atm = M_atm / Mp
        y1 -= y1_loss
        y2 -= y2_loss
        y3 -= y3_loss
        y4 -= y4_loss
        y5 -= y5_loss
        y6 -= y6_loss
        y7 -= y7_loss
        y1 = max(y1, 0)
        y2 = max(y2, 0)
        y3 = max(y3, 0)
        y4 = max(y4, 0)
        y5 = max(y5, 0)
        y6 = max(y6, 0)
        y7 = max(y7, 0)

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
        "N_H": y1_a,
        "N_He": y2_a,
        "N_D": y3_a,
        "N_O": y4_a,
        "N_C": y5_a,
        "N_N": y6_a,
        "N_S": y7_a,
        "N_H_int": y1_a_int,
        "N_He_int": y2_a_int,
        "N_D_int": y3_a_int,
        "N_O_int": y4_a_int,
        "N_C_int": y5_a_int,
        "N_N_int": y6_a_int,
        "N_S_int": y7_a_int,
        "x1": x1_a,
        "x2": x2_a,
        "x3": x3_a,
        "x4": x4_a,
        "x5": x5_a,
        "x6": x6_a,
        "x7": x7_a,
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
    if save_molecules == True:
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
    if n_atmodeller != 0:
        solutions["atmodeller_final"] = atmod_full_output

    return solutions
