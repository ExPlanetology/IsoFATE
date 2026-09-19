# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main IsoFATE script for coupled model."""

import diffrax
import equinox as eqx
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
from isofate.constants import const
from isofate.escape import EscapeMechanism, EscapeState, XUVEscape
from isofate.escape_core import EscapeNumberFlux, Phi_1_2, Phi_minor_species
from isofate.isofunks import R_atm, R_env
from isofate.mantle_iron import MantleIronState
from isofate.options import IsocalcOptions
from isofate.species import DEFAULT_SPECIES, SYMBOLS
from isofate.system import Planet, System
from isofate.utils import gravitational_acceleration


def isocalc(
    system: System,
    F0,
    time=5e9,
    isofate_species_abund: ArrayLike = (0, 0, 0, 0, 0, 0, 0),
    options: IsocalcOptions = IsocalcOptions(),
    escape: EscapeMechanism = XUVEscape(),
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

    # Every other IsocalcOptions field is read-only for the whole run and referenced directly as
    # `options.<field>` below. `mu` is the one exception: it becomes this loop's own
    # time-evolving local state (like f_atm below) - `options.mu` supplies only its *initial*
    # value and is never read again. (`options.mantle_iron` similarly seeds a local
    # `mantle_iron_state` below, once `interior_atmosphere` is available.)
    mu = options.mu

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, N_He, N_D, N_O, N_C, N_N, N_S = isofate_species_abund

    # d, T, and Fp are confirmed fixed for the whole run (never reassigned anywhere below), so
    # they're read from `system` once, here. F0 is deliberately NOT derived from System: it's a
    # modeling choice (e.g. F0 = Fp*1e-3 "for M stars"), not a strict derived quantity, so the
    # caller must still supply it directly. `system.star.mass` is no longer cached separately -
    # `system.tidal_reduction_factor(Rp)` reads it directly (see below).
    planet: Planet = system.planet
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

    escape_number_flux = EscapeNumberFlux()
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
        mu = np.dot(y, atomic_masses) / N_tot

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
            F0=F0,
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

            b = DEFAULT_SPECIES.binary_diffusion.get(light_name, heavy_name, T)

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
                M_atm = step.M_atm
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

        # advance to next step
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


@eqx.filter_jit
def _integrate_isocalc_jax(
    t0_seconds: ArrayLike,
    t_a: Array,
    y0: Array,
    M_atm0: ArrayLike,
    F0: ArrayLike,
    R_B: ArrayLike,
    thermal: bool,
    t_total: ArrayLike,
    system: System,
    escape: EscapeMechanism,
    escape_number_flux: EscapeNumberFlux,
) -> tuple[Array, Array, dict[str, Array]]:
    """The diffrax-solvable core of `isocalc_jax`: an ODE integration over `(y, M_atm)` plus the
    diagnostics back-computed from its saved trajectory. Split out from `isocalc_jax` itself (which
    also does plain-Python/NumPy setup - atmodeller scaffolding, etc. - not worth tracing) so this
    can be `eqx.filter_jit`'d: repeated calls (e.g. across an MCMC/optimization loop) then reuse one
    compiled program instead of dispatching every jnp op individually.

    `system`/`escape`/`escape_number_flux` are `eqx.Module`s (pytrees) - `filter_jit` already
    partitions their array leaves (traced) from any non-array config fields (held static)
    without needing explicit annotations, so they're passed through as-is.

    Args:
        thermal: A plain Python bool, not a traced array - `eqx.filter_jit` holds non-array
            arguments static automatically (required here, since `R_env` branches on it with a
            raw Python `if`). Every other argument here is expected to be an actual array (see
            `isocalc_jax`'s call site, which wraps its locals in `jnp.asarray` before calling) so
            that varying them across calls reuses this same compiled program rather than
            retracing.

    Returns:
        (y_a, Matm_a, alg_a): the saved trajectory (already inf-filled with the held terminal
        state past the exhaustion event) and the vmapped `_algebraic` diagnostics for every t_a
        entry.
    """
    n_tot = t_a.shape[0]
    atomic_masses = escape_number_flux.species.atomic_masses
    R_H = system.hill_radius  # Hill radius [m]
    Mp = system.planet.mass
    T = system.equilibrium_temperature
    d = system.semi_major_axis
    Fp = system.insolation

    def _algebraic(t, y, M_atm):
        """Everything derivable from (t, y, M_atm) alone - shared by the vector field and the
        post-solve diagnostics below, so the physics chain (radius_env -> ... -> Phi) is defined
        once.

        Clips y/M_atm to >= 0 before use: the adaptive step-size controller can propose trial
        steps that briefly overshoot into slightly negative territory near exhaustion (the
        original discrete loop clipped `y` after every step for the same reason), and a negative
        M_atm/f_atm would otherwise feed a fractional power (R_env's c2**0.59 term) with a
        negative base.
        """
        y = jnp.maximum(y, 0.0)
        M_atm = jnp.maximum(M_atm, 0.0)
        N_tot = jnp.sum(y)
        safe_N_tot = jnp.where(N_tot > 0, N_tot, 1.0)
        mu = jnp.dot(y, atomic_masses) / safe_N_tot
        x = jnp.where(N_tot > 0, y / safe_N_tot, jnp.zeros_like(y))

        f_atm = M_atm / Mp
        radius_env = R_env(Mp, f_atm, Fp, t, thermal)
        radius_atm = R_atm(T, Mp, system.planet.rocky_radius, radius_env, mu)
        # was `min(R_B, R_H, radius_p)` in isocalc's plain-Python loop - Python's builtin min() on
        # a traced value, same class of fix as Fxuv/Phi_1_2/Phi_minor_species earlier this
        # session.
        radius_p = jnp.minimum(
            R_B, jnp.minimum(R_H, system.planet.rocky_radius + radius_atm + radius_env)
        )

        Vpot = system.gravitational_potential(radius_p)
        A = 4 * jnp.pi * radius_p**2
        g = gravitational_acceleration(Mp, radius_p)

        state = EscapeState(
            radius_p=radius_p,
            Mp=Mp,
            T=T,
            F0=F0,
            Vpot=Vpot,
            d=d,
            A=A,
            mu=mu,
            radius_env=radius_env,
            f_atm=f_atm,
            t_now=t,
            t_total=t_total,
        )
        phi = escape.compute_mass_flux(state)
        Phi, phi_c = escape_number_flux.get_number_flux(y, T, g, phi)

        return dict(
            mu=mu,
            x=x,
            f_atm=f_atm,
            radius_env=radius_env,
            radius_p=radius_p,
            Vpot=Vpot,
            A=A,
            phi=phi,
            phi_c=phi_c,
            Phi=Phi,
        )

    def _vector_field(t, state, _args):
        """RHS of the isocalc ODE system - only the two genuine integration states' derivatives
        (dy/dt, dM_atm/dt). Everything else is recomputed from the solution afterward, below.
        """
        y, M_atm = state
        alg = _algebraic(t, y, M_atm)
        dy_dt = -alg["Phi"] * alg["A"]
        dM_atm_dt = jnp.dot(dy_dt, atomic_masses)
        return dy_dt, dM_atm_dt

    # Exact `M_atm <= 0`/`sum(y) <= 0` makes the ODE's right-hand side effectively singular as the
    # dominant species' abundance -> 0 (Phi_minor_species' N_2/N_1 term blows up), which stalls
    # Tsit5's adaptive step-size controller (step size underflows before the exact zero is ever
    # reached, rather than converging). Firing once M_atm/sum(y) drop below a small-but-nonzero
    # fraction of their initial values avoids this while still meaning "the atmosphere is gone" to
    # any reasonable tolerance.
    sum_y0 = jnp.sum(y0)
    _exhaustion_fraction = 1e-6

    def _exhausted(t, state, _args, **kwargs):
        """Event condition: entire atmosphere lost - replaces isocalc's `if M_atm <= 0 or
        sum(y) <= 0: break`.
        """
        y, M_atm = state
        return (M_atm <= _exhaustion_fraction * M_atm0) | (
            jnp.sum(y) <= _exhaustion_fraction * sum_y0
        )

    term = diffrax.ODETerm(_vector_field)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0_seconds,
        # Not `t0_seconds + t_total`: t_a's last entry runs one delta_t past that (a pre-existing
        # quirk of t_a's own formula, shared with isocalc, harmless there since t_a is only a
        # diagnostic label in the discrete loop) - diffrax requires saveat.ts to lie within
        # [t0, t1], so t1 is taken directly from t_a instead of re-derived independently.
        t1=t_a[-1],
        dt0=None,
        y0=(y0, M_atm0),
        args=None,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e3),
        saveat=diffrax.SaveAt(ts=t_a),
        event=diffrax.Event(cond_fn=_exhausted),
        max_steps=100_000,
    )
    y_a, Matm_a = sol.ys

    # Entries of t_a at/after the point where the event fired come back as inf (diffrax's
    # SaveAt(ts=...)+Event interaction - empirically confirmed, not documented behavior). Replace
    # them by holding the last finite (i.e. last actually-integrated) trajectory value constant,
    # falling back to the initial condition when no saved entry is finite at all (e.g. a short
    # exhaustion time relative to the requested output cadence, so every t_a entry postdates it).
    finite_mask = jnp.isfinite(Matm_a) & jnp.all(jnp.isfinite(y_a), axis=1)
    any_finite = jnp.any(finite_mask)
    last_finite_idx = jnp.max(jnp.where(finite_mask, jnp.arange(n_tot), -1))
    last_finite_idx = jnp.maximum(last_finite_idx, 0)
    y_fallback = jnp.where(any_finite, y_a[last_finite_idx], y0)
    Matm_fallback = jnp.where(any_finite, Matm_a[last_finite_idx], M_atm0)
    y_a = jnp.where(finite_mask[:, None], y_a, y_fallback)
    Matm_a = jnp.where(finite_mask, Matm_a, Matm_fallback)

    # Back-compute every diagnostic from the saved trajectory in one vmapped pass, rather than a
    # per-timestep Python loop.
    alg_a = jax.vmap(_algebraic)(t_a, y_a, Matm_a)

    return y_a, Matm_a, alg_a


def isocalc_jax(
    system: System,
    F0,
    time=5e9,
    isofate_species_abund: Array = jnp.array([0, 0, 0, 0, 0, 0, 0]),
    options: IsocalcOptions = IsocalcOptions(),
    escape: EscapeMechanism = XUVEscape(),
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

    # Every other IsocalcOptions field is read-only for the whole run and referenced directly as
    # `options.<field>` below. `mu` is the one exception: it becomes this loop's own
    # time-evolving local state (like f_atm below) - `options.mu` supplies only its *initial*
    # value and is never read again. (`options.mantle_iron` similarly seeds a local
    # `mantle_iron_state` below, once `interior_atmosphere` is available.)
    mu = options.mu

    # isofate_species_abund is ordered per isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N,
    # S) - the same order used throughout this function for y, atomic_masses, and species_names
    # below.
    N_H, _, N_D, _, _, _, _ = isofate_species_abund

    # T is confirmed fixed for the whole run (never reassigned below), so it's read from
    # `system` once, here, to seed R_B - `_integrate_isocalc_jax` re-derives T/d/Fp/Mp from
    # `system` itself (already one of its arguments), so they don't need to be threaded through
    # separately. F0 is deliberately NOT derived from System: it's a modeling choice (e.g. F0 =
    # Fp*1e-3 "for M stars"), not a strict derived quantity, so the caller must still supply it
    # directly. `system.star.mass` is no longer cached separately -
    # `system.tidal_reduction_factor(Rp)` reads it directly (see below).
    planet: Planet = system.planet
    T: ArrayLike = system.equilibrium_temperature

    # Only planet.mass and planet.f_atm are ever read, and only here, once, to seed the
    # *initial* conditions: Mp never changes over the run, but f_atm is immediately reassigned
    # to a plain float and becomes this loop's own time-evolving local variable (see M_atm/f_atm
    # below) - it must never be read from `planet` (or `system`) again after this point.
    Mp = planet.mass
    f_atm = planet.f_atm

    ###_____Initialize physical values_____###

    escape_number_flux = EscapeNumberFlux()
    R_B = system.bondi_radius(mu, T)  # Bondi radius [m]

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
    M_atm0 = Mp * f_atm  # initial atmospheric mass [kg]
    M_atm = M_atm0
    # Atmospheric number of atoms per species [atoms], ordered per
    # isofate.species.ELEMENTS/SYMBOLS (H, He, D, O, C, N, S), same as isofate_species_abund above.
    y = np.array(isofate_species_abund, dtype=float)
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

    y_a, Matm_a, alg_a = _integrate_isocalc_jax(
        jnp.asarray(t0_seconds),
        jnp.asarray(t_a),
        jnp.asarray(y),
        jnp.asarray(M_atm),
        jnp.asarray(F0),
        jnp.asarray(R_B),
        options.thermal,
        jnp.asarray(t_total),
        system,
        escape,
        escape_number_flux,
    )
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
