'''
Collin Cherubim
June 30, 2025
Test script for atmodeller function for isocalc to call for isofate-atmodeller coupling
'''

import logging
import optimistix as optx
from atmodeller import SolverParameters
from atmodeller.solubility import get_solubility_models
solubility_models = get_solubility_models()

#isofate imports
from isofate.constants import const
from isofate.orbit_params import *
from isofate.isofunks import *

#other imports
import numpy as np

# logger = debug_logger()
# logger.setLevel(logging.INFO)

def AtmodellerCoupler(Teq, Mp, Rp, mu, melt_fraction, mantle_iron_dict,
                      N_H_atm, N_He_atm, N_O_atm, N_C_atm, N_N_atm, N_S_atm,
                      N_H_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, interior_atmosphere):

    results = {}
    gamma = 7/5
    T_surface, P_surface = make_atmosphere_descent(Teq, mu, Rp, Mp, gamma, 2)
    surface_temperature: float = np.min([6000, T_surface]) # K
    if melt_fraction != False:
        mantle_melt_fraction: float = melt_fraction
    elif melt_fraction == False:
        mantle_melt_fraction: float = MeltFraction(Mp, np.clip(T_surface, 10, 16000))
    planet_mass: float = Mp
    surface_radius: float = R_core(Mp)

    # element masses
    mass_H: float = (N_H_atm + N_H_int)*const.mu_H
    mass_O: float = (N_O_atm + N_O_int)*const.mu_O
    mass_C: float = (N_C_atm + N_C_int)*const.mu_C
    mass_He: float = (N_He_atm + N_He_int)*const.mu_He
    mass_N: float = (N_N_atm + N_N_int)*const.mu_N
    mass_S: float = (N_S_atm + N_S_int)*const.mu_S
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
        "S": max(MASS_CUTOFF, mass_S)
    }

    # solver = optx.Newton
    # solver_parameters = SolverParameters(solver=solver, throw=True)
    model = interior_atmosphere.update_state(
        planet_mass=planet_mass,
        mantle_melt_fraction=mantle_melt_fraction,
        surface_radius=surface_radius,
        temperature=surface_temperature,
    ).update_constraints(
        mass_constraints=mass_constraints,
        # solver_parameters=solver_parameters
    )

    output = model.solve_with_default()
    # to_numpy=True returns read-only arrays; entries that need in-place-style updates below are
    # replaced wholesale (new array of the same (1, 1) shape) rather than item-assigned
    sol = output.to_dict(output_format="elements_species", to_numpy=True)
    results['N_H_atm'] = sol['H']['gas']['number_moles'][0][0]*const.avogadro
    results['N_H_int'] = sol['H']['silicate_melt']['number_moles'][0][0]*const.avogadro
    results['N_He_atm'] = sol['He']['gas']['number_moles'][0][0]*const.avogadro
    results['N_He_int'] = sol['He']['silicate_melt']['number_moles'][0][0]*const.avogadro

    if mantle_iron_dict:
        if mantle_iron_dict['type'] == 'dynamic':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro
                results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro
            else:
                mantle_iron_dict['mass_Fe'] = mantle_melt_fraction*Mp*mantle_iron_dict['Fe_mass_fraction']
                if mantle_iron_dict['mass_Fe'] > 0:
                    mantle_iron_dict['mass_Fe2'] = mantle_iron_dict['X_Fe2']*mantle_iron_dict['mass_Fe']
                    n_Fe2 = mantle_iron_dict['mass_Fe2']/const.M_Fe - 4*sol['O2_g']['gas']['number_moles'][0][0] # oxidize Fe2+ to Fe3+
                    n_O2_atm = sol['O2_g']['gas']['number_moles'][0][0] - 0.25*(mantle_iron_dict['mass_Fe2']/const.M_Fe - n_Fe2)
                    delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/const.M_Fe - n_Fe2)
                    sol['O2_g']['gas']['number_moles'] = np.array([[np.max([n_O2_atm,0])]])
                    results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro - 2*delta_n_O2_atm*const.avogadro
                    sol['O']['gas']['number_moles'] = np.array([[results['N_O_atm']/const.avogadro]])
                    results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro + 2*delta_n_O2_atm*const.avogadro
                    sol['O']['silicate_melt']['number_moles'] = np.array([[results['N_O_int']/const.avogadro]])
                    mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*const.M_Fe, 0]) # update remaining Fe2 mass
                    mantle_iron_dict['X_Fe2'] = mantle_iron_dict['mass_Fe2']/mantle_iron_dict['mass_Fe']
                else:
                    results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro
                    results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro
        elif mantle_iron_dict['type'] == 'static':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro
                results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro
            else:
                n_Fe2 = mantle_iron_dict['mass_Fe2']/const.M_Fe - 4*sol['O2_g']['gas']['number_moles'][0][0] # oxidize Fe2+ to Fe3+; this goes neg. when n_O2 > n_Fe2/4
                n_O2_atm = sol['O2_g']['gas']['number_moles'][0][0] - 0.25*(mantle_iron_dict['mass_Fe2']/const.M_Fe - n_Fe2) # goes to zero when when n_O2 >= n_Fe2/4 (won't go neg.)
                delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/const.M_Fe - n_Fe2) # calculates exactly the n_O2 reacted away
                sol['O2_g']['gas']['number_moles'] = np.array([[np.max([n_O2_atm,0])]]) # goes to zero even when n_Fe2 goes neg., which should put O2 back in atm. Confirmed: doesn't matter b/c of line results['N_O_atm'] = ...
                results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro - 2*delta_n_O2_atm*const.avogadro # takes away only O2 reacted from total O_atm inventory
                sol['O']['gas']['number_moles'] = np.array([[results['N_O_atm']/const.avogadro]])
                results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro + 2*delta_n_O2_atm*const.avogadro
                sol['O']['silicate_melt']['number_moles'] = np.array([[results['N_O_int']/const.avogadro]])
                mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*const.M_Fe, 0]) # update remaining Fe2 mass
    else:
        results['N_O_atm'] = sol['O']['gas']['number_moles'][0][0]*const.avogadro
        results['N_O_int'] = sol['O']['silicate_melt']['number_moles'][0][0]*const.avogadro
    results['N_C_atm'] = sol['C']['gas']['number_moles'][0][0]*const.avogadro
    results['N_C_int'] = sol['C']['silicate_melt']['number_moles'][0][0]*const.avogadro
    results['N_N_atm'] = sol['N']['gas']['number_moles'][0][0]*const.avogadro
    results['N_N_int'] = sol['N']['silicate_melt']['number_moles'][0][0]*const.avogadro
    results['N_S_atm'] = sol['S']['gas']['number_moles'][0][0]*const.avogadro
    results['N_S_int'] = sol['S']['silicate_melt']['number_moles'][0][0]*const.avogadro
    results['M_atm'] = sol['gas']['phase']['mass'][0][0]
    results['T_surface'] = T_surface
    results['T_surface_atmod'] = surface_temperature

    return results, sol, mantle_iron_dict
