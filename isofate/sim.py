# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a simulation of atmospheric escape for a given planet-star system using IsoFATE."""

import time as TIME

import matplotlib.pyplot as plt

from isofate.constants import const
from isofate.escape.mechanisms import XUVEscape
from isofate.initial_condition import ProtosolarInitialCondition
from isofate.isofate_coupler import isocalc
from isofate.isoplot import isoplot
from isofate.parameters import IsocalcOptions, Parameters
from isofate.presets import LHS1140b, LHS1140Star
from isofate.species import DEFAULT_SPECIES
from isofate.system import Planet, Star, System

# LHS 1140 / LHS 1140 b
star: Star = LHS1140Star
planet: Planet = LHS1140b
system = System(star=star, planet=planet)

# kept as plain names since the rest of this script still refers to them directly
t_jump = star.t_jump
# Initial atmospheric mass fraction, used by the initial condition (ProtosolarInitialCondition)
# below; the drivers derive the atmosphere mass/fraction from the initial abundances
f_atm = 0.01  # 0.00085
Mp = planet.mass
P = planet.period

Fp = system.insolation  # [W/m2]
T = system.equilibrium_temperature  # planetary eq temp [K]
F0 = Fp * 1e-3  # use for M star
F_final = 0.17  # LHS 1140; 0.170 for GJ 699 MUSCLES; use 0.033 for GJ 1132 MUSCLES
# F_final = 0.175 # K2-3 c Diamond-Lowe et al. 2022
# F_final = 1e2*Fe_xuv
# F0 = Fp*10**(-3.5)*(M_star/Ms) # use for G, K stars
flux_model = "power law"
stellar_type = "M1"
# t_sat = 2e8 # XUV saturation time [yr]
t_sat = t_jump * 1e9  # XUV saturation time [yr]
time = 5e9  # total simulation time [yr]
t0 = 1e6  # start time [yr]
t_pms = 0  # pms phase duration [yr]
step_fn = True
RR = True
eps = 0.15
rad_evol = True
n_steps = int(1e5)
# Collin starts with 1e3
# For 1e4 the run-time is reasonable.
# TODO: Dan turned off to refactor without Atmodeller to start with.  Then we'll add it back in.
n_atmodeller = int(1e4)  # <-- FIXME: set much smaller Atmodeller runs slow (1e2)
thermal = True
dynamic_phi = True
OtoH_enhancement = 1
# initial atmosphere: protosolar elemental ratios scaled to f_atm of the planet mass
initial_condition = ProtosolarInitialCondition(
    atmosphere_mass_fraction=f_atm, OtoH_enhancement=OtoH_enhancement
)
# atomic masses [kg/atom], by symbol, for the diagnostics below
atomic_mass = DEFAULT_SPECIES.mass_by_symbol
# protosolar mole ratios to H, for D/H and O/H in solar units below
protosolar = ProtosolarInitialCondition.PROTOSOLAR_MOLE_RATIOS_TO_H

# these print statements serve as a check when running sim.py
print("n_steps =", n_steps)
print("n_atmodeller =", n_atmodeller)
print("f_atm =", f_atm)
print("Mp =", Mp / const.Me, "M_earth")
print("F0 =", round(F0, 3), "W/m2")
print("flux model:", flux_model)
print("stellar type:", stellar_type)
print("P =", P * const.s2day, "days")
print("d =", round(system.semi_major_axis / const.au2m, 3), "au")
print("Fp =", round(Fp, 1), "W/m2")
print("Teq =", round(T, 1), "K")
print("time =", time / 1e9, "Gyr")
print("rad_evol =", rad_evol)
print("dynamic_phi =", dynamic_phi)


escape_mechanism = XUVEscape(
    F0=F0,
    eps=eps,
    t0=t0,
    t_sat=t_sat,
    beta=-1.23,
    step_fn=step_fn,
    F_final=F_final,
    t_pms=t_pms,
    pms_factor=1e2,
    flux_model=flux_model,
    activity="medium",
    stellar_type=stellar_type,
    RR=RR,
)
options = IsocalcOptions(
    rad_evol=rad_evol,
    n_steps=n_steps,
    t_start=t0,
    thermal=thermal,
    n_atmodeller=n_atmodeller,
    dynamic_phi=dynamic_phi,
)

parameters = Parameters(
    system,
    escape_mechanism=escape_mechanism,
    initial_condition=initial_condition,
    isocalc_options=options,
)
# initial atmospheric abundances [atoms], for the diagnostics printed below
N_H, N_He, N_D, N_O, N_C, N_N, N_S = (float(n) for n in parameters.initial_abundances())

# run simulation (from isofate.py)
isocalc_start = TIME.time()
output = isocalc(parameters, time)
print(f"isocalc runtime: {TIME.time() - isocalc_start:.2f} s")
sol = output.to_dict(to_numpy=True)

# path = '/Users/collin/Documents/Harvard/Research/atm_escape/IsoFATE/monte_carlo/atmodeller/corrected_Psi/transient_D_world_full_isofate'
# outfile = open(path, 'wb')
# pickle.dump(sol, outfile)
# outfile.close()

print("checkpoint 1")

# solutions for the diagnostics printed below (isoplot reads its own from the output)
rp_a = sol["Rp"]
fenv_a = sol["fatm"]
NH_a = sol["N_H"]
NHe_a = sol["N_He"]
ND_a = sol["N_D"]
NO_a = sol["N_O"]
NC_a = sol["N_C"]
# calculate mass fraction of He

Y = (
    NHe_a
    * atomic_mass["He"]
    / (
        NH_a * atomic_mass["H"]
        + NHe_a * atomic_mass["He"]
        + ND_a * atomic_mass["D"]
        + NO_a * atomic_mass["O"]
        + NC_a * atomic_mass["C"]
    )
)

print("\n")
print("final f_atm =", fenv_a[-1])
print(
    "final f_atm by species =",
    (
        NH_a[-1] * atomic_mass["H"]
        + NHe_a[-1] * atomic_mass["He"]
        + ND_a[-1] * atomic_mass["D"]
        + NO_a[-1] * atomic_mass["O"]
        + NC_a[-1] * atomic_mass["C"]
    )
    / Mp,
)
print("initial f_atm =", fenv_a[0])
print(
    "initial f_atm by species =",
    (
        N_H * atomic_mass["H"]
        + N_He * atomic_mass["He"]
        + N_D * atomic_mass["D"]
        + N_O * atomic_mass["O"]
        + N_C * atomic_mass["C"]
    )
    / Mp,
)
print("final D/H =", ND_a[-1] / NH_a[-1] / protosolar["D"], "[Solar]")
print("final O/H =", NO_a[-1] / NH_a[-1] / protosolar["O"], "[Solar]")
print(
    "final X_He (molar concn) =",
    NHe_a[-1] / (NH_a[-1] + NHe_a[-1] + ND_a[-1] + NO_a[-1] + NC_a[-1]),
)  # molar concentration
print("final Y_He (mass concn) =", Y[-1])  # mass fraction
print("final planetary radius =", rp_a[-1] / const.Re, "R_Earth")
print("\n")

# plotting (isofate.isoplot.isoplot)
fig = isoplot(output, parameters)

# path = '/Users/collin/Documents/Harvard/Research/atm_escape/IsoFATE/case_studies/LHS1140b_eps80_f0018_Fi1_tjump_Ffinal170_XUV+RR_atmod1e2_ntime1e5_t5e9_tpms0'
# path = '/Users/collin/Documents/Harvard/Research/atm_escape/IsoFATE/case_studies/N2world_dynamic_phi_fatm0398_XUV+RR_atmod1e2_ntime1e5_t5e9_tpms0'
# path = '/Users/collin/Documents/Harvard/Research/atm_escape/IsoFATE/case_studies/HAT-P-11b_fatm18_aphelion_XUV+RR_atmod0_ntime1e5_tsat1e9_tpms0'
# plt.savefig(path+'.png', dpi = 300, bbox_inches = 'tight')


# with open(path, 'wb') as file:
#     pickle.dump(sol, file)

plt.show()
