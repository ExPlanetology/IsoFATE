# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preset Star/Planet instances for known and example systems, built from the `system.py` classes."""

from isofate.constants import const
from isofate.system import Planet, Star

# Sun-like star
# R_star = 1.0*Rs # [m]
# M_star = 1.0*Ms
# T_star = 6000 # [K]
DefaultStar: Star = Star(radius=const.Rs, mass=const.Ms, temperature=6000)

# M1 star
# R_star = 0.5*Rs # [m]
# M_star = 0.5*Ms
# T_star = 3600 # [K]
M1Star: Star = Star(radius=0.5 * const.Rs, mass=0.5 * const.Ms, temperature=3600)

# K5 star
# R_star = 0.7*Rs # [m]
# M_star = 0.7*Ms
# T_star = 4440 # [K]
K5Star: Star = Star(radius=0.7 * const.Rs, mass=0.7 * const.Ms, temperature=4440)

# LHS 1140
R_star = 0.22 * const.Rs  # [m]
M_star = 0.18 * const.Ms  # [kg]
T_star = 3096  # [K]
# Currently coded as a method in the class
# t_jump = 5.9 - 15.4 * (M_star / const.Ms)
# TODO: This was wired in, but presumably should be an override?
# L = 0.0038 * const.Ls
LHS1140Star: Star = Star(radius=R_star, mass=M_star, temperature=T_star)  # , luminosity=L)

# # Kepler-138
# R_star = 0.535*Rs # [m]
# M_star = 0.535*Ms # [kg]
# T_star = 3841 # [K]
Kepler138Star: Star = Star(radius=0.535 * const.Rs, mass=0.535 * const.Ms, temperature=3841)

# K2-3
# R_star = 0.546*Rs # [m]
# M_star = 0.549*Ms # [kg]
# T_star = 3844 # [K]
K23Star: Star = Star(radius=0.546 * const.Rs, mass=0.549 * const.Ms, temperature=3844)

# GJ 3090
# R_star = 0.516*Rs # [m]
# M_star = 0.519*Ms # [kg]
# T_star = 3556 # [K]
GJ3090Star: Star = Star(radius=0.516 * const.Rs, mass=0.519 * const.Ms, temperature=3556)

# K2-18
# R_star = 0.469*Rs # [m]
# M_star = 0.495*Ms
# T_star = 3500 # [K]
K218Star: Star = Star(radius=0.469 * const.Rs, mass=0.495 * const.Ms, temperature=3500)

# HAT-P-11, K4V
# R_star = 0.752*Rs # [m]
# M_star = 0.809*Ms # [kg]
# T_star = 4780 # [K]
HATP11Star: Star = Star(radius=0.752 * const.Rs, mass=0.809 * const.Ms, temperature=4780)

# TRAPPIST-1
# R_star = 0.1192*Rs # [m]
# M_star = 0.0898*Ms # [kg]
# T_star = 2566 # [K]
# t_jump = 5.9 - 15.4*(M_star/Ms)
TRAPPIST1Star: Star = Star(radius=0.1192 * const.Rs, mass=0.0898 * const.Ms, temperature=2566)

# TODO: Here define some planets

# Mp = 1.0510228321316297*Me
# P = 1.923431660504492/s2day # 0.2 au for solar type star

# Mp = 2.67899838*Me
# P = 19.76878782/s2day
# f_atm = 1.03428670e-02

# Mp = 1.15662286*Me
# P = 150/s2day
# f_atm = 0.00179197

# O2 planet
# Mp = 10.110718251912926*Me
# P = 1.1579370531867088/s2day
# f_atm = 1.46472048e-01

# O2 world
# Mp = 12.408678941478096*Me
# P = 2.28472886/s2day
# f_atm = 2.17823165e-02

# Mp = 1.1020903707177876*Me
# P = 85.80220254822505/s2day
# f_atm = 0.00134977

# f_atm = 0.04763652467166532
# Mp = 3.2335611753347924*Me
# P = 2.915963713301288/s2day

# K2-18 b
# f_atm = 0.001
# Mp = 8.63*Me
# P = 33/s2day

# LHS 1140 b
# f_atm = 0.00085
# Mp = 5.6 * const.Me
# P = 24.74 / const.s2day
LHS1140b: Planet = Planet(mass=5.6 * const.Me, period=24.74 / const.s2day)

# GJ 3090 b
# f_atm = 0.03
# Mp = 3.34*Me
# P = 2.9/s2day
# Rp = 2.13

# # Kepler-138 d
# f_atm = 0.012
# Mp = 2.1*Me
# P = 23/s2day

# K2-3 c
# f_atm = 0.015
# Mp = 2.68*Me
# P = 24.6/s2day

# # IL world
# # f_atm = 0.001049211388465194
# # Mp = 1.2474905792906028*Me
# # P = 86.23173213822659
# f_atm = 2.00655135e-03
# Mp = 1.3615101475752467*Me
# P = 51.53009252/s2day

# # HAT-P-11 b
# f_atm = 0.13
# Mp = 0.081*Mjup
# # Mp = 0.073*Mjup
# P = 4.8878162/s2day
# # P = Period(0.0425*au2m, M_star) # perihelion
# # P = Period(0.0635*au2m, M_star) # aphelion

# # TRAPPIST-1 c
# f_atm = 0.0265
# Mp = 1.308*Me
# P = 2.42/s2day
# # F_final = 1e2*Fe_xuv

# TRAPPIST-1 f
# f_atm = 0.005
# Mp = 1.039*Me
# P = 9.21/s2day
# F_final = 1e2*Fe_xuv

# TRAPPIST-1 g
# f_atm = 0.005
# Mp = 1.321*Me
# P = 12.35/s2day

# # TRAPPIST-1 h
# f_atm = 0.02
# Mp = 0.0326*Me
# P = 18.77/s2day

# # N2 dynamic test planet
# # f_atm = 0.03795544265173505
# f_atm = 0.0398
# Mp = 2.0063872276215733*Me
# P = 2.7202404227417087/s2day
