# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Star and Planet parameter objects for defining simulation scenarios.

sim.py currently hand-types one active (R_star, M_star, T_star) / (Mp, P, f_atm) block plus many
commented-out alternatives for other systems. These dataclasses are meant to replace each of those
blocks with a single `Star(...)`/`Planet(...)` instance per system.
"""

from dataclasses import dataclass

import numpy as np

from isofate.constants import const
from isofate.isofunks import R_Bondi, R_core, R_Hill
from isofate.orbit_params import EqTemp, Insolation, Luminosity, SemiMajor


@dataclass
class Star:
    """A host star.

    Args:
        radius: Stellar radius [m]
        mass: Stellar mass [kg]
        temperature: Stellar effective temperature [K]
        luminosity: Stellar luminosity [W] (optional; if not provided, computed from radius and
            temperature)
    """

    radius: float
    mass: float
    temperature: float
    _luminosity: float | None = None

    def __init__(
        self, radius: float, mass: float, temperature: float, luminosity: float | None = None
    ):
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self._luminosity = luminosity

    @property
    def luminosity(self) -> float:
        """Stellar luminosity [W], via the Stefan-Boltzmann law."""
        if self._luminosity is not None:
            return self._luminosity
        else:
            return Luminosity(self.radius, self.temperature)

    @property
    def t_jump(self) -> float:
        """XUV saturation-time scaling factor [Gyr] for M dwarfs (t_sat = t_jump*1e9 [yr]).

        Only meaningful for M dwarfs; carried over as-is from the fit used in sim.py.
        """
        # TODO: Maybe remove if only relevant for LHS 1140?
        return 5.9 - 15.4 * (self.mass / const.Ms)

    def period_for_semi_major_axis(self, a: float) -> float:
        """Orbital period [s] a planet at semi-major axis `a` [m] would need, around this star.
        The inverse of `System.semi_major_axis` - a "what-if" helper for picking a
        `Planet.period` at construction time, before any Planet/System exists yet."""
        return (a**3 * 4 * np.pi**2 / const.G / self.mass) ** 0.5


@dataclass(frozen=True)
class Planet:
    """A planet orbiting a Star.

    Args:
        mass: Planet mass [kg]
        period: Orbital period [s]
        f_atm: Atmospheric mass fraction [ndim]
    """

    mass: float
    period: float
    f_atm: float

    @property
    def atmosphere_mass(self) -> float:
        """Initial atmospheric mass [kg]."""
        return self.mass * self.f_atm

    @property
    def core_radius(self) -> float:
        """Planetary core radius [m] (rocky component; Lopez & Fortney 2014)."""
        return R_core(self.mass)


@dataclass
class System:
    """A planet orbiting a star.

    Groups the quantities that genuinely depend on both bodies (or the orbit between them), as
    opposed to properties of the Star or Planet alone. `mu` (mean atmospheric particle mass) is
    deliberately never stored here: unlike everything else on this class, it evolves over the
    course of a simulation (see isocalc), so it must stay something the caller passes in each
    time rather than a fixed System property.

    Args:
        star: The host star
        planet: The orbiting planet
    """

    star: Star
    planet: Planet

    @property
    def semi_major_axis(self) -> float:
        """Orbital semi-major axis [m]."""
        return SemiMajor(self.star.mass, self.planet.period)

    @property
    def insolation(self) -> float:
        """Incident bolometric flux at the planet [W/m2]."""
        return Insolation(self.star.luminosity, self.semi_major_axis)

    @property
    def equilibrium_temperature(self) -> float:
        """Planetary equilibrium temperature [K], assuming zero albedo."""
        return EqTemp(self.insolation)

    @property
    def hill_radius(self) -> float:
        """Hill radius [m]."""
        return R_Hill(self.planet.mass, self.star.mass, self.semi_major_axis)

    def bondi_radius(self, mu: float, T: float | None = None) -> float:
        """Bondi radius [m].

        Args:
            mu: Mean atmospheric particle mass [kg] - time-evolving, must be supplied by the
                caller (see the class docstring).
            T: Temperature [K]. Defaults to `equilibrium_temperature` if not given.
        """
        return R_Bondi(self.planet.mass, mu, T if T is not None else self.equilibrium_temperature)


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
LHS1140b: Planet = Planet(mass=5.6 * const.Me, period=24.74 / const.s2day, f_atm=0.00085)

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
