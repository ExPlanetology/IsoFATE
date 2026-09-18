# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical constants and binary diffusion coefficients used in IsoFATE."""

from dataclasses import dataclass

from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.sci_utils import constants as _constants

# NOTE: Re and Me are deliberately NOT sourced from atmodeller.sci_utils.earth: earth.radius
# (6.371e6 m, the mean/volumetric Earth radius) differs from Re (6.378e6 m, the equatorial
# radius) by ~0.1% - a real convention difference, not just precision, and Re feeds directly into
# Planet.rocky_radius's Lopez & Fortney 2014 mass-radius scaling relation, which may assume one
# specific convention. earth.mass (5.972e24 kg) is also less precise than Me (5.9722e24 kg), so
# swapping would be a downgrade rather than an improvement.


@dataclass(frozen=True)
class PhysicalConstants:
    inv_cm2m: float = 100  # convert inverse cm to inverse m
    avogadro: float = _constants.Avogadro  # Avogadro's number [particles/mole]
    s2day: float = 1 / (3600 * 24)  # convert s to day
    s2yr: float = 1 / (3600 * 24 * 365)  # convert s to year
    au2m: float = _constants.au  # convert au to m
    cgs2si_flux: float = 1 / 1000  # convert flux from erg/cm2/s to W/m2
    erg2joule: float = 1e-7  # convert ergs to Joules
    gcm2kgm: float = 1000  # convert g/cm3 to kg/m3 (SI)
    J2E_mass: float = 1.898e27 / 5.972e24  # convert Jupiter mass to Earth mass [kg]
    J2E_rad: float = 7.1492e7 / 6.3781e6  # convert Jupiter radius to Earth radius [m]
    R_gas: float = GAS_CONSTANT  # gas constant [J/mol/K]
    kb: float = _constants.Boltzmann  # Boltzmann constant [m2 kg/s2 K]
    sbc: float = _constants.Stefan_Boltzmann  # Stefan Boltzmann constant W/m2/K4
    g_e: float = 9.8  # grav field strength [m/s/s]
    G: float = _constants.gravitational_constant  # [m3/kg/s2]
    h: float = _constants.h  # [J s]
    c: float = _constants.c  # [m/s]
    Re: float = 6.378e6  # Earth radius [m]
    Me: float = 5.9722e24  # Earth mass [kg]
    Fe: float = 1366  # Earth bolometric flux [W/m2]
    Ms: float = 1.98847e30  # Solar mass [kg]
    Rs: float = 6.957e8  # Solar radius [m]
    Ls: float = 3.828e26  # Solar luminosity [W]
    Mjup: float = 1.898e27  # Jupiter mass [kg]
    Rjup: float = 7.1492e7  # Jupiter radius [m]
    M_N2: float = 0.028014  # molar mass N2 [kg/mol]
    M_H2: float = 0.002016  # molar mass H2 [kg/mol]
    M_HD: float = 0.003024  # molar mass HD [kg/mol]
    M_D: float = 0.002014  # molar mass of D [kg/mol]
    M_H: float = 0.001008  # molar mass of H [kg/mol]
    M_He: float = 0.0040026  # molar mass of He [kg/mol]
    M_O: float = 0.015999  # molar mass of O [kg/mol]
    M_C: float = 0.012011  # molar mass of C [kg/mol]
    M_N: float = 0.014007  # molar mass of N [kg/mol]
    M_S: float = 0.032065  # molar mass of S [kg/mol]
    M_Fe: float = 0.055845  # molar mass of Fe [kg/mol]
    M_H2O: float = 0.018015  # molar mass of H2O [kg/mol]
    vsmow: float = 1.5574e-4  # Vienna Standard Mean Ocean Water (D/H for Earth's oceans)
    DtoH_solar: float = 0.0000194  # D/H solar mole ratio from Lodders 2003
    HetoH_solar: float = (
        0.07991  # He/H solar mole ratio from Lodders 2003 (0.2377/0.7491*mu_H/mu_He)
    )
    HetoH_protosolar: float = (
        0.09709  # He/H proto-solar mole ratio Lodders 2003 (0.2741/0.711*mu_H/mu_He)
    )
    HetoH_protosolar_mass: float = (
        0.38551  # He/H proto-solar mass ratio Lodders 2003 (0.2741/0.711)
    )
    OtoH_protosolar: float = (
        0.00058  # O/H proto-solar mole ratio from Lodders 2003 Table 2 (1.413e7/2.431e10)
    )
    CtoH_protosolar: float = 0.00029  # C/H proto-solar mole ratio from Lodders 2003 Table 2
    NtoH_protosolar: float = (
        0.000080  # N/H proto-solar mole ratio from Lodders 2003 Table 2 (1.950e6/2.431e10)
    )
    StoH_protosolar: float = (
        0.000018  # S/H proto-solar mole ratio from Lodders 2003 Table 2 (4.449e5/2.431e10)
    )
    n_OperTO: float = 7.83e22  # mols O per TO
    Venus_TO: float = 1.32e-5  # number of terrestrial oceans in Venus' atmosphere (needs checking)

    @property
    def Fe_xuv(self):
        """Earth XUV flux [W/m2] [Ribas et al. 2005]"""
        return 3.88 * self.cgs2si_flux

    @property
    def mu_H2(self):
        """Molecular mass of H2 [kg/molecule]"""
        return self.M_H2 / self.avogadro

    @property
    def mu_HD(self):
        """Molecular mass of HD [kg/molecule]"""
        return self.M_HD / self.avogadro

    @property
    def mu_H(self):
        """Atomic mass of H [kg/atom]"""
        return self.M_H / self.avogadro

    @property
    def mu_D(self):
        """Atomic mass of D [kg/atom]"""
        return self.M_D / self.avogadro

    @property
    def mu_He(self):
        """Atomic mass of He [kg/atom]"""
        return self.M_He / self.avogadro

    @property
    def mu_O(self):
        """Atomic mass of O [kg/atom]"""
        return self.M_O / self.avogadro

    @property
    def mu_C(self):
        """Atomic mass of C [kg/atom]"""
        return self.M_C / self.avogadro

    @property
    def mu_N(self):
        """Atomic mass of N [kg/atom]"""
        return self.M_N / self.avogadro

    @property
    def mu_S(self):
        """Atomic mass of S [kg/atom]"""
        return self.M_S / self.avogadro

    @property
    def mu_Fe(self):
        """Atomic mass of Fe [kg/atom]"""
        return self.M_Fe / self.avogadro

    @property
    def DtoH_solar_mass(self):
        """D/H solar mass ratio from Lodders 2003"""
        return self.DtoH_solar * (self.mu_D / self.mu_H)

    @property
    def OtoH_protosolar_mass(self):
        """O/H proto-solar mass ratio from Lodders 2003"""
        return self.OtoH_protosolar * (self.mu_O / self.mu_H)

    @property
    def CtoH_protosolar_mass(self):
        """O/H proto-solar mass ratio from Lodders 2003"""
        return self.CtoH_protosolar * (self.mu_C / self.mu_H)

    @property
    def NtoH_protosolar_mass(self):
        """O/H proto-solar mass ratio"""
        return self.NtoH_protosolar * (self.mu_N / self.mu_H)

    @property
    def StoH_protosolar_mass(self):
        """O/H proto-solar mass ratio"""
        return self.StoH_protosolar * (self.mu_S / self.mu_H)

    @property
    def mu_HHe(self):
        """H/He with solar abundances"""
        return 0.00122 / self.avogadro

    @property
    def mu_H2He(self):
        """H2/He with solar abundances"""
        return 0.00227 / self.avogadro

    @property
    def mu_solar(self):
        """Average particle mass for solar metallicity"""
        return 0.00235 / self.avogadro


const = PhysicalConstants()
