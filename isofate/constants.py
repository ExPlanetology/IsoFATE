# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical constants and binary diffusion coefficients used in IsoFATE."""

import equinox as eqx
from atmodeller.sci_utils import GAS_CONSTANT, earth
from atmodeller.sci_utils import constants as _constants


# NOTE: Re is deliberately NOT sourced from atmodeller.sci_utils.earth: earth.radius (6.371e6 m,
# the mean/volumetric Earth radius) differs from Re (6.378e6 m, the equatorial radius) by ~0.1% -
# a real convention difference, not just precision - and Re feeds directly into R_rocky's Lopez &
# Fortney 2014 mass-radius scaling relation (isofunks.py), which is pinned to the equatorial-radius
# convention specifically ("Re not in paper, typo"). Swapping it shifts escape-flux results by
# ~0.1% (confirmed: breaks tests/test_escape.py's CPMLEscape/CombinedEscape regression pins).
# Me *is* sourced from earth.mass - atmodeller's earth.mass is now equally precise (5.9722e24 kg).


class PhysicalConstants(eqx.Module):
    inv_cm2m: float = 100  # convert inverse cm to inverse m
    avogadro: float = _constants.Avogadro  # Avogadro's number [particles/mole]
    s2day: float = 1 / (3600 * 24)  # convert s to day
    s2yr: float = 1 / (3600 * 24 * 365)  # convert s to year
    au2m: float = _constants.au  # convert au to m
    cgs2si_flux: float = 1 / 1000  # convert flux from erg/cm2/s to W/m2
    erg2joule: float = 1e-7  # convert ergs to Joules
    gcm2kgm: float = 1000  # convert g/cm3 to kg/m3 (SI)
    R_gas: float = GAS_CONSTANT  # gas constant [J/mol/K]
    kb: float = _constants.Boltzmann  # Boltzmann constant [m2 kg/s2 K]
    sbc: float = _constants.Stefan_Boltzmann  # Stefan Boltzmann constant W/m2/K4
    g_e: float = 9.8  # grav field strength [m/s/s]
    G: float = _constants.gravitational_constant  # [m3/kg/s2]
    h: float = _constants.h  # [J s]
    c: float = _constants.c  # [m/s]
    Re: float = 6.378e6  # Earth radius [m]
    Me: float = earth.mass  # Earth mass [kg]
    Fe: float = 1366  # Earth bolometric flux [W/m2]
    Ms: float = 1.98847e30  # Solar mass [kg]
    Rs: float = 6.957e8  # Solar radius [m]
    Ls: float = 3.828e26  # Solar luminosity [W]
    Mjup: float = 1.898e27  # Jupiter mass [kg]
    Rjup: float = 7.1492e7  # Jupiter radius [m]
    M_N2: float = 0.028014  # molar mass N2 [kg/mol]
    M_H2: float = 0.002016  # molar mass H2 [kg/mol]
    M_Fe: float = 0.055845  # molar mass of Fe [kg/mol]
    M_H2O: float = 0.018015  # molar mass of H2O [kg/mol]
    vsmow: float = 1.5574e-4  # Vienna Standard Mean Ocean Water (D/H for Earth's oceans)
    n_OperTO: float = 7.83e22  # mols O per TO
    Venus_TO: float = 1.32e-5  # number of terrestrial oceans in Venus' atmosphere (needs checking)

    @property
    def Fe_xuv(self):
        """Earth XUV flux [W/m2] [Ribas et al. 2005]"""
        return 3.88 * self.cgs2si_flux

    @property
    def J2E_mass(self):
        """Convert Jupiter mass to Earth mass [kg]"""
        return self.Mjup / self.Me

    @property
    def J2E_rad(self):
        """Convert Jupiter radius to Earth radius [m]"""
        return self.Rjup / self.Re

    @property
    def mu_H2(self):
        """Molecular mass of H2 [kg/molecule]"""
        return self.M_H2 / self.avogadro

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


const: PhysicalConstants = PhysicalConstants()
