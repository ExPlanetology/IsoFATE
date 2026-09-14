'''
Collin Cherubim
October 2, 2022
Frequently used physical constants
'''
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicalConstants:
    inv_cm2m: float = 100 # convert inverse cm to inverse m
    avogadro: float = 6.022e23 # Avogadro's number [particles/mole]
    s2day: float = 1/(3600*24) # convert s to day
    s2yr: float = 1/(3600*24*365) # convert s to year
    au2m: float = 1.496e11 # convert au to m
    cgs2si_flux: float = 1/1000 # convert flux from erg/cm2/s to W/m2
    erg2joule: float = 1e-7 # convert ergs to Joules
    gcm2kgm: float = 1000 # convert g/cm3 to kg/m3 (SI)
    J2E_mass: float = 1.898e27/5.972e24 # convert Jupiter mass to Earth mass [kg]
    J2E_rad: float = 7.1492e7/6.3781e6 # convert Jupiter radius to Earth radius [m]
    R_gas: float = 8.314462 # gas constant [J/mol/K]
    kb: float = 1.38e-23 # Boltzmann constant [m2 kg/s2 K]
    sbc: float = 5.67e-8 # Stefan Boltzmann constant W/m2/K4
    g_e: float = 9.8 # grav field strength [m/s/s]
    G: float = 6.6743e-11 # [m3/kg/s2]
    h: float = 6.62607015e-34 # [J s]
    c: float = 2.99792458e8 # [m/s]
    Re: float = 6.378e6 # Earth radius [m]
    Me: float = 5.9722e24 # Earth mass [kg]
    Fe: float = 1366 # Earth bolometric flux [W/m2]
    Fe_xuv: float = field(init=False) # Earth XUV flux [W/m2] [Ribas et al. 2005]
    Ms: float = 1.98847e30 # Solar mass [kg]
    Rs: float = 6.957e8 # Solar radius [m]
    Ls: float = 3.828e26 # Solar luminosity [W]
    Mjup: float = 1.898e27 # Jupiter mass [kg]
    Rjup: float = 7.1492e7 # Jupiter radius [m]
    M_N2: float = 0.028014 # molar mass N2 [kg/mol]
    M_H2: float = 0.002016 # molar mass H2 [kg/mol]
    M_HD: float = 0.003024 # molar mass HD [kg/mol]
    M_D: float = 0.002014 # molar mass of D [kg/mol]
    M_H: float = 0.001008 # molar mass of H [kg/mol]
    M_He: float = 0.0040026 # molar mass of He [kg/mol]
    M_O: float = 0.015999 # molar mass of O [kg/mol]
    M_C: float = 0.012011 # molar mass of C [kg/mol]
    M_N: float = 0.014007 # molar mass of N [kg/mol]
    M_S: float = 0.032065 # molar mass of S [kg/mol]
    M_Fe: float = 0.055845 # molar mass of Fe [kg/mol]
    M_H2O: float = 0.018015 # molar mass of H2O [kg/mol]
    mu_H2: float = field(init=False) # molecular mass of H2 [kg/molecule]
    mu_HD: float = field(init=False) # molecular mass of HD [kg/molecule]
    mu_H: float = field(init=False) # atomic mass of H [kg/atom]
    mu_D: float = field(init=False) # atomic mass of D [kg/atom]
    mu_He: float = field(init=False) # atomic mass of He [kg/atom]
    mu_O: float = field(init=False) # atomic mass of O [kg/atom]
    mu_C: float = field(init=False) # atomic mass of C [kg/atom]
    mu_N: float = field(init=False) # atomic mass of N [kg/atom]
    mu_S: float = field(init=False) # atomic mass of S [kg/atom]
    mu_Fe: float = field(init=False) # atomic mass of Fe [kg/atom]
    vsmow: float = 1.5574e-4 # Vienna Standard Mean Ocean Water (D/H for Earth's oceans)
    DtoH_solar: float = 0.0000194 # D/H solar mole ratio from Lodders 2003
    DtoH_solar_mass: float = field(init=False) # D/H solar mass ratio from Lodders 2003
    HetoH_solar: float = 0.07991 # He/H solar mole ratio from Lodders 2003 (0.2377/0.7491*mu_H/mu_He)
    HetoH_protosolar: float = 0.09709 # He/H proto-solar mole ratio Lodders 2003 (0.2741/0.711*mu_H/mu_He)
    HetoH_protosolar_mass: float = 0.38551 # He/H proto-solar mass ratio Lodders 2003 (0.2741/0.711)
    OtoH_protosolar: float = 0.00058 # O/H proto-solar mole ratio from Lodders 2003 Table 2 (1.413e7/2.431e10)
    OtoH_protosolar_mass: float = field(init=False) # O/H proto-solar mass ratio from Lodders 2003
    CtoH_protosolar: float = 0.00029 # C/H proto-solar mole ratio from Lodders 2003 Table 2
    CtoH_protosolar_mass: float = field(init=False) # O/H proto-solar mass ratio from Lodders 2003
    NtoH_protosolar: float = 0.000080 # N/H proto-solar mole ratio from Lodders 2003 Table 2 (1.950e6/2.431e10)
    NtoH_protosolar_mass: float = field(init=False) # O/H proto-solar mass ratio
    StoH_protosolar: float = 0.000018 # S/H proto-solar mole ratio from Lodders 2003 Table 2 (4.449e5/2.431e10)
    StoH_protosolar_mass: float = field(init=False) # O/H proto-solar mass ratio
    mu_HHe: float = field(init=False) # H/He with solar abundances
    mu_H2He: float = field(init=False) # H2/He with solar abundances
    mu_solar: float = field(init=False) # average particle mass for solar metallicity
    n_OperTO: float = 7.83e22 # mols O per TO
    Venus_TO: float = 1.32e-5 # number of terrestrial oceans in Venus' atmosphere (needs checking)

    def __post_init__(self):
        object.__setattr__(self, 'Fe_xuv', 3.88*self.cgs2si_flux)
        object.__setattr__(self, 'mu_H2', self.M_H2/self.avogadro)
        object.__setattr__(self, 'mu_HD', self.M_HD/self.avogadro)
        object.__setattr__(self, 'mu_H', self.M_H/self.avogadro)
        object.__setattr__(self, 'mu_D', self.M_D/self.avogadro)
        object.__setattr__(self, 'mu_He', self.M_He/self.avogadro)
        object.__setattr__(self, 'mu_O', self.M_O/self.avogadro)
        object.__setattr__(self, 'mu_C', self.M_C/self.avogadro)
        object.__setattr__(self, 'mu_N', self.M_N/self.avogadro)
        object.__setattr__(self, 'mu_S', self.M_S/self.avogadro)
        object.__setattr__(self, 'mu_Fe', self.M_Fe/self.avogadro)
        object.__setattr__(self, 'DtoH_solar_mass', self.DtoH_solar*(self.mu_D/self.mu_H))
        object.__setattr__(self, 'OtoH_protosolar_mass', self.OtoH_protosolar*(self.mu_O/self.mu_H))
        object.__setattr__(self, 'CtoH_protosolar_mass', self.CtoH_protosolar*(self.mu_C/self.mu_H))
        object.__setattr__(self, 'NtoH_protosolar_mass', self.NtoH_protosolar*(self.mu_N/self.mu_H))
        object.__setattr__(self, 'StoH_protosolar_mass', self.StoH_protosolar*(self.mu_S/self.mu_H))
        object.__setattr__(self, 'mu_HHe', 0.00122/self.avogadro)
        object.__setattr__(self, 'mu_H2He', 0.00227/self.avogadro)
        object.__setattr__(self, 'mu_solar', 0.00235/self.avogadro)


const = PhysicalConstants()


class BinaryDiffusionCoefficients:
    '''Temperature-dependent binary diffusion coefficients, each of the form
    prefactor * T**exponent [molecules/m/s].'''

    def b_H_D(self, T):
        """Binary diffusion coefficient from Genda & Ikoma 2008 for D in H (not measured directly)."""
        return 7.183e19*T**0.728

    def b_H_He(self, T):
        """Binary diffusion coefficient from Mason & Marrero 1970 for H in He."""
        return 1.04e20*T**0.732

    def b_He_D(self, T):
        """Binary diffusion coefficient approximated from b_H_D using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 5.087e19*T**0.728

    def b_H_O(self, T):
        """Binary diffusion coefficient from Wordsworth et al 2018."""
        return 4.8e19*T**0.75

    def b_He_O(self, T):
        """Binary diffusion coefficient approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 2.61e19*T**0.75

    def b_H_C(self, T):
        """Binary diffusion coefficient approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 4.85e19*T**0.75

    def b_He_C(self, T):
        """Binary diffusion coefficient approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 2.64e19*T**0.75

    def b_H_N(self, T):
        """Binary diffusion coefficient approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 4.85e19*T**0.75

    def b_He_N(self, T):
        """Binary diffusion coefficient approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 2.65e19*T**0.75

    def b_H_S(self, T):
        """Binary diffusion coefficient approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 4.73e19*T**0.75

    def b_He_S(self, T):
        """Binary diffusion coefficient approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)."""
        return 2.48e19*T**0.75


binary_diffusion = BinaryDiffusionCoefficients()
