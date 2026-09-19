"""
Collin Cherubim
June 6, 2024
IsoFATE+Atmodeller coupler functions
"""

# for melt fraction data
import importlib.resources

import numpy.random
from scipy.interpolate import RegularGridInterpolator as RGI

# import imports
from isofate.constants import const
from isofate.orbit_params import *

_MELT_FRACTION_INTERPOLATOR = None

# incident XUV flux
# NOTE: Fxuv (the base power-law XUV flux model) moved to isofate.escape, alongside phi_E/phi_RR
# which are its only callers.


def Fxuv_a(t, F0, t_sat=5e8, beta=-1.23):
    """
    Calculates incident XUV flux
    Adapted from Ribas et al 2005
    Consistent with empirical data from MUSCLES spectra for early M dwarfs

    Inputs:
        - t: time/age [s; array]
        - F0: initial incident XUV flux [W/m2]
        - t_sat: saturation time [yr]; change this for different stellar types (M1:500Myr, G:50Myr)
        - beta: exponential term [ndim]
    Output: incident XUV flux [W/m2]
    """
    output = np.zeros(len(t))

    for i in range(len(t)):
        if t[i] * const.s2yr < t_sat:
            output[i] = F0
        else:
            output[i] = F0 * (t[i] * const.s2yr / t_sat) ** beta

    return output


def Fxuv_SF(t):
    """
    Calculates incident XUV flux
    Adapted from Sanz-Forcada et al 2011 semi-empirical study for M to F stars

    Inputs:
       - t: time/age [s]
       - d: orbital distance [m]
    Output: incident XUV flux [W/m2]
    """
    L_EUV = 10 ** (22.12 - 1.24 * np.log10(t * const.s2yr / 1e9))
    return L_EUV


def Fxuv_Ribas(t):
    """
    Calculates XUV luminosity
    Adapted from Ribas et al 2005 Sun in time program (eq 1)

    Inputs:
       - t: time/age [s]
    Output: incident XUV flux [W/m2]
    """
    tau = t * const.s2yr / 1e9
    F = 29.7 * tau**-1.23
    F = F * const.cgs2si_flux  # [W/m2]
    L = F * 4 * np.pi * const.au2m**2
    return L  # [W]


# NOTE: Fxuv_hazmat, phi_E, phiE_CP moved to isofate.escape (see also the Fxuv note above).

# NOTE: Phi_1_2, Phi_D_GC23, Phi_D_Z90_mod, Phi_D_Z90_mod2, Phi_minor_species, and
# EscapeNumberFlux moved to isofate.escape_fractionation. Phi_D_Z90/Phi_O_Z90/Phi_C_Z90/
# Phi_N_Z90/Phi_S_Z90 also moved there and were later removed entirely, once fully
# superseded by EscapeNumberFlux.get_number_flux (which calls Phi_minor_species directly).

#####_____ Lopez & Fortney 2014 thermal evolution equations _____#####

# planetary rocky-component radius


def R_rocky(Mp):
    """
    Calculates the radius of the planet's rocky (condensed-matter) component, excluding any
    gaseous envelope. Adapted from Lopez & Fortney 2014.

    Input: planetary mass [kg]
    Output: rocky-component radius [m]
    """
    return const.Re * (Mp / const.Me) ** 0.25  # Re not in paper, typo


# planetary atmosphere radius


def R_atm(Teq, Mp, R_rocky, R_env, mu):
    """
    Calculates radius of radiative atmosphere above RCB (stratosphere)
    Adapted from Lopez & Fortney 2014

    Inputs:
        - Teq: planet equilibrium temperature [K]
        - Mp: planet mass [kg]
        - R_rocky: rocky-component radius [m]
        - Renv: envelope radius [m]
        - mu: mean molecular mass [kg/particle]

    Outputs: radiative atmosphere radius [m]
    """
    g = const.G * Mp / ((R_rocky + R_env) ** 2)  # field strength at base of atm
    H = const.kb * Teq / (g * mu)  # scale height
    return 9 * H


# planetary envelope radius


def R_env(Mp, f_env, Fp, age, thermal=True):
    """
    Calculates radius of lower convective envelope (troposphere)
    Adapted from Lopez & Fortney 2014
    R_env = R_p - R_rocky - R_atm

    Inputs:
      - Mp: planet mass [kg]
      - f_env: envelope mass fraction [ndim]
      - Fp: incident bolometric flux [W/m2]
      - age: age [s]
      - thermal: toggles radius dependence on thermal evolution [True/False]

    Outputs: R_env: radius of the H/He envelope [m]
    """
    c1 = Mp / const.Me  # Me = Earth mass [kg]
    c2 = f_env / 0.05
    c3 = Fp / const.Fe  # Fe = Earth incident bolometric flux [W/m2]
    if thermal == True:
        c4 = age * const.s2yr / 5e9
    elif thermal == False:
        c4 = 1
    R_env = 2.06 * const.Re * c1 ** (-0.21) * c2 ** (0.59) * c3 ** (0.044) * c4 ** (-0.18)
    return R_env


# atmospheric mass fraction


def f_env(R_rocky, R_env, Rp, Mp, Teq, mu, Fp, t, thermal=True):
    """
    Calculates planetary atmospheric mass fraction
    by rearrangement of R_rocky, R_atm, and R_env equations
    Adapted from Lopez & Fortney 2014

    Inputs:
      - R_rocky: planet rocky-component radius [m]
      - R_env: planet envelope radius (convective part) [m]
      - Rp: total planet radius [m]
      - Mp: planet mass [kg]
      - Teq: planet equilibrium temperature [K]
      - mu: mean molecular mass [kg/particle]
      - Fp: incident bolometric flux [W/m2]
      - t: age [s]
      - thermal: toggles radius dependence on thermal evolution [True/False]

    Outputs: f_env: mass fraction of the H/He envelope relative to total mass [ndim]
    """
    t = t * const.s2yr / 1e9
    if thermal == True:
        return np.real(
            0.05
            * (
                (Rp - const.Re * (Mp / const.Me) ** 0.25 - R_atm(Teq, Mp, R_rocky, R_env, mu))
                * (1 / (2.06 * const.Re))
                * (Mp / const.Me) ** (0.21)
                * (Fp / const.Fe) ** (-0.044)
                * (t / 5) ** (0.18)
            )
            ** (1 / 0.59)
        )
    elif thermal == False:
        return np.real(
            0.05
            * (
                (Rp - const.Re * (Mp / const.Me) ** 0.25 - R_atm(Teq, Mp, R_rocky, R_env, mu))
                * (1 / (2.06 * const.Re))
                * (Mp / const.Me) ** (0.21)
                * (Fp / const.Fe) ** (-0.044)
            )
            ** (1 / 0.59)
        )


# in-house planetary radius calculation


def R_grid(Mp, f_atm, Teq, mu, k, n_tot=int(1e4)):
    """
    Integrates radius over pressure grid for dry adiabat
    Inputs:
        - Mp: planet mass [kg]
        - f_atm: atmospheric mass fraction [ndim]
        - Teq: planetary equilibrium temperature [K]
        - mu: mean atomic mass [kg]
        - k: ratio of specific gas constant to specific heat capacity [ndim]
        - n_tot: grid size
    Output: radius grid of length n_tot [m]
    """
    P_rcb = 1e4  # pressure at radiative-convective boundary where planetary radius is defined [Pa; 0.1 bar]
    P_s = P_surf(f_atm, Mp)  # surface pressure, thin atm approx
    T_s = Teq * (P_s / P_rcb) ** k
    P_a = np.logspace(
        np.log10(P_s), np.log10(P_rcb), n_tot
    )  # pressure grid for radius calculation
    R = const.R_gas / mu / const.avogadro  # specific gas constant
    Rp_grid = np.zeros(n_tot)
    Rp = R_rocky(Mp)
    Rp_grid[0] = Rp
    T_a = np.zeros(n_tot)
    T_a[0] = T_s
    for i in range(n_tot - 1):
        T = np.max([T_s * (P_a[i + 1] / P_s) ** k, Teq])
        rho = P_a[i + 1] / R / T
        g = const.G * Mp / Rp_grid[i] ** 2
        dP = P_a[i] - P_a[i + 1]  # use for log spaced pressure grid
        dRp = dP / (rho * g)  # Euler method; barometric law (hydrostatic)
        Rp += dRp
        Rp_grid[i + 1] = Rp
        T_a[i + 1] = T
    return Rp_grid


def R_rcb(Mp, fatm, mu, Tsurf, Rgas=const.kb / const.mu_H2, cp=14514, Prcb=1e4):
    """
    Calculates planetary radius at Prcb from first principles
    assuming dry adiabat and hydrostatic balance
    Inputs:
        - Mp: planet mass [kg]
        - fatm: atmospheric mass fraction [ndim]
        - mu: average atomic mass of atmospheric species [kg]
        - Tsurf: surface temperature [K]
        - Rgas: specific gas constant [J/kg/K]
        - cp: specific heat capacity [J/kg/K] (default value for H2 on NIST database; He at all T: 20.79 J/mol K)
        - Prcb: atmospheric pressure at RCB [Pa] (estimated at 0.1 bar, Robinson & Catling 2012)
    Output: planetary radius at Prcb [m]
    """
    Rc = R_rocky(Mp)  # rocky-component radius [m]
    Ps = P_surf(Mp, fatm)  # surface pressure [Pa]
    k = Rgas / cp  # for dry adiabat
    rcb = Rc + (const.G * Mp * mu / const.kb) * (k * Ps**k / Tsurf) / (Prcb**k - Ps**k)
    return rcb


# Computes number of terrestrial oceans based on planet mass for setting lower bound on N_H to end simulation


def TO(Mp, f_atm="null", n_TO="null"):
    """
    Calculates the hydrogen atom number from final desired f_atm or number of terrestrial oceans.
    If f_atm is specificed, fn computes H remaining assuming all is bonded to oxygen in envelope
    assuming solar abundance. If n_TO is specified, fn computes the same for the given value of n_TO.
    All values assume solar abundance values from Lodders 2003
    *** Specify only one: f_atm OR n_TO!
    Inputs:
        - Mp: planet mass [kg]
        - f_atm: final envelope mass fraction [ndim]
        - n_TO: number of terrestrial oceans remaining on planet
    Output:
        - array[0] = number of hydrogen atoms remaining on planet
        - array[1] = number of terrestrial oceans remaining on planet
    """

    n_OperTO = 7.83e22  # mols O per TO

    if n_TO == "null" and f_atm == "null":
        print("Error: Must specify value for either f_atm or n_TO")
        return None

    elif n_TO == "null":
        n_H = 0.7491 * Mp * f_atm / const.M_H  # mols of H in envelope
        n_O = 4.899e-4 * n_H  # mols of O in envelope (Lodders solar abundance)
        n_TO = n_O / n_OperTO  # number of terrestrial oceans worth of oxygen in system
        N_H = 2 * n_O * const.avogadro  # atoms of H in envelope

    elif f_atm == "null":
        n_O = n_TO * n_OperTO  # mols of O in envelope
        N_H = 2 * n_O * const.avogadro  # atoms of H in envelope

    return np.array([N_H, n_TO])


def WMF(Mp, wmf):
    """
    Calculates number of terrestrial oceans on a planet for given planet mass and water mass fraction
    Inputs:
        - Mp: planet mass [kg]
        - wmf: water mass fraction [ndim]
    Output: number of terrestrial oceans [float]
    """
    return Mp * wmf / 0.018 / const.n_OperTO


def R_Bondi(Mp, mu, Teq, gamma=7 / 5):
    """
    Bondi radius calculation

    Inputs:
        - gamma: adiabatic index (heat capacity ratio) [ndim]
        - Teq: planetary equilibrium temperature [K]
        - Mp: planetary mass [kg]
        - mu: average particle mass [kg]
    Output: Bondi radius [m]
    """
    R_B = (gamma - 1) * const.G * Mp * mu / (gamma * const.kb * Teq)  # Bondi radius
    return R_B


def R_Hill(Mp, Mstar, a):
    """
    Hill radius calculation

    Inputs:
        - Mp: planetary mass [kg]
        - Mstar: stellar mass [kg]
        - a: orbital distance [m]
    Output: Hill radius [m]
    """
    R_H = a * (Mp / 3 / Mstar) ** (1 / 3)
    return R_H


# NOTE: phi_kill moved to isofate.escape. F0_kill below is dead code (never called anywhere in
# the repo) and still references the bare name `phi_kill`, now undefined in this module - since
# Python only resolves that name at call time and F0_kill is never called, this doesn't break
# import or tests. If F0_kill is ever revived, update it to `from isofate.escape import
# phi_kill` locally inside the function (not a module-level import - that would create a
# circular import, since isofate.escape already imports R_rocky from this module).
def F0_kill(Mp, Rp, M_atm, age, eps=0.15):
    C = 0.893818  # integral of power law portion of F_XUV function
    Vpot = const.G * Mp / Rp
    phi = phi_kill(M_atm, Rp, age)
    return 40 * Vpot * phi / eps - 2 * C


def b_H2_HD(T):
    """
    Input: T, temperature [K]
    Output: Binary diffusion coefficient for H2 in HD from Genda 2008 [molecules/m/s]
    """
    return 4.48e19 * T**0.75


# NOTE: phi_RR moved to isofate.escape.


# NOTE: dead code (never called anywhere in the repo); still references the bare name `Fxuv`,
# now undefined in this module since Fxuv moved to isofate.escape - harmless since it's never
# called (same situation as F0_kill above).
def phi_RMC(t, F0, t_sat, Rp):
    F = Fxuv(t, F0, t_sat)
    phi = 4e9 * np.sqrt(F / (5e5 * const.cgs2si_flux)) / (4 * np.pi * Rp**2)
    return phi


def Rp_prim(Mp, f_atm, Fp, t0, T, mu, M_star, d):
    """
    Calculates primordial planet radius
    Takes minimum of Lopez/Fortney 2014 calculation, Hill radius, Bondi radius
    Inputs:
     - Mp: planet mass [kg]
     - f_atm: atm mass fraction [ndim]
     - Fp: bolometric incident planetary flux [W/m2]
     - t0: simulation start time [s]
     - T: planet eq temp [K]
     - M_star: stellar mass [kg]
     - d: orbital distance [m]
    Output: planet radius [m]
    """
    r_core = R_rocky(Mp)
    r_env = R_env(Mp, f_atm, Fp, t0)
    r_atm = R_atm(T, Mp, r_core, r_env, mu)
    R_LF = r_core + r_env + r_atm
    R_B = R_Bondi(Mp, mu, T)
    R_H = R_Hill(Mp, M_star, d)
    if type(R_LF) != "int":
        Rp = np.zeros(len(R_LF))
        for i in range(len(R_LF)):
            Rp[i] = np.min([R_LF[i], R_B[i], R_H[i]])
    else:
        Rp = np.min([R_LF, R_B, R_H])
    return Rp


def epsilon(Mp, Rp):
    v_esc = np.sqrt(2 * const.G * Mp / Rp)
    eps = 0.1 * (v_esc / 15e3) ** (-2)
    return eps


def radius_valley(P, Rp, upper, lower):
    """
    Checks if planet falls in the "fractionation valley," near the radius valley
    Inputs:
     - P: orbital period [days]
     - Rp: planetary radius [Earth radii]
     - upper: tuple or array with upper[0] = slopes and upper[1] = intercept for upper limit of valley
     - lower: tuple or array with lower[0] = slopes and lower[1] = intercept for lower limit of valley
    """
    if Rp < 10 ** (upper[0] * np.log10(P) + upper[1]) and Rp > 10 ** (
        lower[0] * np.log10(P) + lower[1]
    ):
        return True
    else:
        return False


def Johnson_reduction(eps, F_xuv, Rp, Vpot):
    """
    Energy-limited escape rate reduction factor due to thermal and translational energy loss (Johnson et al 2013)
    Used in Hu et al 2015 and Malsky & Rogers 2020
    Inputs:
     - eps: heating efficiency factor [ndim]
     - F_xuv: Incident planetary XUV flux [W/m2]
     - Rp: planetary radius [m]
     - Vpot: planetary gravitational potential [J/kg]
    Output: escape flux reduction factor, f_r
    """
    # Q_net = eps*L_xuv*Rp**2/4/a**2
    Q_net = eps * F_xuv * np.pi * Rp**2
    U = Vpot * const.mu_HHe  # grav potential energy
    gamma = (
        5 / 3
    )  # [ndim] adiabatic index; Malsky & Rogers 2020 uses 5/3, Gupta & Schlichting use 7/5 in CPML
    sigma = 5e-20  # [m2] collisional cross section from Malsky & Rogers 2020
    Kn = 1  # [ndim] Knudsen number from Malsky & Rogers 2020
    Q_c = (4 * np.pi * Rp * gamma * U / sigma / Kn) * np.sqrt(2 * U / const.mu_HHe)
    f_r = Q_c / Q_net
    if f_r < 1:
        # return f_r, Q_net, Q_c # use for johnson_reduction.py script
        return f_r
    elif f_r >= 1:
        # return 1, Q_net, Q_c
        return 1


def f_atm_pred(Mc):
    """
    Predicts f_atm from planet core mass based on
    models of gas accretion, boil off and disk dispersal.
    Used in Ginzburg et al 2016, Gupta+S 2019, Gupta et al 2022
    Input: planet core mass [kg]
    Output: atmopsheric mass fraction [ndim]
    """
    return 0.05 * np.sqrt(Mc / const.Me)


def f_atm_pred2(Mc, Teq):
    """
    Predicts f_atm at time of disk dispersal from planet core mass and Teq based on
    models of gas accretion, boil off and disk dispersal from Ginzburg et al 2016 (eq 18)
    Input: planet core mass [kg], equilibrium temperature [K]
    Output: atmopsheric mass fraction [ndim]
    """
    return 0.02 * (Mc / const.Me) ** 0.8 * (Teq / 1e3) ** (-0.25)


def f_atm_pred2_alt(Mc, P, T_star, M_star, R_star, distribution=False, sigma_fraction=0.3):
    """
    Predicts f_atm at time of disk dispersal from planet core mass and Teq based on
    models of gas accretion, boil off and disk dispersal from Ginzburg et al 2016 (eq 18)
    Input: planet core mass [kg], equilibrium temperature [K]
    Output: atmopsheric mass fraction [ndim]
    """

    L = Luminosity(R_star, T_star)
    smax = SemiMajor(M_star, P)
    Fp = Insolation(L, smax)
    Teq = EqTemp(Fp)
    if distribution == False:
        return 0.02 * (Mc / const.Me) ** 0.8 * (Teq / 1e3) ** (-0.25)
    else:
        f_atm_mean = 0.02 * (Mc / const.Me) ** 0.8 * (Teq / 1e3) ** (-0.25)
        f_atm = numpy.random.normal(f_atm_mean, sigma_fraction * f_atm_mean)
        return np.clip(f_atm, 0.001, None)


def f_atm_pred3(Mc, Teq):
    """
    Predicts f_atm after disk dispersal\outer layer blow-off from planet core mass and Teq based on
    models of gas accretion, boil off and disk dispersal from Ginzburg et al 2016 (eq 24)
    Input: planet core mass [kg], equilibrium temperature [K]
    Output: atmopsheric mass fraction [ndim]
    """
    return 0.01 * (Mc / const.Me) ** 0.44 * (Teq / 1e3) ** (0.25)


def NtoM(N1, N2, MM1, MM2):
    """
    Converts molar concentration to mass concentration:
    N2/(N1 + N2) --> m2/(m1 + m2)
    Inputs:
        - N1: moles of species 1 [mol or particles]
        - N2: moles of species 2 [mol or particles]
        - MM1: molar mass of species 1 [kg/mol]
        - MM2: molar mass of species 2 [kg/mol]
    Output: Mass concentration
    """

    return N2 * MM2 / (N1 * MM1 + N2 * MM2)


def P_surf(Mp, f_atm):
    """
    Calculates planetary surface pressure
    Inputs:
        - Mp: planet mass [kg]
        - f_atm: atmospheric mass fraction [ndim]
    Output: surface pressure [Pa]
    """
    M_atm = Mp * f_atm
    Rcore = R_rocky(Mp)
    grav = const.G * Mp / Rcore**2
    area = 4 * np.pi * Rcore**2
    P = grav * M_atm / area
    return P


def T_surf(Teq, Mp, fatm, R=const.kb / const.mu_H2, cp=14514, Peq=1e4):
    """
    Calculates planetary surface temperature assuming dry adiabat
    Inputs:
        - Teq: equilibrium temperature [K]
        - Mp: planetary mass [kg]
        - fatm: atmospheric mass fraction [ndim]
        - R: specific gas constant [J/kg/K]
        - cp: specific heat capacity [J/kg/K] (default value for H2 on NIST database; He at all T: 20.79 J/mol K)
        - Peq: atmospheric pressure at RCB [Pa] (estimated at 0.1 bar, Robinson & Catling 2012)
    Output: surface temperature [K]
    """
    k = R / cp
    Rc = R_rocky(Mp)
    Ts = (Teq / Peq**k) * (const.G * Mp**2 * fatm / (4 * np.pi * Rc**4)) ** k
    return Ts


def SpecHeatCap(T):
    if T < 1000:
        A = 33.066178
        B = -11.363417
        C = 11.432816
        D = -2.772874
        E = -0.158558
    elif 1000 <= T < 2500:
        A = 18.563083
        B = 12.257357
        C = -2.859786
        D = 0.268238
        E = 1.977990
    elif 2500 <= T <= 6000:
        A = 43.413560
        B = -4.293079
        C = 1.272428
        D = -0.096876
        E = -20.533862
    elif T > 6000:
        return 42
    t = T / 1000
    return A + B * t + C * t**2 + D * t**3 + E / t**2


def MeltFraction(Mp, T):
    """
    Calculate mantle melt fraction Ψ using a pre-computed grid.
    This function uses a lazy-loading pattern to ensure the data grid
    and interpolator are loaded from disk only once.
    Input:
        - Mp: planet mass [kg]
        - T: surface temperature [K]
    Output:
        - melt fraction of the silicate layer [ndim]
    """
    global _MELT_FRACTION_INTERPOLATOR
    # On the first call, the variable will be None. Do the one-time setup.
    if _MELT_FRACTION_INTERPOLATOR is None:
        # with importlib.resources.files('isofate.data').joinpath('melt_fraction_grid.npz') as data_path:
        with importlib.resources.as_file(
            importlib.resources.files("isofate.data").joinpath("melt_fraction_grid.npz")
        ) as data_path:
            data = np.load(data_path)
            temp_grid = data["temp_grid"]
            mass_grid = data["mass_grid"]
            psi_grid = data["psi_grid"]

        # Create the interpolator object once and store it in our module-level variable.
        _MELT_FRACTION_INTERPOLATOR = RGI(
            points=(mass_grid, temp_grid),  # Note: RGI expects a tuple of points
            values=psi_grid.transpose(),
            method="linear",
        )
    return _MELT_FRACTION_INTERPOLATOR((Mp / const.Me, T))


def TSM(Rp, Teq, Mp, Rstar, m_J, scale_factor=1.26):
    """
    Calculates transmission spectroscopy metric from Kepmton et al 2018
    Inputs:
        - Rp: planet radius [Rearth]
        - Teq: eq temp assuming zero albedo
        - Mp: planet mass [Mearth]
        - Rstar: stellar radius [Rsun]
        - m_J: J-band mag [ndim]
    Output: TSM score [ndim]
    """
    return scale_factor * Rp**3 * Teq * 10 ** (-m_J / 5) / (Mp * Rstar**2)


#####_____ANALYTIC SOLUTIONS_____#####

# for phi < phi_c


def x2_subcrit(x2_0, tau, t):
    return x2_0 / (1 - t / tau)


# for phi > phi_c


def x2_supercrit(x2_0, Mp, Rp, T, phi_1, tau, t):
    # phi_1 must be in particles/m2/s
    m1 = const.M_H2 / const.avogadro  # kg/particle
    m2 = const.M_HD / const.avogadro  # kg/particle
    b = 4.48e19 * T**0.75  # [molecules/m/s] from Genda 2008 for H2 in HD
    g = const.G * Mp / Rp**2  # N/kg
    gamma = (m2 - m1) * b * g / (const.kb * T * phi_1)  # ndim
    print(gamma)
    return x2_0 / (1 - (t / tau)) ** gamma
