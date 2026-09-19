# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Escape-mechanism classes and their underlying physics functions.

Each `EscapeMechanism` subclass owns its own tuning constants as explicit fields (not an
`IsocalcOptions` instance) and turns one timestep's physical state (`EscapeState`) into an
atmospheric mass flux [kg/m2/s]. `isocalc()` (isofate_coupler.py) is constructed with one
`EscapeMechanism` instance and calls its `compute_mass_flux` once per timestep, in place of
the old `options.mechanism`/`options.RR` string dispatch.
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from isofate import override
from isofate.constants import const
from isofate.isofunks import R_rocky
from isofate.utils import gravitational_acceleration


class EscapeState(eqx.Module):
    """Per-timestep physical state handed to `EscapeMechanism.compute_mass_flux`.

    Config constants (eps, t0, rho_rcb, ...) are NOT here - they live on the
    `EscapeMechanism` instance itself, baked in at construction.
    """

    radius_p: ArrayLike
    """Total planet radius [m]."""
    Mp: ArrayLike
    """Planet mass [kg]."""
    T: ArrayLike
    """Equilibrium temperature [K]."""
    F0: ArrayLike
    """Initial incident XUV flux [W/m2]."""
    Vpot: ArrayLike
    """Gravitational potential at the outer layer [J/kg]."""
    d: ArrayLike
    """Orbital distance [m]."""
    A: ArrayLike
    """Planet surface area [m2]."""
    mu: ArrayLike
    """Mean atmospheric particle mass [kg]."""
    radius_env: ArrayLike
    """Envelope radius [m]."""
    f_atm: ArrayLike
    """Atmospheric mass fraction [ndim]."""
    t_now: ArrayLike
    """Current simulation time [s]."""
    t_total: ArrayLike
    """Total simulation time [s]."""


class EscapeMechanism(eqx.Module):
    """Base class for atmospheric-escape mechanisms."""

    @abstractmethod
    def compute_mass_flux(self, state: EscapeState) -> ArrayLike:
        """Atmospheric mass flux [kg/m2/s] for this mechanism, at the given timestep state."""


# ---- shared XUV-flux-model helpers -----------------------------------------------------------


def Fxuv(t, F0, t0=1e6, t_sat=5e8, beta=-1.23, step_fn=False, F_final=0, t_pms=0, pms_factor=1e2):
    """
    Calculates incident XUV flux
    Adapted from Ribas et al 2005
    Consistent with empirical data from MUSCLES spectra for early M dwarfs

    Inputs:
        - t: time/age [s] - traced
        - F0: initial main sequence incident XUV flux [W/m2] - traced
        - t0: start time [yr] - static (fixed per XUVEscape instance, confirmed constant across
          isocalc's per-timestep loop - see isofate_coupler.py)
        - t_sat: saturation time [yr]; change this for different stellar types (M1:500Myr, G:50Myr) - static
        - beta: exponential term [ndim] - static
        - step_fn: True for step function from F0 to F_final [Bool] - static
        - F_final: if step_fn == True, set the final XUV flux [W/m2] - static
        - t_pms: pre-main sequence phase duration (power law decay) [yr] - static
        - pms_factor: Fxuv_pms_0/Fxuv_sat; ~1e2 for mid-to-late M stars (Ramirez & Kaltenegger 2014) [ndim] - static

    Output: incident XUV flux [W/m2]
    """
    time = t * const.s2yr

    # t_pms is static (see Inputs above) - resolved with a plain Python `if`, same as step_fn
    # below, so log10(t_pms) is only ever evaluated when t_pms > 0 (never computed-then-discarded
    # when t_pms=0, the common pre-main-sequence-disabled default - avoids ever hitting log10(0)).
    if t_pms > 0:
        F_pms0 = F0 * pms_factor
        s = (jnp.log10(F0) - jnp.log10(F_pms0)) / (jnp.log10(t_pms) - jnp.log10(t0))
        pre_main_sequence = F_pms0 * (time / t0) ** s
        is_pre_main_sequence = (time > 0) & (time < t_pms)
    else:
        pre_main_sequence = 0.0  # never selected below (is_pre_main_sequence is statically False)
        is_pre_main_sequence = False

    saturated = F0
    post_saturation = F_final if step_fn else F0 * (time / t_sat) ** beta

    return jnp.where(
        is_pre_main_sequence,
        pre_main_sequence,
        jnp.where(time < t_sat, saturated, post_saturation),
    )


def Fxuv_Johnstone(t, d, stellar_type):
    """
    Calculates incident XUV flux
    Adapted from Johnstone et al 2021 semi-empirical XUV tracks
    Raw files available here: https://zenodo.org/records/4266670#.X6rMuq4o9H5

    Inputs:
       - t: time/age [s]
       - d: orbital distance [m]
       - stellar_type: 'M1', 'K5', 'G5' [str]
    Output: incident XUV flux [W/m2]
    """

    if stellar_type == "M1":
        path = "/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/0p5Msun_50percentile_basic.dat"
    elif stellar_type == "K5":
        path = "/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/0p7Msun_50percentile_basic.dat"
    elif stellar_type == "G5":
        path = "/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/1p0Msun_50percentile_basic.dat"

    data = np.loadtxt(path, unpack=True)
    age = data[0] * 1e6 / const.s2yr  # [s]
    L_EUV = (data[4] + data[5] + data[6]) * const.erg2joule  # [W]
    F_EUV = L_EUV / (4 * np.pi * d**2)  # [W/m2]

    return np.interp(t, age, F_EUV)


def Fxuv_hazmat(t, d, activity):
    """
    Semi-empirical XUV flux estimates based on HAZMAT and MUSCLES
    programs. Consistent with Fxuv power law approximation.

    Inputs: t (time, s); d (orbital distance, m); activity ('high', 'medium', 'low')
    Outputs: incident planetary XUV flux [W/m2]
    """
    flux_10myr_uq = 45400  # [W/m2]
    flux_10myr_med = 41681
    flux_10myr_lq = 30355
    flux_45myr_uq = 36661
    flux_45myr_med = 3661
    flux_45myr_lq = 30288
    flux_120myr_uq = 47629
    flux_120myr_med = 21601
    flux_120myr_lq = 21601
    flux_650myr_uq = 48673
    flux_650myr_med = 20213
    flux_650myr_lq = 4144
    flux_5000myr_uq = 3964
    flux_5000myr_med = 1146
    flux_5000myr_lq = 1100

    if activity == "high":
        if t * const.s2yr / 1e6 < 10:
            return flux_10myr_uq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 45:
            return flux_45myr_uq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 120:
            return flux_120myr_uq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 650:
            return flux_650myr_uq * (0.515 * const.Rs) ** 2 / (d**2)
        else:
            return flux_5000myr_uq * (0.515 * const.Rs) ** 2 / (d**2)
    elif activity == "medium":
        if t * const.s2yr / 1e6 < 10:
            return flux_10myr_med * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 45:
            return flux_45myr_med * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 120:
            return flux_120myr_med * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 650:
            return flux_650myr_med * (0.515 * const.Rs) ** 2 / (d**2)
        else:
            return flux_5000myr_med * (0.515 * const.Rs) ** 2 / (d**2)
    elif activity == "low":
        if t * const.s2yr / 1e6 < 10:
            return flux_10myr_lq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 45:
            return flux_45myr_lq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 120:
            return flux_120myr_lq * (0.515 * const.Rs) ** 2 / (d**2)
        elif t * const.s2yr / 1e6 < 650:
            return flux_650myr_lq * (0.515 * const.Rs) ** 2 / (d**2)
        else:
            return flux_5000myr_lq * (0.515 * const.Rs) ** 2 / (d**2)


# ---- per-mechanism physics functions -----------------------------------------------------------


def phi_E(
    t,
    Vpot,
    d,
    F0,
    *,
    eps=0.15,
    t0=1e6,
    t_sat=5e8,
    beta=-1.23,
    step_fn=False,
    F_final=0,
    t_pms=0,
    pms_factor=1e2,
    flux_model="power law",
    activity="medium",
    stellar_type="M1",
):
    """
    Calculates energy-limited mass flux for XUV-driven hydrodynamic escape
    Adapted from Wordsworth et al. 2018

    Inputs:
        - t: system age [s]
        - Vpot: planetary grav. potential [J/kg]
        - d: orbital distance [m]
        - F0: initial incident XUV flux [W/m2]
        - eps, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor, flux_model, activity,
          stellar_type: XUVEscape's tuning constants

    Output: mass flux [kg/m2/s]
    """
    # NOTE: I think this is the default branch that is currently being tripped for simple tests.
    if flux_model == "power law":
        return eps * Fxuv(t, F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor) / (4 * Vpot)
    # FIXME: Will probably break JAX.  To refactor.
    # elif flux_model == "phoenix":
    #    return eps * Fxuv_hazmat(t, d, activity) / (4 * Vpot)
    # FIXME: This will break JAX.  Data must be loaded outside of the JAX-traced function.
    # elif flux_model == "Johnstone":
    #    return eps * Fxuv_Johnstone(t, d, stellar_type)


def phiE_CP(Teq, Mp, rho_rcb, eps, Vpot, area, mu, R_env):
    """
    Atmospheric mass flux for core-powered mass loss scenario
    Adapted from Gupta & Schlicting 2020

    Inputs:
        - Teq: planetary equilibrium temperature [K]
        - Mp: planetary mass [kg]
        - rho_rcb: density at the RCB [kg/m3]; ref value =1 kg/m3 from eq 7 Gupta & Schlichting 2020
        - eps: heat transfer efficiency factor [ndim]
        - Vpot: planetary gravitational potential [J/kg]
        - area: planetary surface area [m2]
        - mu: average particle mass [kg]
    Output: mass flux [kg/m2/s]
    """
    R_c = R_rocky(Mp)
    V_pot = const.G * Mp / R_c
    gamma = 7 / 5  # adiabatic index for H2
    R_B = (gamma - 1) * const.G * Mp * mu / (gamma * const.kb * Teq)  # Bondi radius
    kappa = 0.01  # opacity at RCB [m2/kg]; Ginzburg et al. 2016 and eq 7 Gupta & Schlichting 2020 (Freedman et al 2008)
    L = 64 * np.pi * const.sbc * Teq**4 * R_B / (3 * kappa * rho_rcb)  # planetary core luminosity
    phi_L = L / (V_pot * area)  # mass flux

    c_s = np.sqrt(const.kb * Teq / const.mu_H)  # sound speed [m/s]
    R_rcb = R_c + R_env  # rcb radius [m]
    phi_B = (
        c_s * rho_rcb * np.exp(-const.G * Mp / (c_s**2 * R_rcb))
    )  # Bondi-limited escape (eq 10 Gupta & Schlichting 2020; eq 26 Ginzburg et al. 2016)

    return min(phi_L, phi_B)


def phi_kill(M_atm, Rp, age):
    """
    Calculates mass escape flux to ensure removal of entire atmosphere
    Use with caution, work in progress
    Inputs:
        - M_atm: atmospheric mass fraction; Mp*f_atm [ndim]
        - Rp: planet radius [m]
        - age: system age/simulation time [s]
    Output: mass escape flux [kg/m2/s]
    """
    A = 4 * np.pi * Rp**2
    return M_atm / A / age * 10


def phi_RR(
    Rp,
    Mp,
    Teq,
    t,
    F0,
    *,
    t0=1e6,
    t_sat=5e8,
    beta=-1.23,
    step_fn=False,
    F_final=0,
    t_pms=0,
    pms_factor=1e2,
):
    """
    Radiation recombination-limited escape rate used in Lopez & Rice 2018 and others
    Prescription from Murray-Clay et al 2009, and used in Wordsworth et al 2018
    Inputs:
     - Rp: radius of the XUV photosphere [m]
     - Mp: planet mass [kg]
     - Teq: planetary equilibrium temperature [K]
     - t: system age (time) [s]
     - F0: initial planetary incident XUV flux [W/m2]
     - t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor: XUVEscape's tuning constants
    Output: mass flux [kg/m2/s]
    """
    F = Fxuv(t, F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor)
    g = gravitational_acceleration(Mp, Rp)  # grav field strength at base of flow [m/s2]
    T = 1e4  # temp is thermostatted at 1e4 K by radiation [K]
    nu_0 = 4.835e15  # EUV ionizing radiation frequency (~60 nm/ 20 eV) [Hz]
    alpha_rec = (
        2.7e-13 * (Teq / 1e4) ** (-0.9) / 1e6
    )  # case B recombination coeff for H (Murray-Clay et al 2009 pg 4) [m3/atom/s]
    # H_base = kb*Teq/mu_solar/g # scale height at base of flow [m]
    # n_wind = np.sqrt(Fxuv/h/nu_0/H_base/alpha_rec) # number density at flow base [particles/m3]
    c_s = jnp.sqrt(
        2 * const.kb * T / const.mu_H
    )  # sounds speed at sonic point [m/s] Murray-Clay et al 2009
    R_s = const.G * Mp / (2 * c_s**2)  # sonic point [m] Murray-Clay et al 2009

    ### Lopez & Rice 2018 formulation
    # p1 = c_s*n_wind*mu_solar # divided both sides by 4 pi R_s^2 to change units to per area and removed negative sign
    # p2 = np.sqrt(Fxuv*G*Mp/(h*nu_0*alpha_rec*c_s**2*R_base**2))
    # p3 = np.exp((R_base/R_s - 1)*G*Mp/R_base/c_s**2)
    # return p1*p2*p3

    ### Murray-Clay et al 2009 formulation
    p1 = 4 * jnp.pi * R_s**2 * c_s
    p2 = jnp.sqrt(F * const.mu_H**3 * g / (const.h * nu_0 * alpha_rec * 2 * const.kb * Teq))
    p3 = jnp.exp((Rp / R_s - 1) * const.G * Mp / Rp / c_s**2)

    return p1 * p2 * p3 / (4 * jnp.pi * Rp**2)


# ---- concrete EscapeMechanism subclasses -------------------------------------------------------


class XUVEscape(EscapeMechanism):
    """XUV-driven hydrodynamic escape, optionally capped by radiation-recombination limiting.

    Args:
        eps: Heat transfer efficiency [ndim].
        t0: Reference start time for the XUV power-law flux model [yr]. Independent of
            `IsocalcOptions.t0` (isocalc's own simulation-start-time knob), even though they
            typically get set to the same value.
        t_sat, beta, step_fn, F_final, t_pms, pms_factor: `Fxuv` power-law shape params.
        flux_model: 'power law' | 'phoenix' | 'Johnstone'.
        activity: Only used when flux_model == 'phoenix'.
        stellar_type: Only used when flux_model == 'Johnstone'.
        RR: Whether to cap the energy-limited flux with the radiation-recombination-limited
            flux (`phi_RR`), taking the min of the two (Wordsworth et al. 2018 prescription).
    """

    eps: ArrayLike = 0.15
    t0: ArrayLike = 1e6
    t_sat: ArrayLike = 5e8
    beta: ArrayLike = -1.23
    step_fn: bool = False
    F_final: ArrayLike = 0
    t_pms: ArrayLike = 0
    pms_factor: ArrayLike = 1e2
    flux_model: str = "power law"
    activity: str = "medium"
    stellar_type: str = "M1"
    RR: bool = True

    @override
    def compute_mass_flux(self, state: EscapeState) -> ArrayLike:
        phi_energy_limited = phi_E(
            state.t_now,
            state.Vpot,
            state.d,
            state.F0,
            eps=self.eps,
            t0=self.t0,
            t_sat=self.t_sat,
            beta=self.beta,
            step_fn=self.step_fn,
            F_final=self.F_final,
            t_pms=self.t_pms,
            pms_factor=self.pms_factor,
            flux_model=self.flux_model,
            activity=self.activity,
            stellar_type=self.stellar_type,
        )
        if not self.RR:
            return phi_energy_limited
        phi_recombination_limited = phi_RR(
            state.radius_p,
            state.Mp,
            state.T,
            state.t_now,
            state.F0,
            t0=self.t0,
            t_sat=self.t_sat,
            beta=self.beta,
            step_fn=self.step_fn,
            F_final=self.F_final,
            t_pms=self.t_pms,
            pms_factor=self.pms_factor,
        )
        return jnp.minimum(phi_recombination_limited, phi_energy_limited)


class CPMLEscape(EscapeMechanism):
    """Core-powered mass loss (Gupta & Schlichting 2020).

    Args:
        eps: Heat transfer efficiency [ndim] - independent of `XUVEscape.eps`, even though
            today's shared `IsocalcOptions.eps` default (0.15) happened to be the same value
            for both mechanisms.
        rho_rcb: Gas density at the RCB [kg/m3]; ref value 1 (Gupta & Schlichting 2020 eq 7).
    """

    eps: ArrayLike = 0.15
    rho_rcb: ArrayLike = 1.0

    @override
    def compute_mass_flux(self, state: EscapeState) -> ArrayLike:
        return phiE_CP(
            state.T,
            state.Mp,
            self.rho_rcb,
            self.eps,
            state.Vpot,
            state.A,
            state.mu,
            state.radius_env,
        )


class PhiKillEscape(EscapeMechanism):
    """Forces removal of the entire atmosphere by the end of the run. No tuning constants."""

    @override
    def compute_mass_flux(self, state: EscapeState) -> ArrayLike:
        return phi_kill(state.Mp * state.f_atm, state.radius_p, state.t_total - state.t_now)


class CombinedEscape(EscapeMechanism):
    """Sums the mass flux of several component mechanisms, e.g. "XUV+CPML" ==
    `CombinedEscape((XUVEscape(...), CPMLEscape(...)))`.
    """

    components: tuple[EscapeMechanism, ...]

    @override
    def compute_mass_flux(self, state: EscapeState) -> ArrayLike:
        return sum(component.compute_mass_flux(state) for component in self.components)
