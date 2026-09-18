# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration for `isocalc`, kept in its own module (no dependency on `isofunks.py` or
`isofate_coupler.py`) so that both can import `IsocalcOptions` without a circular import:
`isofunks.py` functions (`phi_E`, `phi_RR`) take it directly, and `isofate_coupler.py` builds it.
"""

from dataclasses import dataclass

from jaxtyping import ArrayLike

from isofate.constants import const


@dataclass
class IsocalcOptions:
    """Mode switches and tuning constants for `isocalc`, fixed for the whole run.

    These are numerical/modeling choices, as opposed to `isocalc`'s other arguments (`system`,
    `F0`, `time`, and the initial `N_x` abundances), which describe the actual physical
    initial-value problem being solved and so stay direct `isocalc` arguments.

    Args:
        mechanism: One of 'XUV', 'XUV+RR', 'CPML', 'XUV+CPML', 'fix phi subcritical',
            'fix phi supercritical', 'phi kill'.
        rad_evol: Set to False to fix the planet radius at the rocky radius.
        melt_fraction_override: Fixed mantle melt fraction; if False, it is instead calculated
            from Mp and T_surface.
        mu: Average atmospheric particle mass [kg]; default is H/He solar composition. Only the
            initial value - `isocalc` recomputes it every step from the evolving abundances.
        eps: Heat transfer efficiency [ndim].
        activity: For the `Fxuv_hazmat` flux model (semi-empirical MUSCLES survey data): 'low'
            (lower quartile), 'medium' (median), or 'high' (upper quartile).
        flux_model: 'power law' for the analytic power law, 'phoenix' for `Fxuv_hazmat`,
            'Johnstone' for `Fxuv_Johnstone`.
        stellar_type: 'M1', 'K5', or 'G5'; only used when `flux_model == 'Johnstone'`.
        Rp_override: Scalar planet radius [m] to manually fix a constant radius (radius will not
            evolve); False to disable.
        t_sat: XUV power-law saturation time [yr]; 5e8 matches semi-empirical MUSCLES data.
        step_fn: Toggles a step-function XUV flux evolution (drops to `F_final` at `t_pms`).
        F_final: Final relative XUV flux level (of F0) once `step_fn` engages.
        t_pms: Pre-main-sequence phase duration [yr].
        pms_factor: XUV enhancement factor applied during the pre-main-sequence phase.
        n_steps: Number of timesteps; convergence occurs at 1e6.
        t0: Simulation start time [yr].
        rho_rcb: Gas density at the RCB in the CPML phi equation [kg/m3].
        RR: Toggles the radiation-recombination effect (Ly-alpha cooling; Murray-Clay et al 2009).
        thermal: Toggles planet radius contraction in the Lopez/Fortney equations (False removes
            the age term).
        beta: Exponent in the Fxuv power-law function; determines the rate of XUV decrease.
            -1.23 is consistent with MUSCLES data.
        n_atmodeller: Interval of timesteps between each Atmodeller call.
        save_molecules: Save molecular abundances at every timestep (True) or only the final
            abundances (False).
        mantle_iron_dict: Allows Fe in the mantle to react with O2. `['type']="dynamic"` reacts
            only molten mantle Fe; `['type']="static"` reacts all mantle Fe; also specify
            `['Fe_mass_fraction']`. False disables this.
        dynamic_phi: Toggle dynamic phi calculation based on the most abundant species (True) or
            static phi calculation (False).
    """

    mechanism: str = "XUV"
    rad_evol: bool = True
    melt_fraction_override: ArrayLike | bool = False
    mu: ArrayLike = const.mu_solar
    eps: ArrayLike = 0.15
    activity: str = "medium"
    flux_model: str = "power law"
    stellar_type: str = "M1"
    Rp_override: ArrayLike | bool = False
    t_sat: ArrayLike = 5e8
    step_fn: bool = False
    F_final: ArrayLike = 0
    t_pms: ArrayLike = 0
    pms_factor: ArrayLike = 1e2
    n_steps: int = int(1e5)
    t0: ArrayLike = 1e6
    rho_rcb: ArrayLike = 1.0
    RR: bool = True
    thermal: bool = True
    beta: ArrayLike = -1.23
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    mantle_iron_dict: dict | bool = False
    dynamic_phi: bool = False
