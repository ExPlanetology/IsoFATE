# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration for `isocalc` - mode switches and tuning constants unrelated to escape
mechanism choice, fixed for the whole run. Escape-mechanism-specific config (mechanism choice,
RR toggle, XUV flux model, CPML tuning, ...) lives on the `isofate.escape.EscapeMechanism`
subclass instances instead (see `isofate/escape.py`), passed to `isocalc` as a separate
`escape` argument.
"""

from dataclasses import dataclass

from jaxtyping import ArrayLike

from isofate.constants import const
from isofate.mantle_iron import MantleIronConfig


@dataclass
class IsocalcOptions:
    """Mode switches and tuning constants for `isocalc`, fixed for the whole run.

    These are numerical/modeling choices, as opposed to `isocalc`'s other arguments (`system`,
    `F0`, `time`, and the initial `N_x` abundances), which describe the actual physical
    initial-value problem being solved and so stay direct `isocalc` arguments.

    Args:
        rad_evol: Set to False to fix the planet radius at the rocky radius.
        melt_fraction_override: Fixed mantle melt fraction; if False, it is instead calculated
            from Mp and T_surface.
        mu: Average atmospheric particle mass [kg]; default is H/He solar composition. Only the
            initial value - `isocalc` recomputes it every step from the evolving abundances.
        n_steps: Number of timesteps; convergence occurs at 1e6.
        t0: Simulation start time [yr].
        thermal: Toggles planet radius contraction in the Lopez/Fortney equations (False removes
            the age term).
        n_atmodeller: Interval of timesteps between each Atmodeller call.
        save_molecules: Save molecular abundances at every timestep (True) or only the final
            abundances (False).
        mantle_iron: Allows Fe in the mantle to react with O2. `reaction_type="dynamic"` reacts
            only molten mantle Fe; `reaction_type="static"` reacts all mantle Fe; also specify
            `fe_mass_fraction`. None disables this.
        dynamic_phi: Toggle dynamic phi calculation based on the most abundant species (True) or
            static phi calculation (False).
    """

    rad_evol: bool = True
    melt_fraction_override: ArrayLike | bool = False
    mu: ArrayLike = const.mu_solar
    n_steps: int = int(1e5)
    t0: ArrayLike = 1e6
    thermal: bool = True
    n_atmodeller: int = int(1e2)
    save_molecules: bool = False
    mantle_iron: MantleIronConfig | None = None
    dynamic_phi: bool = False
