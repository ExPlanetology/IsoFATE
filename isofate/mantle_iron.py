# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mantle Fe-O2 reaction config/state, used by isocalc/AtmodellerCoupler to let mantle Fe react
with atmospheric O2. Mirrors isofate.escape's config/state split: `MantleIronConfig` is the
user-facing, fixed-for-the-run choice; `MantleIronState` is the evolving per-timestep state,
rebuilt via `dataclasses.replace` (never mutated in place).
"""

from typing import Literal

import equinox as eqx
from jaxtyping import ArrayLike


class MantleIronConfig(eqx.Module):
    """User-facing choice of which Fe reservoir reacts with atmospheric O2, and how much.

    Args:
        reaction_type: "dynamic" reacts only currently-molten mantle Fe (mass_Fe recomputed every
            AtmodellerCoupler call from the current melt fraction). "static" reacts the full
            initial mantle Fe budget once and tracks the remaining Fe2+/oxidized split thereafter.
        fe_mass_fraction: Mass fraction of Fe within the mantle.
    """

    reaction_type: Literal["dynamic", "static"]
    fe_mass_fraction: float


class MantleIronState(eqx.Module):
    """Evolving mantle-Fe-O2 reaction state.

    Args:
        config: The fixed-for-the-run reaction config.
        mass_Fe: Current reacting Fe mass [kg] - fixed for "static", recomputed every
            AtmodellerCoupler call for "dynamic".
        mass_Fe2: Remaining ferrous (Fe2+) mass [kg].
    """

    config: MantleIronConfig
    mass_Fe: ArrayLike
    mass_Fe2: ArrayLike

    @property
    def x_Fe2(self) -> ArrayLike:
        """Fraction of mass_Fe that remains Fe2+."""
        # FIXME: Switch logic is not JAX compliant
        return self.mass_Fe2 / self.mass_Fe if self.mass_Fe > 0 else 0.0

    @classmethod
    def initial(cls, config: MantleIronConfig, mantle_mass: ArrayLike) -> "MantleIronState":
        """Seeds the initial state: all reacting Fe starts as Fe2+."""
        mass_Fe: ArrayLike = mantle_mass * config.fe_mass_fraction

        return cls(config=config, mass_Fe=mass_Fe, mass_Fe2=mass_Fe)
