# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small shared calculations used across isofate's plain and JAX implementations."""

import jax.numpy as jnp
from jax.typing import ArrayLike

from isofate.constants import const


def sphere_area(radius: ArrayLike) -> ArrayLike:
    """Surface area of a sphere.

    Args:
        radius: Sphere radius [m]

    Returns:
        Surface area [m2]
    """
    return 4 * jnp.pi * radius**2


def gravitational_acceleration(mass: ArrayLike, radius: ArrayLike) -> ArrayLike:
    """Gravitational acceleration at `radius` from a point/enclosed mass `mass`.

    Args:
        mass: Enclosed mass [kg]
        radius: Radial distance from the center [m]

    Returns:
        Gravitational field strength [m/s2]
    """
    return const.G * mass / radius**2
