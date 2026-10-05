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


def scale_height(temperature: ArrayLike, mass: ArrayLike, gravity: ArrayLike) -> ArrayLike:
    """Isothermal ideal-gas scale height, H = k_B T / (m g).

    The per-particle form: `k_B` (Boltzmann's constant [J/K]) pairs with a particle mass [kg],
    rather than `R_gas` [J/mol/K], which would require a molar mass [kg/mol].

    Args:
        temperature: Temperature [K]
        mass: Particle mass [kg] - a mean molecular mass, or per-species masses (broadcasts)
        gravity: Gravitational acceleration [m/s2]

    Returns:
        Scale height [m]
    """
    return const.kb * temperature / (mass * gravity)
