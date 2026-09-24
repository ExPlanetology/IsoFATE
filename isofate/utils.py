# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small shared calculations used across isofate's plain and JAX implementations."""

import jax.numpy as jnp
from jax import Array
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


def safe_divide(numerator: ArrayLike, denominator: ArrayLike, fallback: ArrayLike = 0.0) -> Array:
    """Elementwise `numerator / denominator`, substituting `fallback` wherever `denominator` is
    exactly zero.

    The denominator is guarded (replaced with 1.0) before dividing, so the branch discarded by the
    final `jnp.where` never computes an actual 0/0 - which would otherwise corrupt `jax.grad`
    through the `jnp.where` even though that branch is never selected.

    Args:
        numerator: Dividend.
        denominator: Divisor; may be zero.
        fallback: Value to substitute wherever `denominator == 0` (default 0.0).

    Returns:
        `numerator / denominator` elementwise, or `fallback` wherever `denominator == 0`.
    """
    safe_denominator = jnp.where(denominator == 0, 1.0, denominator)
    return jnp.where(denominator == 0, fallback, numerator / safe_denominator)
