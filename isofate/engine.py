# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Planet-radius and escape physics shared by the isocalc drivers: the envelope/atmosphere
thickness and total radius (Lopez & Fortney 2014), the Bondi and escape radii, and the tidally
reduced gravitational potential. The time integration lives in `isofate.integrators`.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from isofate.constants import const
from isofate.parameters import Parameters
from isofate.system import Planet, System
from isofate.utils import (
    gravitational_acceleration,
    gravitational_potential,
    scale_height,
)


def convective_envelope_thickness(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Radial thickness contributed by the convective portion of the H/He envelope (down to the
    radiative-convective boundary), one of the three additive terms making up the total planet
    radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez and Fortney, 2014. Zero if `IsocalcOptions.rad_evol` is False (the radius
    is fixed at the rocky radius).

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Thickness contribution of the convective envelope [m]
    """
    planet: Planet = parameters.system.planet
    envelope_mass_fraction: Array = parameters.envelope_mass_fraction(y)
    if not parameters.isocalc_options.rad_evol:
        return jnp.zeros_like(envelope_mass_fraction)
    thermal: bool = parameters.isocalc_options.thermal
    Fp: Array = parameters.system.insolation

    c1: Array = planet.mass / const.Me  # Me = Earth mass [kg]
    # FIXME: Collin to clarify the magic number below
    c2: Array = envelope_mass_fraction / 0.05
    c3: Array = Fp / const.Fe  # Fe = Earth incident bolometric flux [W/m2]
    if thermal:
        # FIXME: Collin to clarify the magic number below
        c4: ArrayLike = age * const.s2yr / 5e9
    else:
        c4 = 1

    return 2.06 * const.Re * c1 ** (-0.21) * c2 ** (0.59) * c3 ** (0.044) * c4 ** (-0.18)


def radiative_atmosphere_thickness(parameters: Parameters, y: Array, age: ArrayLike) -> ArrayLike:
    """Radial thickness contributed by the radiative outer atmosphere above the
    radiative-convective boundary - the third of the three additive terms making up the total
    planet radius: R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Adapted from Lopez and Fortney, 2014.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Thickness contribution of the radiative atmosphere [m]
    """
    planet: Planet = parameters.system.planet
    envelope_thickness: ArrayLike = convective_envelope_thickness(parameters, y, age)
    mu: Array = parameters.atmosphere_mean_mu(y)
    equilibrium_temperature: Array = parameters.system.equilibrium_temperature

    g: ArrayLike = gravitational_acceleration(
        planet.mass, planet.rocky_radius + envelope_thickness
    )

    return 9 * scale_height(equilibrium_temperature, mu, g)


def total_radius(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Total planet radius [m]: the rocky-core radius plus the two additive envelope/atmosphere
    thickness terms, computed internally.

    R_p = R_rocky + convective_envelope_thickness + radiative_atmosphere_thickness.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Total planet radius [m]
    """
    planet: Planet = parameters.system.planet
    envelope_thickness: Array = convective_envelope_thickness(parameters, y, age)
    atmosphere_thickness: ArrayLike = radiative_atmosphere_thickness(parameters, y, age)

    return planet.rocky_radius + envelope_thickness + atmosphere_thickness


def bondi_radius(parameters: Parameters, y: Array):
    """Bondi radius calculation.

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`

    Returns:
        Bondi radius [m]
    """
    planet_mass: Array = parameters.system.planet.mass
    mu: Array = parameters.atmosphere_mean_mu(y)
    equilibrium_temperature: Array = parameters.system.equilibrium_temperature
    gamma: ArrayLike = parameters.atmosphere.adiabatic_index

    return (gamma - 1) * const.G * planet_mass * mu / (gamma * const.kb * equilibrium_temperature)


def escape_radius(parameters: Parameters, y: Array, age: ArrayLike) -> Array:
    """Effective outer radius of the planet, from which the atmosphere escapes.

    The structural radius (`total_radius`) capped by the Bondi radius, beyond which gas is not
    thermally bound to the planet, and by the Hill radius, beyond which the star's gravity
    dominates. If `IsocalcOptions.rad_evol` is False, the radius is instead fixed at the rocky
    radius (no envelope, and no Bondi or Hill cap).

    Args:
        parameters: Parameters
        y: Per-species abundances [atoms], ordered per `parameters.isofate_species.species`
        age: age [s]

    Returns:
        Escape radius [m]
    """
    if not parameters.isocalc_options.rad_evol:
        # FIXME: The Atmodeller coupling does not converge with this zero-thickness atmosphere
        # (see AtmodellerCoupler.run), so a fixed radius only works with n_atmodeller = 0
        return jnp.asarray(parameters.system.planet.rocky_radius, dtype=float)

    return jnp.minimum(
        bondi_radius(parameters, y),
        jnp.minimum(parameters.system.hill_radius, total_radius(parameters, y, age)),
    )


def tidal_reduction_factor(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
    """Gravitational potential reduction factor due to stellar tidal forces (Erkaev et al. 2007).

    Args:
        system: System parameters
        radius: Planet radius [m] - pass the current total (rocky + envelope) radius when it's
            time-evolving.
        floor: Minimum value returned. Defaults to `0.01`.

    Returns:
        Gravitational potential reduction factor [ndim]
    """
    delta: Array = system.planet.mass / system.star.mass
    lam: Array = system.semi_major_axis / radius
    zeta: Array = lam * (delta / 3) ** (1 / 3)
    V_reduction: Array = 1 - 3 / 2 / zeta + 1 / 2 / jnp.power(zeta, 3)

    return jnp.maximum(V_reduction, floor)


def tidal_gravitational_potential(system: System, radius: ArrayLike, floor: float = 0.01) -> Array:
    """Gravitational potential at the outer layer [J/kg], reduced by `tidal_reduction_factor`.

    Args:
        system: System parameters - `system.planet.mass` is read from it (and passed through to
            `tidal_reduction_factor`).
        radius: Planet radius [m] - pass the current total (rocky + envelope) radius when it's
            time-evolving.
        floor: Minimum `tidal_reduction_factor` value used. Defaults to `0.01`.

    Returns:
        Gravitational potential [J/kg]
    """
    tidal_factor: Array = tidal_reduction_factor(system, radius, floor)

    return tidal_factor * gravitational_potential(system.planet.mass, radius)
