# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Star and Planet parameter objects for defining simulation scenarios."""

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from isofate.constants import const
from isofate.isofunks import R_Bondi, R_Hill
from isofate.orbit_params import EqTemp, Insolation, Luminosity, SemiMajor


class Star(eqx.Module):
    """A host star.

    Args:
        radius: Stellar radius [m]
        mass: Stellar mass [kg]
        temperature: Stellar effective temperature [K]
        luminosity: Stellar luminosity [W] (optional; if not provided, computed from radius and
            temperature)
    """

    radius: ArrayLike
    mass: ArrayLike
    temperature: ArrayLike
    _luminosity: ArrayLike | None = None

    def __init__(
        self,
        radius: ArrayLike,
        mass: ArrayLike,
        temperature: ArrayLike,
        luminosity: ArrayLike | None = None,
    ):
        self.radius = radius
        self.mass = mass
        self.temperature = temperature
        self._luminosity = luminosity

    @property
    def luminosity(self) -> ArrayLike:
        """Stellar luminosity [W], via the Stefan-Boltzmann law."""
        if self._luminosity is not None:
            return self._luminosity
        else:
            return Luminosity(self.radius, self.temperature)

    @property
    def t_jump(self) -> ArrayLike:
        """XUV saturation-time scaling factor [Gyr] for M dwarfs (t_sat = t_jump*1e9 [yr]).

        Only meaningful for M dwarfs; carried over as-is from the fit used in sim.py.
        """
        # TODO: Maybe remove if only relevant for LHS 1140?
        return 5.9 - 15.4 * (self.mass / const.Ms)

    def period_for_semi_major_axis(self, a: ArrayLike) -> ArrayLike:
        """Orbital period [s] a planet at semi-major axis `a` [m] would need, around this star.
        The inverse of `System.semi_major_axis` - a "what-if" helper for picking a
        `Planet.period` at construction time, before any Planet/System exists yet."""
        return (a**3 * 4 * np.pi**2 / const.G / self.mass) ** 0.5


class Planet(eqx.Module):
    """A planet.

    Args:
        mass: Planet mass [kg]
        period: Orbital period [s]
        f_atm: Atmospheric mass fraction [ndim]
    """

    mass: ArrayLike
    period: ArrayLike
    f_atm: ArrayLike

    @property
    def atmosphere_mass(self) -> ArrayLike:
        """Initial atmospheric mass [kg]."""
        return self.mass * self.f_atm

    @property
    def rocky_radius(self) -> ArrayLike:
        """Radius of the planet's rocky (condensed-matter) component [m], excluding any gaseous
        envelope (Lopez & Fortney 2014). Named to avoid confusion with a metallic core radius.

        NOTE: const.Re is missing from the paper (typo).
        """
        return const.Re * (self.mass / const.Me) ** 0.25


class System(eqx.Module):
    """A planet orbiting a star.

    Groups the quantities that genuinely depend on both bodies (or the orbit between them), as
    opposed to properties of the Star or Planet alone. `mu` (mean atmospheric particle mass) is
    deliberately never stored here: unlike everything else on this class, it evolves over the
    course of a simulation (see isocalc), so it must stay something the caller passes in each
    time rather than a fixed System property.

    Args:
        star: The host star
        planet: The orbiting planet
    """

    star: Star
    planet: Planet

    @property
    def semi_major_axis(self) -> ArrayLike:
        """Orbital semi-major axis [m]."""
        return SemiMajor(self.star.mass, self.planet.period)

    @property
    def insolation(self) -> ArrayLike:
        """Incident bolometric flux at the planet [W/m2]."""
        return Insolation(self.star.luminosity, self.semi_major_axis)

    @property
    def equilibrium_temperature(self) -> ArrayLike:
        """Planetary equilibrium temperature [K], assuming zero albedo."""
        return EqTemp(self.insolation)

    @property
    def hill_radius(self) -> ArrayLike:
        """Hill radius [m]."""
        return R_Hill(self.planet.mass, self.star.mass, self.semi_major_axis)

    def bondi_radius(self, mu: ArrayLike, T: ArrayLike | None = None) -> ArrayLike:
        """Bondi radius [m].

        Args:
            mu: Mean atmospheric particle mass [kg] - time-evolving, must be supplied by the
                caller (see the class docstring).
            T: Temperature [K]. Defaults to `equilibrium_temperature` if not given.
        """
        return R_Bondi(self.planet.mass, mu, T if T is not None else self.equilibrium_temperature)
