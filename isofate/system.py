# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Star and Planet parameter objects for defining simulation scenarios."""

import importlib
import importlib.resources
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib.resources.abc import Traversable
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from atmodeller.jax_utils import FloatArray, NpFloat, as_j64
from atmodeller.sci_utils import earth
from jax import Array
from jax.scipy.interpolate import RegularGridInterpolator
from jax.typing import ArrayLike

from isofate.constants import const
from isofate.isofunks import R_Bondi

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
MELT_FRACTION_DATA_SOURCE: Path = Path("melt_fraction_grid.npz")
"""Source of the melt fraction grid data, relative to `DATA_DIRECTORY`"""


class Star(eqx.Module):
    """A host star.

    Args:
        radius: Stellar radius [m]
        mass: Stellar mass [kg]
        temperature: Stellar effective temperature [K]
        luminosity: Stellar luminosity [W]. Defaults to ``None``, meaning it will be computed from
            radius and temperature.
    """

    radius: Array
    mass: Array
    temperature: Array
    _luminosity: Array

    def __init__(
        self,
        radius: ArrayLike,
        mass: ArrayLike,
        temperature: ArrayLike,
        luminosity: ArrayLike | None = None,
    ):
        self.radius = as_j64(radius)
        self.mass = as_j64(mass)
        self.temperature = as_j64(temperature)
        self._luminosity = as_j64(luminosity if luminosity is not None else jnp.nan)

    @property
    def luminosity(self) -> Array:
        """Stellar luminosity [W], via the Stefan-Boltzmann law."""
        computed: Array = (
            4 * jnp.pi * jnp.square(self.radius) * const.sbc * jnp.power(self.temperature, 4)
        )
        return jnp.where(jnp.isnan(self._luminosity), computed, self._luminosity)

    @property
    def t_jump(self) -> Array:
        """XUV saturation-time scaling factor [Gyr] for M dwarfs (t_sat = t_jump*1e9 [yr]).

        Only meaningful for M dwarfs; carried over as-is from the fit used in sim.py.
        """
        # TODO: Maybe remove if only relevant for LHS 1140?
        return 5.9 - 15.4 * (self.mass / const.Ms)

    def period_for_semi_major_axis(self, a: ArrayLike) -> Array:
        """Orbital period [s] a planet at semi-major axis `a` [m] would need, around this star.
        The inverse of `System.semi_major_axis` - a "what-if" helper for picking a
        `Planet.period` at construction time, before any Planet/System exists yet."""
        return jnp.sqrt((jnp.power(a, 3) * 4 * jnp.square(jnp.pi) / const.G / self.mass))


class Planet(eqx.Module):
    """A planet.

    NOTE: A potential refactor is to use the atmodeller Planet class more directly.

    Args:
        mass: Planet mass [kg]
        period: Orbital period [s]
        albedo: Planetary albedo [ndim]. Defaults to ``0``.
        _rocky_radius: Radius of the planet's rocky (condensed-matter) component [m] (optional; if
            not provided, computed from mass via Lopez & Fortney 2014). Supply this directly when
            a real, observationally-measured radius is known and should be used instead of the
            generic mass-scaling estimate - independent of whether a gaseous envelope is also
            allowed to evolve on top of it (see `isocalc`'s `rad_evol` option).
        core_mass_fraction: Mass fraction of the planet in its metallic core. Defaults to Earth.
        temperature: Planet surface temperature [K]. Defaults to 2000 K.
        mantle_melt_fraction: Mantle melt fraction. Defaults to ``None`` to compute from mass
            and temperature using a pre-computed grid.
    """

    mass: Array
    period: Array
    albedo: Array
    core_mass_fraction: Array
    temperature: Array
    _mantle_melt_fraction: Array
    _mantle_melt_fraction_interpolator: Callable[[tuple[ArrayLike, ArrayLike]], Array]
    _rocky_radius: Array

    def __init__(
        self,
        mass: ArrayLike,
        period: ArrayLike,
        albedo: ArrayLike = 0.0,
        core_mass_fraction: ArrayLike = earth.core_mass_fraction,
        temperature: ArrayLike = 2000,
        mantle_melt_fraction: ArrayLike | None = None,
        rocky_radius: ArrayLike | None = None,
    ):
        self.mass = as_j64(mass)
        self.period = as_j64(period)
        self.albedo = as_j64(albedo)
        self.core_mass_fraction = as_j64(core_mass_fraction)
        self.temperature = as_j64(temperature)
        self._mantle_melt_fraction = as_j64(
            mantle_melt_fraction if mantle_melt_fraction is not None else jnp.nan
        )
        self._mantle_melt_fraction_interpolator = self._get_melt_fraction_interpolator()
        self._rocky_radius = as_j64(rocky_radius if rocky_radius is not None else jnp.nan)

    @classmethod
    def _get_melt_fraction_interpolator(cls) -> Callable[[tuple[ArrayLike, ArrayLike]], Array]:
        """Constructs a RegularGridInterpolator for mantle melt fraction."""
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(MELT_FRACTION_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            data_np: NpFloat = np.load(datapath)
            temp_grid: NpFloat = data_np["temp_grid"]  # pyright: ignore
            mass_grid: NpFloat = data_np["mass_grid"]  # pyright: ignore
            psi_grid: NpFloat = data_np["psi_grid"]  # pyright: ignore

        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (mass_grid, temp_grid), psi_grid.transpose(), method="linear"
        )

        def interpolator_hashable_function_wrapper(x: tuple[ArrayLike, ArrayLike]) -> FloatArray:
            """Converts interpolator to a hashable function"""
            return interpolator(x)

        return interpolator_hashable_function_wrapper

    @property
    def has_fixed_radius(self) -> Array:
        """True if `rocky_radius` was supplied explicitly rather than computed from mass.

        A traced value-level `Array`, not a Python `bool` - see `_rocky_radius`'s comment.
        """
        return jnp.logical_not(jnp.isnan(self._rocky_radius))

    def mantle_melt_fraction(self, temperature: ArrayLike) -> Array:
        """Mantle melt fraction, either supplied at construction or computed from mass and
        temperature using a pre-computed grid.

        Branches on `jnp.isnan` (a traced value-level condition), not a Python `is not None`
        check, since `_mantle_melt_fraction` is always a concrete Array (a `jnp.nan` sentinel
        when not supplied, never Python `None`) - see the field's docstring/comment.
        """
        interpolated = self._mantle_melt_fraction_interpolator((self.mass / const.Me, temperature))
        return jnp.where(
            jnp.isnan(self._mantle_melt_fraction), interpolated, self._mantle_melt_fraction
        )

    @property
    def rocky_radius(self) -> Array:
        """Radius of the planet's rocky (condensed-matter) component [m], excluding any gaseous
        envelope. Named to avoid confusion with a metallic core radius.

        Returns the value supplied at construction if given; otherwise computed from mass
        (Lopez & Fortney 2014).

        NOTE: const.Re is missing from the paper (typo).
        """
        computed = const.Re * jnp.power(self.mass / const.Me, 1 / 4)
        return jnp.where(jnp.isnan(self._rocky_radius), computed, self._rocky_radius)


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
    def semi_major_axis(self) -> Array:
        """Orbital semi-major axis [m]."""
        a: Array = jnp.power(
            ((const.G * self.star.mass / 4 / jnp.square(jnp.pi)) * jnp.square(self.planet.period)),
            1 / 3,
        )
        return a

    @property
    def insolation(self) -> Array:
        """Incident bolometric flux at the planet [W/m2]."""
        insolation: Array = self.star.luminosity / (4 * jnp.pi * jnp.square(self.semi_major_axis))

        return insolation

    @property
    def equilibrium_temperature(self) -> Array:
        """Planetary equilibrium temperature [K], assuming zero albedo."""
        Teq: Array = jnp.power((self.insolation * (1 - self.planet.albedo) / 4 / const.sbc), 1 / 4)

        return Teq

    @property
    def hill_radius(self) -> Array:
        """Hill radius [m]."""
        hill_radius: Array = self.semi_major_axis * jnp.power(
            self.planet.mass / 3 / self.star.mass, 1 / 3
        )

        return hill_radius

    def bondi_radius(self, mu: ArrayLike, temperature: ArrayLike | None = None) -> ArrayLike:
        """Bondi radius [m].

        Args:
            mu: Mean atmospheric particle mass [kg] - time-evolving, must be supplied by the
                caller (see the class docstring).
            T: Temperature [K]. Defaults to `equilibrium_temperature` if not given.

        Returns:
            Bondi radius [m]
        """
        _temperature: ArrayLike = (
            temperature if temperature is not None else self.equilibrium_temperature
        )

        return R_Bondi(self.planet.mass, mu, _temperature)

    def tidal_reduction_factor(
        self, radius: ArrayLike | None = None, floor: float = 0.01
    ) -> Array:
        """Gravitational potential reduction factor due to stellar tidal forces (Erkaev et al.
        2007).

        Args:
            radius: Planet radius [m]. Defaults to `planet.rocky_radius` if not given; pass the
                current total (rocky + envelope) radius explicitly when it's time-evolving (see
                the class docstring).
            floor: Minimum value returned. Defaults to `0.01`.

        Returns:
            Gravitational potential reduction factor [ndim]
        """
        _radius: ArrayLike = radius if radius is not None else self.planet.rocky_radius

        delta: Array = self.planet.mass / self.star.mass
        lam: Array = self.semi_major_axis / _radius
        zeta: Array = lam * (delta / 3) ** (1 / 3)
        V_reduction: Array = 1 - 3 / 2 / zeta + 1 / 2 / jnp.power(zeta, 3)

        return jnp.maximum(V_reduction, floor)

    def gravitational_potential(
        self, radius: ArrayLike | None = None, floor: float = 0.01
    ) -> Array:
        """Gravitational potential at the outer layer [J/kg], reduced by `tidal_reduction_factor`.

        Args:
            radius: Planet radius [m]. Defaults to `planet.rocky_radius` if not given; pass the
                current total (rocky + envelope) radius explicitly when it's time-evolving (see
                the class docstring).
            floor: Minimum `tidal_reduction_factor` value used. Defaults to `0.01`.

        Returns:
            Gravitational potential [J/kg]
        """
        _radius: ArrayLike = radius if radius is not None else self.planet.rocky_radius
        K: Array = self.tidal_reduction_factor(_radius, floor)

        return K * const.G * self.planet.mass / _radius
