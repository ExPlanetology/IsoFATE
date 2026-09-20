# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Canonical registry of the species tracked by IsoFATE's escape/interior model.

Single source of truth for the 7 tracked species (H, He, D, O, C, N, S), their atomic masses,
and their binary diffusion coefficients, used throughout isocalc()/isofunks.py instead of each
site hand-typing its own copy.
"""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from atmodeller.jax_utils import NpBool, NpFloat
from atmodeller.sci_utils import constants, unit_conversion
from jax import Array
from jax.typing import ArrayLike
from molmass import Formula

SYMBOLS: tuple[str, ...] = ("H", "He", "D", "O", "C", "N", "S")


class BinaryDiffusionCoefficients(eqx.Module):
    """Temperature-dependent binary diffusion coefficients, each of the form
    b = prefactor * T**exponent [molecules/m/s].

    Stored as symmetric matrices indexed by `symbols` order (b(i, j) == b(j, i), so no
    light/heavy sorting is needed to look one up). Only species pairs listed in `COEFFICIENTS`
    are filled in; undefined pairs fall back to `default_pair`.

    Subclass and override `COEFFICIENTS` (and `default_pair` if needed) to swap in a different
    literature source or species set without touching the lookup logic itself.

    Args:
        symbols: Element/isotope symbols, in the order the coefficient matrices are indexed by
            (default: `SYMBOLS`).
        default_pair: Species pair to fall back to for pairs not in `COEFFICIENTS`.
    """

    # (species1, species2, prefactor, exponent, source)
    # [molecules/m/s]
    COEFFICIENTS: ClassVar[tuple[tuple[str, str, float, float, str], ...]] = (
        ("H", "D", 7.183e19, 0.728, "Genda & Ikoma 2008, D in H (not measured directly)"),
        ("H", "He", 1.04e20, 0.732, "Mason & Marrero 1970 (and Hu, Seager, Yung 2015), H in He"),
        ("He", "D", 5.087e19, 0.728, "Approximated from H-D (Genda/Ikoma 2008 Appendix C)"),
        ("H", "O", 4.8e19, 0.75, "Wordsworth et al 2018"),
        ("He", "O", 2.61e19, 0.75, "Approximated from H-O (Genda/Ikoma 2008 Appendix C)"),
        ("H", "C", 4.85e19, 0.75, "Approximated from H-O (Genda/Ikoma 2008 Appendix C)"),
        ("He", "C", 2.64e19, 0.75, "Approximated from He-O (Genda/Ikoma 2008 Appendix C)"),
        ("H", "N", 4.85e19, 0.75, "Approximated from H-O (Genda/Ikoma 2008 Appendix C)"),
        ("He", "N", 2.65e19, 0.75, "Approximated from He-O (Genda/Ikoma 2008 Appendix C)"),
        ("H", "S", 4.73e19, 0.75, "Approximated from H-O (Genda/Ikoma 2008 Appendix C)"),
        ("He", "S", 2.48e19, 0.75, "Approximated from He-O (Genda/Ikoma 2008 Appendix C)"),
        # TODO: Currently breaks the code if species are not in SYMBOLS
        # ("H2", "HD", 4.48e19, 0.75, "Genda 2008, H2-HD"),
    )

    symbols: tuple[str, ...] = SYMBOLS
    default_pair: tuple[str, str] = ("H", "He")
    _symbol_index: dict[str, int] = eqx.field(init=False)
    _prefactor: NpFloat = eqx.field(init=False)
    _exponent: NpFloat = eqx.field(init=False)

    def __post_init__(self):
        self.symbols = tuple(self.symbols)
        self._symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        n: int = len(self.symbols)

        self._prefactor = np.full((n, n), np.nan)
        self._exponent = np.full((n, n), np.nan)
        for species1, species2, prefactor, exponent, _source in self.COEFFICIENTS:
            i, j = self._symbol_index[species1], self._symbol_index[species2]
            self._prefactor[i, j] = self._prefactor[j, i] = prefactor
            self._exponent[i, j] = self._exponent[j, i] = exponent

        # Backfill undefined pairs with the default pair's coefficient (already set above, since
        # default_pair must appear in COEFFICIENTS, and symmetric regardless of its order) so
        # get() needs no fallback branch.
        i, j = self._symbol_index[self.default_pair[0]], self._symbol_index[self.default_pair[1]]
        default_prefactor, default_exponent = self._prefactor[i, j], self._exponent[i, j]
        undefined: NpBool = np.isnan(self._prefactor)
        self._prefactor[undefined] = default_prefactor
        self._exponent[undefined] = default_exponent

    def get(self, species1: str, species2: str, temperature: ArrayLike) -> Array:
        """Binary diffusion coefficient between two species at temperature T [K].

        Symmetric in species order (b(species1, species2) == b(species2, species1)); pairs not
        in `COEFFICIENTS` default to `default_pair`.

        Args:
            species1: Symbol of the first species.
            species2: Symbol of the second species.
            temperature: Temperature [K] at which to evaluate the coefficient.

        Returns:
            Binary diffusion coefficient [molecules/m/s] between the two species at the given
            temperature.
        """
        i, j = self._symbol_index[species1], self._symbol_index[species2]

        return self._prefactor[i, j] * jnp.power(temperature, self._exponent[i, j])

    def get_row_by_mask(self, mask: ArrayLike, temperature: ArrayLike) -> Array:
        """Binary diffusion coefficients between the species selected by the one-hot `mask` (over
        `self.symbols`) and every other tracked species, at temperature T [K].

        The vectorized counterpart of `get()`: lets the selected species be data-dependent (a mask
        produced at runtime, e.g. from `jax.nn.one_hot` on a `jax.lax.top_k` result, or a static
        one-hot constant) instead of a symbol string known at trace time.

        Args:
            mask: One-hot vector selecting a species, ordered per `self.symbols`.
            temperature: Temperature [K] at which to evaluate the coefficients.

        Returns:
            Binary diffusion coefficients [molecules/m/s] between the selected species and every
            species in `self.symbols`, in that order.
        """
        prefactor_row = jnp.dot(jnp.asarray(self._prefactor).T, mask)
        exponent_row = jnp.dot(jnp.asarray(self._exponent).T, mask)
        return prefactor_row * jnp.power(temperature, exponent_row)

    def get_by_mask(self, mask1: ArrayLike, mask2: ArrayLike, temperature: ArrayLike) -> Array:
        """Binary diffusion coefficient between the two species selected by one-hot masks
        `mask1`/`mask2` (over `self.symbols`) at temperature T [K] - the mask-based counterpart of
        `get()`.
        """
        return jnp.dot(self.get_row_by_mask(mask1, temperature), jnp.asarray(mask2))


DEFAULT_BINARY_DIFFUSION: BinaryDiffusionCoefficients = BinaryDiffusionCoefficients()


class IsoFATESpecies(eqx.Module):
    """IsoFATE species container.

    Atomic masses [kg/atom] are computed once at construction from `molmass` (IUPAC standard
    atomic weights) rather than hand-typed constants, so they stay correct for any symbol without
    maintaining a lookup table by hand.

    A plain dataclass rather than an `eqx.Module`: nothing here is ever passed through
    `jax.jit`/`vmap`/`grad`, so there's no need to pay `eqx.Module`'s per-access method-wrapping
    cost (it wraps every bound method access in a fresh `BoundMethod` pytree so methods survive
    those transforms). `binary_diffusion.get` is called ~16 times per timestep via
    `Phi_minor_species` and doesn't need that.

    Args:
        species: Element/isotope symbols, in the canonical tracked order (default: the 7 species
            IsoFATE tracks - H, He, D, O, C, N, S; see `SYMBOLS` above).
        binary_diffusion: Binary diffusion coefficient lookup to use; swap in a differently
            configured or subclassed `BinaryDiffusionCoefficients` to use a different literature
            source.
    """

    species: tuple[str, ...] = SYMBOLS
    atomic_masses: NpFloat = eqx.field(init=False)
    mass_by_symbol: dict[str, float] = eqx.field(init=False)
    _symbol_index: dict[str, int] = eqx.field(init=False, repr=False)

    def __post_init__(self):
        self.species = tuple(self.species)
        self.atomic_masses = np.array(
            [
                Formula(symbol).mass * unit_conversion.g_to_kg / constants.Avogadro
                for symbol in self.species
            ]
        )
        self.mass_by_symbol = dict(zip(self.species, self.atomic_masses.tolist()))
        self._symbol_index = {symbol: i for i, symbol in enumerate(self.species)}

    def index(self, symbol: str) -> int:
        """Index of `symbol` within `self.species` (and thus `self.atomic_masses` and
        `scale_heights()`'s output, which share that ordering).
        """
        return self._symbol_index[symbol]

    def scale_heights(self, temperature: ArrayLike, gravity: ArrayLike) -> ArrayLike:
        """Atmospheric scale height for each species at a given temperature and gravity.

        H = k_B * T / (m * g), the per-particle form of the ideal-gas scale height - `k_B`
        (Boltzmann's constant [J/K]) pairs with `atomic_masses` [kg/atom], rather than `R_gas`
        [J/mol/K] which would require molar masses [kg/mol].

        Args:
            temperature: Temperature [K]
            gravity: Gravitational acceleration [m/s2]

        Returns:
            Scale height [m] for each species, ordered per `self.species`
        """
        return constants.Boltzmann * temperature / (self.atomic_masses * gravity)

    def atmosphere_mean_mu(self, y: Array) -> Array:
        """Mean atmospheric particle mass [kg], given per-species abundances `y` [atoms],
        ordered per `self.species`.

        Returns 0 when the total abundance is zero, rather than letting 0/0 propagate as NaN -
        this keeps the value finite even when a caller only conditionally uses it (e.g. near-total
        atmospheric exhaustion in isocalc_jax), so a discarded branch can't corrupt a gradient
        through the selecting `jnp.where`.

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`.

        Returns:
            Mean atmospheric particle mass [kg]
        """
        N_tot: Array = jnp.sum(y)
        safe_N_tot: Array = jnp.where(N_tot > 0, N_tot, 1.0)

        return jnp.where(N_tot > 0, self.atmosphere_mass(y) / safe_N_tot, 0.0)

    def atmosphere_mass(self, y: Array) -> Array:
        """Total atmospheric mass [kg].

        Args:
            y: Per-species abundances `y` [atoms], ordered per `self.species`

        Returns:
            Total atmospheric mass [kg]
        """
        return jnp.dot(y, self.atomic_masses)


DEFAULT_SPECIES: IsoFATESpecies = IsoFATESpecies()
