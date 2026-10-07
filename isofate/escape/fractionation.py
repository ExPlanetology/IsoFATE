# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-species number-flux physics for diffusion-limited atmospheric escape (Wordsworth et al.
2018 light/heavy binary-mixture fluxes; Zahnle et al. 1990 trace-species fluxes)."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from atmodeller.jax_utils import safe_divide
from jax import Array
from jaxtyping import ArrayLike

from isofate import override
from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    DEFAULT_SPECIES,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)


class EscapeNumberFluxBase(eqx.Module):
    """Shared fields and minor-species flux helper for `EscapeNumberFlux`/
    `EscapeNumberFluxDynamic` - the two differ only in how they pick the dominant light/heavy
    pair (always H/He vs. whichever two species are currently most abundant).

    Args:
        binary_diffusion: Binary diffusion coefficient table (defaults to `DEFAULT_BINARY_DIFFUSION`)
        species: Tracked-species registry (defaults to `DEFAULT_SPECIES`)
    """

    species: IsoFATESpecies = DEFAULT_SPECIES
    binary_diffusion: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION

    @abstractmethod
    def get_number_flux(
        self, y: ArrayLike, T: ArrayLike, g: ArrayLike, phi: ArrayLike
    ) -> tuple[Array, Array]:
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux."""

    def _phi_1_2(
        self,
        phi: ArrayLike,
        b: ArrayLike,
        H1: ArrayLike,
        H2: ArrayLike,
        m1: ArrayLike,
        m2: ArrayLike,
        x1: ArrayLike,
        x2: ArrayLike,
        mu: ArrayLike,
    ) -> tuple[Array, Array, Array]:
        """Calculates the number flux for both the light and heavy species in a binary gas mixture
        undergoing escape, together with the critical mass flux.

        Adapted from Wordsworth et al. 2018. Merges the former separate Phi_1/Phi_2 functions -
        every call site called them as a pair with identical arguments, so this computes their
        shared phi_c/mu==0/phi<phi_c logic once instead of twice.

        Args:
            phi: mass flux [kg/m2/s]
            b: binary diffusion coefficient [particles/m/s]
            H1/H2: scale heights of light/heavy species [m]
            m1/m2: molecular mass of light/heavy species [kg/particle]
            x1/x2: molar concentration of light/heavy species (x1=mol_1/mol_tot) [ndim]
            mu: average atmospheric atomic mass [kg/particle]

        Returns:
            number flux of light species [particles/m2/s], number flux of heavy species
            [particles/m2/s], critical mass flux [kg/s/m2]
        """
        # critical mass flux [kg/s/m2]
        phi_c: Array = b * x1 * (m2 - m1) / H1  # pyright: ignore[reportOperatorIssue, reportAssignmentType]

        phi1_below_critical: ArrayLike = phi / m1
        phi2_below_critical: ArrayLike = 0.0
        phi1_above_critical: ArrayLike = safe_divide(x1 * phi + x1 * x2 * (m2 - m1) * b / H2, mu)  # pyright: ignore[reportOperatorIssue]
        phi2_above_critical: ArrayLike = safe_divide(x2 * phi + x1 * x2 * (m1 - m2) * b / H1, mu)  # pyright: ignore[reportOperatorIssue]

        below_critical: Array = phi < phi_c  # pyright: ignore[reportOperatorIssue, reportAssignmentType]
        phi1: Array = jnp.where(below_critical, phi1_below_critical, phi1_above_critical)
        phi2: Array = jnp.where(below_critical, phi2_below_critical, phi2_above_critical)

        phi1 = jnp.where(mu == 0, 0.0, phi1)
        phi2 = jnp.where(mu == 0, 0.0, phi2)

        return phi1, phi2, phi_c

    def _phi_minor_species(
        self,
        phi_1: ArrayLike,
        phi_2: ArrayLike,
        h_1: ArrayLike,
        h_2: ArrayLike,
        scale_heights: ArrayLike,
        n_values: Array,
        t: ArrayLike,
        mask_light: ArrayLike,
        mask_heavy: ArrayLike,
    ) -> Array:
        """Vectorized counterpart of `Phi_minor_species`: computes the minor-species number flux
        for every tracked species at once (ordered per `self.binary_diffusion`'s `symbols`), given
        the dominant light/heavy pair as one-hot masks rather than static indices/symbols -
        trace-safe whether the masks are compile-time constants (`EscapeNumberFlux`'s always-H/He
        case) or data-dependent (`EscapeNumberFluxDynamic`'s runtime-selected pair).

        Entries at the two dominant positions are not meaningful (the caller overwrites them using
        the same masks) but are computed anyway, since avoiding a branch there is the point of
        vectorizing.

        Args:
            phi_1: number flux of lightest dominant species [atoms/s/m2]
            phi_2: number flux of heaviest dominant species [atoms/s/m2]
            h_1: scale height of lightest dominant species [m]
            h_2: scale height of heaviest dominant species [m]
            scale_heights: scale height of every tracked species [m], ordered per
                `self.binary_diffusion`'s `symbols`
            n_values: current abundances, ordered per `self.binary_diffusion`'s `symbols`
            t: temperature [K]
            mask_light: one-hot vector selecting the lightest dominant species (species 1)
            mask_heavy: one-hot vector selecting the heaviest dominant species (species 2)
        """
        b_1_row = self.binary_diffusion.get_row_by_mask(mask_light, t)  # b(light, k) for every k
        b_2_row = self.binary_diffusion.get_row_by_mask(mask_heavy, t)  # b(heavy, k) for every k
        b_1_2 = mask_heavy @ b_1_row  # b(light, heavy), consistent with b_1_row

        alpha_2_row = safe_divide(b_1_row, b_1_2, fallback=1.0)  # b_1_minor/b_1_2
        alpha_3_row = safe_divide(b_1_row, b_2_row, fallback=1.0)  # b_1_minor/b_2_minor

        phi_dl_row = b_1_row * (1 / scale_heights - 1 / h_1)
        phi_dl_2 = b_1_2 * (1 / h_2 - 1 / h_1)

        n_1 = mask_light @ n_values  # lightest dominant species
        n_2 = mask_heavy @ n_values  # heaviest dominant species
        n_total = jnp.sum(n_values)

        f_2 = safe_divide(n_2, n_1)  # n_2/n_1 (heavy/light dominant)
        f_minor_row = safe_divide(n_values, n_1)  # n_minor/n_1, for every species

        # Calculate molar fraction of species 2 in total atmosphere
        x_2 = safe_divide(n_2, n_total)

        # Zahnle et al. 1990 formulation
        num_row = phi_1 - phi_dl_row + alpha_2_row * phi_dl_2 * x_2 + alpha_3_row * phi_2
        denom_row = 1 + alpha_3_row * f_2

        result_row = jnp.maximum(0.0, f_minor_row * num_row / denom_row)
        result_row = jnp.where(n_1 == 0, 0.0, result_row)
        result_row = jnp.where(n_total == 0, 0.0, result_row)

        return result_row


class EscapeNumberFlux(EscapeNumberFluxBase):
    """Bundles a BinaryDiffusionCoefficients table and an IsoFATESpecies registry to compute the
    per-species escape number flux in one call, given the current abundances and physical state.

    H and He are always treated as the light/heavy diffusive pair (Phi_1_2); every other tracked
    species is treated as a minor/trace species relative to that pair
    (`_phi_minor_species`).

    In the original IsoFATE codebase, this corresponds to the ``dynamic_phi==False`` branch.
    """

    @override
    def get_number_flux(
        self, y: ArrayLike, T: ArrayLike, g: ArrayLike, phi: ArrayLike
    ) -> tuple[Array, Array]:
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux.

        Args:
            y: Current abundances [atoms], ordered per `self.species.species` - traced.
            T: Temperature [K] - traced.
            g: Gravitational acceleration [m/s2] - traced.
            phi: Total escape mass flux [kg/m2/s] (from the EscapeMechanism) - traced.

        Returns:
            Phi (number flux [atoms/s/m2] for each species, ordered per `self.species.species`),
            phi_c (critical mass flux [kg/s/m2], from `_phi_1_2`).
        """
        n_species = len(self.species.species)
        H_idx = self.species.index("H")
        He_idx = self.species.index("He")
        mask_H = jax.nn.one_hot(H_idx, n_species)
        mask_He = jax.nn.one_hot(He_idx, n_species)

        H = self.species.scale_heights(T, g)  # ordered per self.species.species
        mu_H = self.species.atomic_masses[H_idx]
        mu_He = self.species.atomic_masses[He_idx]

        y = jnp.asarray(y)

        y_H, y_He = y[H_idx], y[He_idx]
        y_HHe_total = y_H + y_He
        X1 = safe_divide(y_H, y_HHe_total)
        X2 = safe_divide(y_He, y_HHe_total)
        MU = X1 * mu_H + X2 * mu_He
        b_H_He = self.binary_diffusion.get_by_mask(mask_H, mask_He, T)

        Phi_H, Phi_He, phi_c = self._phi_1_2(
            phi, b_H_He, H[H_idx], H[He_idx], mu_H, mu_He, X1, X2, MU
        )

        Phi_minor_all = self._phi_minor_species(
            Phi_H,
            Phi_He,
            H[H_idx],
            H[He_idx],
            H,
            y,
            T,
            mask_H,
            mask_He,
        )
        not_dominant = 1.0 - mask_H - mask_He
        Phi = mask_H * Phi_H + mask_He * Phi_He + not_dominant * Phi_minor_all

        return Phi, phi_c


class EscapeNumberFluxDynamic(EscapeNumberFluxBase):
    """Like `EscapeNumberFlux`, but picks whichever two species are currently most abundant as
    the dominant light/heavy diffusive pair (Phi_1_2), instead of always using H/He - every other
    tracked species is still treated as a minor/trace species relative to that pair
    (`_phi_minor_species`).

    In the original IsoFATE codebase, this corresponds to the ``dynamic_phi==True`` branch.

    Selecting the dominant pair here is inherently data-dependent (it depends on the current
    abundances `y`), so - unlike `EscapeNumberFlux`'s always-static H/He indices - the pair is
    represented as one-hot masks derived from `jax.lax.top_k`/`jnp.where` rather than Python-level
    indices, making this trace-safe (`jax.jit`/`eqx.filter_jit`) despite the dominant pair being
    data-dependent.
    """

    @override
    def get_number_flux(
        self, y: ArrayLike, T: ArrayLike, g: ArrayLike, phi: ArrayLike
    ) -> tuple[Array, Array]:
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux.

        Args:
            y: Current abundances [atoms], ordered per `self.species.species` - traced.
            T: Temperature [K] - traced.
            g: Gravitational acceleration [m/s2] - traced.
            phi: Total escape mass flux [kg/m2/s] (from the EscapeMechanism) - traced.

        Returns:
            Phi (number flux [atoms/s/m2] for each species, ordered per `self.species.species`),
            phi_c (critical mass flux [kg/s/m2], from `_phi_1_2`).
        """
        n_species = len(self.species.species)
        atomic_masses = self.species.atomic_masses
        scale_heights = self.species.scale_heights(T, g)
        N_values = jnp.asarray(y)

        _, top2_idx = jax.lax.top_k(N_values, 2)
        mask_most = jax.nn.one_hot(top2_idx[0], n_species)
        mask_second = jax.nn.one_hot(top2_idx[1], n_species)

        mass_most = mask_most @ atomic_masses
        mass_second = mask_second @ atomic_masses

        # Determine which is lighter (species 1) and heavier (species 2) by MASS - resolved via
        # jnp.where (mass_most/mass_second are traced) rather than a plain Python if.
        most_is_lighter = mass_most <= mass_second
        mask_light = jnp.where(most_is_lighter, mask_most, mask_second)
        mask_heavy = jnp.where(most_is_lighter, mask_second, mask_most)

        mass_1 = mask_light @ atomic_masses
        mass_2 = mask_heavy @ atomic_masses
        H_1 = mask_light @ scale_heights
        H_2 = mask_heavy @ scale_heights
        N1 = mask_light @ N_values  # lightest dominant (species 1)
        N2 = mask_heavy @ N_values  # heaviest dominant (species 2)

        N_tot_binary = N1 + N2
        X1 = safe_divide(N1, N_tot_binary)
        X2 = safe_divide(N2, N_tot_binary)
        MU = X1 * mass_1 + X2 * mass_2

        # Calculate binary diffusion coefficient between the two dominant species
        b = self.binary_diffusion.get_by_mask(mask_light, mask_heavy, T)

        # Calculate escape fluxes for the two dominant species
        Phi_1_calc, Phi_2_calc, phi_c = self._phi_1_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)

        Phi_minor_all = self._phi_minor_species(
            Phi_1_calc,
            Phi_2_calc,
            H_1,
            H_2,
            scale_heights,
            N_values,
            T,
            mask_light,
            mask_heavy,
        )
        not_dominant = 1.0 - mask_light - mask_heavy
        Phi = mask_light * Phi_1_calc + mask_heavy * Phi_2_calc + not_dominant * Phi_minor_all

        return Phi, phi_c


def make_escape_number_flux(
    dynamic_phi: bool,
    species: IsoFATESpecies = DEFAULT_SPECIES,
    binary_diffusion: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION,
) -> EscapeNumberFluxBase:
    """Escape number flux model selected by `dynamic_phi`.

    Args:
        dynamic_phi: Use the two currently most abundant species as the dominant light/heavy pair
            (`EscapeNumberFluxDynamic`, ``True``) or always H/He (`EscapeNumberFlux`, ``False``).
        species: Tracked-species registry. Defaults to `DEFAULT_SPECIES`.
        binary_diffusion: Binary diffusion coefficient table. Defaults to
            `DEFAULT_BINARY_DIFFUSION`.

    Returns:
        Escape number flux model
    """
    flux_class = EscapeNumberFluxDynamic if dynamic_phi else EscapeNumberFlux

    return flux_class(species=species, binary_diffusion=binary_diffusion)
