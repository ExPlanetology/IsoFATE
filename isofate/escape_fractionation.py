# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-species number-flux physics for diffusion-limited atmospheric escape (Wordsworth et al.
2018 light/heavy binary-mixture fluxes; Zahnle et al. 1990 trace-species fluxes). Used by
`isocalc`'s `dynamic_phi` branches (isofate_coupler.py).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    DEFAULT_SPECIES,
    SYMBOLS,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)

# Number flux of light and heavy species


def Phi_1_2(
    phi: ArrayLike,
    b: ArrayLike,
    H1: ArrayLike,
    H2: ArrayLike,
    m1: ArrayLike,
    m2: ArrayLike,
    x1: ArrayLike,
    x2: ArrayLike,
    mu: ArrayLike,
):
    """Calculates the number flux for both the light and heavy species in a binary gas mixture
    undergoing escape, together with the critical mass flux.

    Adapted from Wordsworth et al. 2018. Merges the former separate Phi_1/Phi_2 functions - every
    call site in the codebase called them as a pair with identical arguments, so this computes
    their shared phi_c/mu==0/phi<phi_c logic once instead of twice.

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
    phi_c = b * x1 * (m2 - m1) / H1  # pyright: ignore[reportOperatorIssue]

    # mu is guarded before dividing so the discarded mu==0 branch never computes a 0/0 division,
    # which would otherwise corrupt jax.grad through the jnp.where even though it's never selected.
    safe_mu: Array = jnp.where(mu == 0, 1.0, mu)

    phi1_below_critical: ArrayLike = phi / m1
    phi2_below_critical: ArrayLike = 0.0
    phi1_above_critical: ArrayLike = (x1 * phi + x1 * x2 * (m2 - m1) * b / H2) / safe_mu  # pyright: ignore[reportOperatorIssue]
    phi2_above_critical: ArrayLike = (x2 * phi + x1 * x2 * (m1 - m2) * b / H1) / safe_mu  # pyright: ignore[reportOperatorIssue]

    below_critical: Array = phi < phi_c
    phi1: Array = jnp.where(below_critical, phi1_below_critical, phi1_above_critical)
    phi2: Array = jnp.where(below_critical, phi2_below_critical, phi2_above_critical)

    phi1 = jnp.where(mu == 0, 0.0, phi1)
    phi2 = jnp.where(mu == 0, 0.0, phi2)

    return phi1, phi2, phi_c


# number flux deuterium Gu & Chen 2023


def Phi_D_GC23(Phi_H, Phi_He, H_H, H_D, H_He, N_H, N_He, N_D, T):
    """
    Calculates number flux of deuterium for simultaneous calculation of H/He/D escape
    From Gu & Chen 2023

    Not currently plugged into the Phi calc - an alternate/experimental formulation kept for
    reference.

    Inputs:
        - Phi_i: number flux [particles/m2/s]
        - H_i: scale height [m]
        - N_i: particles of species i
        - T: eq temp [K]
    """
    b_H_D = (
        7.183e19 * T**0.728
    )  # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_He = 1.04e20 * T**0.732  # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_He_D = (
        5.087e19 * T**0.728
    )  # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription
    alpha_2 = b_H_D / b_H_He
    alpha_3 = b_H_D / b_He_D
    Phi_DL_D = b_H_D * (1 / H_D - 1 / H_H)
    Phi_DL_He = b_H_He * (1 / H_He - 1 / H_H)
    X_He = N_He / (N_H + N_He + N_D)
    X_H = N_H / (N_H + N_He + N_D)
    X_D = N_D / (N_H + N_He + N_D)
    num = Phi_H - Phi_DL_D + alpha_2 * Phi_DL_He * X_He + alpha_3 * Phi_He
    denom = X_H + alpha_3 * X_He
    return max(0, X_D * num / denom)


# number flux deuterium derived from Zahnle et al 1990


def Phi_D_Z90_mod(Phi_H, H_D, N_D, N_H, T):
    """
    Phi_D_Z90 solution with He set to zero

    Not currently plugged into the Phi calc - an alternate/experimental formulation kept for
    reference.
    """
    b_H_D = (
        7.183e19 * T**0.728
    )  # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    Phi_DL_D = b_H_D / H_D
    f_D = N_D / N_H
    return f_D * Phi_H - (N_D / (N_D + N_H)) * Phi_DL_D


def Phi_D_Z90_mod2(Phi_H, Phi_He, H_H, H_D, H_He, N_H, N_He, N_D, T):
    """
    Phi_D solution from referee report Cherubim et al 2024

    Not currently plugged into the Phi calc - an alternate/experimental formulation kept for
    reference.
    """
    b_H_D = (
        7.183e19 * T**0.728
    )  # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_He = 1.04e20 * T**0.732  # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_He_D = (
        5.087e19 * T**0.728
    )  # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription (Appendix C)
    alpha_3 = b_H_D / b_He_D
    Phi_DL_D = b_H_D / H_D
    Phi_DL_He = b_H_He / H_He
    x_He = N_He / (N_H + N_He + N_D)
    f_He = N_He / N_H
    f_D = N_D / N_H
    num = Phi_DL_He * x_He - Phi_DL_D + Phi_H + alpha_3 * Phi_He
    denom = 1 + alpha_3 * f_He
    return max(0, f_D * num / denom)


def Phi_minor_species(
    Phi_1,
    Phi_2,
    H_1,
    H_2,
    H_minor,
    N_values,
    T,
    minor_species_idx,
    light_dominant_idx,
    heavy_dominant_idx,
    binary_diffusion: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION,
    species_symbols: tuple[str, ...] = SYMBOLS,
):
    """Calculates number flux for minor species using Zahnle et al. 1990

    Args:
        Phi_1: number flux of lightest dominant species [atoms/s/m2]
        Phi_2: number flux of heaviest dominant species [atoms/s/m2]
        H_1: scale height of lightest dominant species [m]
        H_2: scale height of heaviest dominant species [m]
        H_minor: scale height of minor species [m]
        N_values: list [N_H, N_He, N_D, N_O, N_C, N_N, N_S] - current abundances
        T: temperature [K]
        minor_species_idx: index (0-4) of the minor species being calculated
        light_dominant_idx: index of lightest dominant species (species 1)
        heavy_dominant_idx: index of heaviest dominant species (species 2)
        binary_diffusion: Binary diffusion coefficient lookup to use
        species_symbols: Symbols that `minor_species_idx`/`light_dominant_idx`/
            `heavy_dominant_idx` index into
    """
    minor_name: str = species_symbols[minor_species_idx]
    light_name: str = species_symbols[light_dominant_idx]  # species 1
    heavy_name: str = species_symbols[heavy_dominant_idx]  # species 2

    # Get binary diffusion coefficients
    b_1_minor = binary_diffusion.get(light_name, minor_name, T)  # b between species 1 and minor
    b_2_minor = binary_diffusion.get(heavy_name, minor_name, T)  # b between species 2 and minor
    b_1_2 = binary_diffusion.get(light_name, heavy_name, T)  # b between species 1 and 2

    # b_1_2/b_2_minor are essentially never exactly zero in practice (BinaryDiffusionCoefficients
    # always returns prefactor*T**exponent with a positive prefactor), but both are traced (depend
    # on T), so the zero-guards are resolved via jnp.where rather than a plain Python ternary.
    safe_b_1_2 = jnp.where(b_1_2 == 0, 1.0, b_1_2)
    safe_b_2_minor = jnp.where(b_2_minor == 0, 1.0, b_2_minor)
    alpha_2 = jnp.where(b_1_2 == 0, 1.0, b_1_minor / safe_b_1_2)  # b_1_minor/b_1_2
    alpha_3 = jnp.where(b_2_minor == 0, 1.0, b_1_minor / safe_b_2_minor)  # b_1_minor/b_2_minor

    Phi_DL_minor = b_1_minor * (1 / H_minor - 1 / H_1)
    Phi_DL_2 = b_1_2 * (1 / H_2 - 1 / H_1)

    N_1 = N_values[light_dominant_idx]  # lightest dominant species
    N_2 = N_values[heavy_dominant_idx]  # heaviest dominant species
    N_minor = N_values[minor_species_idx]
    N_total = sum(N_values)

    # Guard N_1/N_total before dividing so the discarded branches never compute 0/0.
    safe_N_1 = jnp.where(N_1 == 0, 1.0, N_1)
    f_2 = N_2 / safe_N_1  # N_2/N_1 (heavy/light dominant)
    f_minor = N_minor / safe_N_1  # N_minor/N_1

    # Calculate molar fraction of species 2 in total atmosphere
    safe_N_total = jnp.where(N_total > 0, N_total, 1.0)
    x_2 = jnp.where(N_total > 0, N_2 / safe_N_total, 0.0)

    # Zahnle et al. 1990 formulation
    num = Phi_1 - Phi_DL_minor + alpha_2 * Phi_DL_2 * x_2 + alpha_3 * Phi_2
    denom = 1 + alpha_3 * f_2

    result = jnp.maximum(0.0, f_minor * num / denom)
    result = jnp.where(N_1 == 0, 0.0, result)
    # NOTE: sum(N_values) == 0 implies N_1 == 0 too, since N_1 is one of N_total's non-negative
    # summands - this guard may be redundant with the N_1==0 guard above. Kept for now, matching
    # the original two-guard structure exactly; a future refactor could potentially drop it if
    # the function is confirmed to handle N_total==0 self-consistently via N_1==0 alone.
    result = jnp.where(N_total == 0, 0.0, result)

    return result


class EscapeNumberFlux(eqx.Module):
    """Bundles a BinaryDiffusionCoefficients table and an IsoFATESpecies registry to compute the
    per-species escape number flux in one call, given the current abundances and physical state.

    H and He are always treated as the light/heavy diffusive pair (Phi_1_2); every other tracked
    species is treated as a minor/trace species relative to that pair (Phi_minor_species).

    In the original IsoFATE codebase, this corresponds to the ``dynamic_phi==False`` branch.

    Args:
        binary_diffusion: Binary diffusion coefficient table (defaults to `DEFAULT_BINARY_DIFFUSION`)
        species: Tracked-species registry (defaults to `DEFAULT_SPECIES`)
    """

    species: IsoFATESpecies = DEFAULT_SPECIES
    binary_diffusion: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION

    def get_number_flux(self, y: ArrayLike, T: ArrayLike, g: ArrayLike, phi: ArrayLike):
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux.

        Args:
            y: Current abundances [atoms], ordered per `self.species.species` - traced.
            T: Temperature [K] - traced.
            g: Gravitational acceleration [m/s2] - traced.
            phi: Total escape mass flux [kg/m2/s] (from the EscapeMechanism) - traced.

        Returns:
            Phi (number flux [atoms/s/m2] for each species, ordered per `self.species.species`),
            phi_c (critical mass flux [kg/s/m2], from Phi_1_2).
        """
        symbols = self.species.species
        H_idx = self.species.index("H")
        He_idx = self.species.index("He")

        H = self.species.scale_heights(T, g)  # ordered per symbols
        mu_H = self.species.atomic_masses[H_idx]
        mu_He = self.species.atomic_masses[He_idx]

        # y_H + y_He == 0 is traced (depends on the evolving abundances), so its guard is
        # resolved via jnp.where rather than a plain Python if - same pattern as Phi_1_2/
        # Phi_minor_species: guard the denominator before dividing so the discarded branch never
        # computes 0/0.
        y_H, y_He = y[H_idx], y[He_idx]
        y_HHe_total = y_H + y_He
        safe_y_HHe_total = jnp.where(y_HHe_total == 0, 1.0, y_HHe_total)
        X1 = jnp.where(y_HHe_total == 0, 0.0, y_H / safe_y_HHe_total)
        X2 = jnp.where(y_HHe_total == 0, 0.0, y_He / safe_y_HHe_total)
        MU = X1 * mu_H + X2 * mu_He
        b_H_He = self.binary_diffusion.get("H", "He", T)

        Phi_H, Phi_He, phi_c = Phi_1_2(phi, b_H_He, H[H_idx], H[He_idx], mu_H, mu_He, X1, X2, MU)

        # H_idx/He_idx/i are all static (Python-int indices fixed by species order, never
        # data-dependent), so building a plain list and stacking at the end is trace-safe -
        # unlike a `Phi = np.zeros(...); Phi[idx] = ...` mutation, which doesn't work on JAX's
        # immutable arrays once phi/y/T are traced.
        values = [None] * len(symbols)
        values[H_idx] = Phi_H
        values[He_idx] = Phi_He
        for i in range(len(symbols)):
            if i in (H_idx, He_idx):
                continue
            values[i] = self._minor_species_flux(Phi_H, Phi_He, H, y, T, i, H_idx, He_idx)
        Phi = jnp.stack(values)

        return Phi, phi_c

    def _minor_species_flux(self, Phi_H, Phi_He, H, y, T, minor_idx, H_idx, He_idx):
        """Thin, self-bound wrapper over the module-level Phi_minor_species."""
        return Phi_minor_species(
            Phi_H,
            Phi_He,
            H[H_idx],
            H[He_idx],
            H[minor_idx],
            y,
            T,
            minor_species_idx=minor_idx,
            light_dominant_idx=H_idx,
            heavy_dominant_idx=He_idx,
            binary_diffusion=self.binary_diffusion,
            species_symbols=self.species.species,
        )


# TODO: Work in progress.  This is the dynamic_phi=True branch
class EscapeNumberFluxDynamic(eqx.Module):
    """Like `EscapeNumberFlux`, but picks whichever two species are currently most abundant as
    the dominant light/heavy diffusive pair (Phi_1_2), instead of always using H/He - every other
    tracked species is still treated as a minor/trace species relative to that pair
    (Phi_minor_species).

    In the original IsoFATE codebase, this corresponds to the ``dynamic_phi==True`` branch.

    NumPy-only: NOT safe to call from a jitted/traced context (`jax.jit`/`eqx.filter_jit`), unlike
    `EscapeNumberFlux`. Selecting the dominant pair here is inherently data-dependent (it depends
    on the current abundances `y`), so - unlike `EscapeNumberFlux`'s always-static H/He indices -
    this uses plain Python `sort()`/`if`/`for` control flow and in-place NumPy array assignment at
    a data-dependent index, neither of which tolerate JAX tracers. Calling this with traced
    `y`/`T`/`g`/`phi` will raise.

    Args:
        binary_diffusion: Binary diffusion coefficient table (defaults to `DEFAULT_BINARY_DIFFUSION`)
        species: Tracked-species registry (defaults to `DEFAULT_SPECIES`)
    """

    species: IsoFATESpecies = DEFAULT_SPECIES
    binary_diffusion: BinaryDiffusionCoefficients = DEFAULT_BINARY_DIFFUSION

    def get_number_flux(self, y: ArrayLike, T: ArrayLike, g: ArrayLike, phi: ArrayLike):
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux.

        Args:
            y: Current abundances [atoms], ordered per `self.species.species` - concrete NumPy,
                not traced (see class docstring).
            T: Temperature [K] - concrete, not traced.
            g: Gravitational acceleration [m/s2] - concrete, not traced.
            phi: Total escape mass flux [kg/m2/s] (from the EscapeMechanism) - concrete, not
                traced.

        Returns:
            Phi (number flux [atoms/s/m2] for each species, ordered per `self.species.species`),
            phi_c (critical mass flux [kg/s/m2], from Phi_1_2).
        """
        symbols = self.species.species
        n_species = len(symbols)
        atomic_masses = self.species.atomic_masses

        N_values = y  # ordered per self.species.species
        abundances_with_idx = [(i, N_values[i]) for i in range(n_species)]
        abundances_with_idx.sort(key=lambda item: item[1], reverse=True)

        most_abundant_idx = abundances_with_idx[0][0]
        second_most_abundant_idx = abundances_with_idx[1][0]

        # Get masses and scale heights for the two most abundant species
        scale_heights = self.species.scale_heights(T, g)

        mass_most = atomic_masses[most_abundant_idx]
        mass_second = atomic_masses[second_most_abundant_idx]
        H_most = scale_heights[most_abundant_idx]
        H_second = scale_heights[second_most_abundant_idx]

        # Determine which is lighter (species 1) and heavier (species 2) by MASS
        if mass_most <= mass_second:
            light_dominant_idx = most_abundant_idx  # species 1 (lighter)
            heavy_dominant_idx = second_most_abundant_idx  # species 2 (heavier)
            mass_1 = mass_most
            mass_2 = mass_second
            H_1 = H_most
            H_2 = H_second
        else:
            light_dominant_idx = second_most_abundant_idx  # species 1 (lighter)
            heavy_dominant_idx = most_abundant_idx  # species 2 (heavier)
            mass_1 = mass_second
            mass_2 = mass_most
            H_1 = H_second
            H_2 = H_most

        # Calculate molar fractions for the two dominant species
        N1 = N_values[light_dominant_idx]  # lightest dominant (species 1)
        N2 = N_values[heavy_dominant_idx]  # heaviest dominant (species 2)
        N_tot_binary = N1 + N2

        X1 = N1 / N_tot_binary  # molar fraction of species 1 in binary mixture
        X2 = N2 / N_tot_binary  # molar fraction of species 2 in binary mixture
        MU = X1 * mass_1 + X2 * mass_2

        # Calculate binary diffusion coefficient between the two dominant species
        light_name = symbols[light_dominant_idx]  # species 1
        heavy_name = symbols[heavy_dominant_idx]  # species 2

        b = self.binary_diffusion.get(light_name, heavy_name, T)

        # Calculate escape fluxes for the two dominant species
        Phi_1_calc, Phi_2_calc, phi_c = Phi_1_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)

        # Assign fluxes to correct species based on light/heavy dominant indices - ordered
        # per self.species.species
        Phi = np.zeros(n_species)
        Phi[light_dominant_idx] = Phi_1_calc
        Phi[heavy_dominant_idx] = Phi_2_calc

        # Calculate fluxes for remaining species using corrected generalized function
        for i in range(n_species):
            if i != light_dominant_idx and i != heavy_dominant_idx:
                Phi[i] = Phi_minor_species(
                    Phi_1_calc,
                    Phi_2_calc,
                    H_1,
                    H_2,
                    scale_heights[i],
                    N_values,
                    T,
                    i,
                    light_dominant_idx,
                    heavy_dominant_idx,
                    binary_diffusion=self.binary_diffusion,
                    species_symbols=symbols,
                )

        return Phi, phi_c
