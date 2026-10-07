# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Original (scalar, per-species) functions still used by the legacy `isocalc` driver. They will
be removed once the JAX drivers fully replace it; their vectorised counterparts are in
`isofate.escape.fractionation`.
"""

import jax.numpy as jnp
from atmodeller.jax_utils import safe_divide
from jax import Array
from jaxtyping import ArrayLike

from isofate.species import DEFAULT_BINARY_DIFFUSION, SYMBOLS, BinaryDiffusionCoefficients


# TODO: Original function still being used by non-JAX branch. Will be removed once the JAX branch
# is fully swapped in and tested.
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

    phi1_below_critical: ArrayLike = phi / m1
    phi2_below_critical: ArrayLike = 0.0
    phi1_above_critical: ArrayLike = safe_divide(x1 * phi + x1 * x2 * (m2 - m1) * b / H2, mu)  # pyright: ignore[reportOperatorIssue]
    phi2_above_critical: ArrayLike = safe_divide(x2 * phi + x1 * x2 * (m1 - m2) * b / H1, mu)  # pyright: ignore[reportOperatorIssue]

    below_critical: Array = phi < phi_c  # pyright: ignore
    phi1: Array = jnp.where(below_critical, phi1_below_critical, phi1_above_critical)
    phi2: Array = jnp.where(below_critical, phi2_below_critical, phi2_above_critical)

    phi1 = jnp.where(mu == 0, 0.0, phi1)
    phi2 = jnp.where(mu == 0, 0.0, phi2)

    return phi1, phi2, phi_c


# TODO: Eventually this can be removed once the JAX version is fully swapped in
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
    # on T), so the zero-guards are resolved via safe_divide rather than a plain Python ternary.
    alpha_2 = safe_divide(b_1_minor, b_1_2, fallback=1.0)  # b_1_minor/b_1_2
    alpha_3 = safe_divide(b_1_minor, b_2_minor, fallback=1.0)  # b_1_minor/b_2_minor

    Phi_DL_minor = b_1_minor * (1 / H_minor - 1 / H_1)
    Phi_DL_2 = b_1_2 * (1 / H_2 - 1 / H_1)

    N_1 = N_values[light_dominant_idx]  # lightest dominant species
    N_2 = N_values[heavy_dominant_idx]  # heaviest dominant species
    N_minor = N_values[minor_species_idx]
    N_total = sum(N_values)

    f_2 = safe_divide(N_2, N_1)  # N_2/N_1 (heavy/light dominant)
    f_minor = safe_divide(N_minor, N_1)  # N_minor/N_1

    # Calculate molar fraction of species 2 in total atmosphere
    x_2 = safe_divide(N_2, N_total)

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
