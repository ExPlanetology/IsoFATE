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
from jaxtyping import ArrayLike

from isofate.species import (
    DEFAULT_BINARY_DIFFUSION,
    SYMBOLS,
    BinaryDiffusionCoefficients,
    IsoFATESpecies,
)

# number flux of light and heavy species


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
        phi: mass flux [kg/m2/s] - traced
        b: binary diffusion coefficient [particles/m/s] - traced
        H1/H2: scale heights of light/heavy species [m] - traced
        m1/m2: molecular mass of light/heavy species [kg/particle] - traced
        x1/x2: molar concentration of light/heavy species (x1=mol_1/mol_tot) [ndim] - traced
        mu: average atmospheric atomic mass [kg/particle] - traced

    Returns:
        number flux of light species [particles/m2/s], number flux of heavy species
        [particles/m2/s], critical mass flux [kg/s/m2]
    """
    phi_c = b * x1 * (m2 - m1) / H1  # critical mass flux [kg/s/m2]

    # mu==0 and phi<phi_c both depend on traced quantities (mu evolves with the abundances every
    # timestep, unlike e.g. Fxuv's t_pms, which is a fixed config value resolvable in plain
    # Python) - both branches are computed unconditionally and selected via jnp.where. mu is
    # guarded before dividing so the discarded mu==0 branch never computes a 0/0 division, which
    # would otherwise corrupt jax.grad through the jnp.where even though it's never selected.
    safe_mu = jnp.where(mu == 0, 1.0, mu)

    Phi1_below_critical = phi / m1
    Phi2_below_critical = 0.0
    Phi1_above_critical = (x1 * phi + x1 * x2 * (m2 - m1) * b / H2) / safe_mu
    Phi2_above_critical = (x2 * phi + x1 * x2 * (m1 - m2) * b / H1) / safe_mu

    below_critical = phi < phi_c
    Phi1 = jnp.where(below_critical, Phi1_below_critical, Phi1_above_critical)
    Phi2 = jnp.where(below_critical, Phi2_below_critical, Phi2_above_critical)

    Phi1 = jnp.where(mu == 0, 0.0, Phi1)
    Phi2 = jnp.where(mu == 0, 0.0, Phi2)

    return Phi1, Phi2, phi_c


# number flux deuterium Gu & Chen 2023


def Phi_D_GC23(Phi_H, Phi_He, H_H, H_D, H_He, N_H, N_He, N_D, T):
    """
    Calculates number flux of deuterium for simultaneous calculation of H/He/D escape
    From Gu & Chen 2023

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
    """
    Calculates number flux for minor species using correct Zahnle et al. 1990 formulation

    Inputs:
        - Phi_1: number flux of lightest dominant species [atoms/s/m2]
        - Phi_2: number flux of heaviest dominant species [atoms/s/m2]
        - H_1: scale height of lightest dominant species [m]
        - H_2: scale height of heaviest dominant species [m]
        - H_minor: scale height of minor species [m]
        - N_values: list [N_H, N_He, N_D, N_O, N_C, N_N, N_S] - current abundances
        - T: temperature [K]
        - minor_species_idx: index (0-4) of the minor species being calculated
        - light_dominant_idx: index of lightest dominant species (species 1)
        - heavy_dominant_idx: index of heaviest dominant species (species 2)
        - binary_diffusion: Binary diffusion coefficient lookup to use.
        - species_symbols: Symbols that `minor_species_idx`/`light_dominant_idx`/
          `heavy_dominant_idx` index into.
    """
    if sum(N_values) == 0:
        return 0

    minor_name = species_symbols[minor_species_idx]
    light_name = species_symbols[light_dominant_idx]  # species 1
    heavy_name = species_symbols[heavy_dominant_idx]  # species 2

    # Get binary diffusion coefficients
    b_1_minor = binary_diffusion.get(light_name, minor_name, T)  # b between species 1 and minor
    b_2_minor = binary_diffusion.get(heavy_name, minor_name, T)  # b between species 2 and minor
    b_1_2 = binary_diffusion.get(light_name, heavy_name, T)  # b between species 1 and 2

    alpha_2 = b_1_minor / b_1_2 if b_1_2 != 0 else 1  # b_1_minor/b_1_2
    alpha_3 = b_1_minor / b_2_minor if b_2_minor != 0 else 1  # b_1_minor/b_2_minor
    Phi_DL_minor = b_1_minor * (1 / H_minor - 1 / H_1)
    Phi_DL_2 = b_1_2 * (1 / H_2 - 1 / H_1)

    N_1 = N_values[light_dominant_idx]  # lightest dominant species
    N_2 = N_values[heavy_dominant_idx]  # heaviest dominant species
    N_minor = N_values[minor_species_idx]

    if N_1 == 0:
        return 0

    f_2 = N_2 / N_1  # N_2/N_1 (heavy/light dominant)
    f_minor = N_minor / N_1  # N_minor/N_1

    # Calculate molar fraction of species 2 in total atmosphere
    N_total = sum(N_values)
    x_2 = N_2 / N_total if N_total > 0 else 0

    # Zahnle et al. 1990 formulation
    num = Phi_1 - Phi_DL_minor + alpha_2 * Phi_DL_2 * x_2 + alpha_3 * Phi_2
    denom = 1 + alpha_3 * f_2

    return max(0, f_minor * num / denom)


class EscapeNumberFlux(eqx.Module):
    """Bundles a BinaryDiffusionCoefficients table and an IsoFATESpecies registry to compute the
    per-species escape number flux in one call, given the current abundances and physical state.

    H and He are always treated as the light/heavy diffusive pair (Phi_1_2); every other tracked
    species is treated as a minor/trace species relative to that pair (Phi_minor_species).

    Not an `eqx.Module`: this is called once per timestep in isocalc's still-plain-Python loop,
    and eqx.Module wraps every bound-method access in a fresh BoundMethod pytree.
    """

    binary_diffusion: BinaryDiffusionCoefficients
    species: IsoFATESpecies

    def get_number_flux(self, y, T, g, phi):
        """Number flux [atoms/s/m2] for every tracked species, plus the critical mass flux.

        Args:
            y: Current abundances [atoms], ordered per `self.species.species`.
            T: Temperature [K].
            g: Gravitational acceleration [m/s2].
            phi: Total escape mass flux [kg/m2/s] (from the EscapeMechanism).

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

        y_H, y_He = y[H_idx], y[He_idx]
        if y_H + y_He == 0:
            X1, X2 = 0, 0
        else:
            X1 = y_H / (y_H + y_He)
            X2 = y_He / (y_H + y_He)
        MU = X1 * mu_H + X2 * mu_He
        b_H_He = self.binary_diffusion.get("H", "He", T)

        Phi_H, Phi_He, phi_c = Phi_1_2(phi, b_H_He, H[H_idx], H[He_idx], mu_H, mu_He, X1, X2, MU)

        Phi = np.zeros(len(symbols))
        Phi[H_idx] = Phi_H
        Phi[He_idx] = Phi_He
        for i in range(len(symbols)):
            if i in (H_idx, He_idx):
                continue
            Phi[i] = self._minor_species_flux(Phi_H, Phi_He, H, y, T, i, H_idx, He_idx)

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
