# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Alternate/experimental deuterium number-flux formulations.

Kept for reference only - not called anywhere in the codebase.

These functions are likely not JAX-compliant as written and should not be used as-is elsewhere
in the codebase. They also hard-code their own binary diffusion coefficients (b_H_D/b_H_He/
b_He_D), rather than going through `isofate.species.BinaryDiffusionCoefficients`, the codebase's
single source of truth for these elsewhere.
"""


def Phi_D_GC23(Phi_H, Phi_He, H_H, H_D, H_He, N_H, N_He, N_D, T):
    """Calculates the number flux of deuterium for simultaneous calculation of H/He/D escape.

    From Gu & Chen 2023

    Args:
        Phi_H: number flux of H [particles/m2/s]
        Phi_He: number flux of He [particles/m2/s]
        H_H: scale height of H [m]
        H_D: scale height of D [m]
        H_He: scale height of He [m]
        N_H: number of H particles [atoms]
        N_He: number of He particles [atoms]
        N_D: number of D particles [atoms]
        T: equilibrium temperature [K]

    Returns:
        Number flux of D [particles/m2/s]
    """
    # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_D = 7.183e19 * T**0.728
    # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_H_He = 1.04e20 * T**0.732
    # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription
    b_He_D = 5.087e19 * T**0.728
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


def Phi_D_Z90_mod(Phi_H, H_D, N_D, N_H, T):
    """Phi_D_Z90 solution with He set to zero.

    Number flux deuterium derived from Zahnle et al 1990

    Args:
        Phi_H: number flux of H [particles/m2/s]
        H_D: scale height of D [m]
        N_D: number of D particles [atoms]
        N_H: number of H particles [atoms]
        T: equilibrium temperature [K]

    Returns:
        Number flux of D [particles/m2/s]
    """
    # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_D = 7.183e19 * T**0.728
    Phi_DL_D = b_H_D / H_D
    f_D = N_D / N_H

    return f_D * Phi_H - (N_D / (N_D + N_H)) * Phi_DL_D


def Phi_D_Z90_mod2(Phi_H, Phi_He, H_H, H_D, H_He, N_H, N_He, N_D, T):
    """Phi_D solution from referee report Cherubim et al 2024.

    Args:
        Phi_H: number flux of H [particles/m2/s]
        Phi_He: number flux of He [particles/m2/s]
        H_H: scale height of H [m]
        H_D: scale height of D [m]
        H_He: scale height of He [m]
        N_H: number of H particles [atoms]
        N_He: number of He particles [atoms]
        N_D: number of D particles [atoms]
        T: equilibrium temperature [K]

    Returns:
        Number flux of D [particles/m2/s]
    """
    del H_H  # used, but presumably was required for consistency with the interface

    # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_D = 7.183e19 * T**0.728
    # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_H_He = 1.04e20 * T**0.732
    # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription (Appendix C)
    b_He_D = 5.087e19 * T**0.728
    alpha_3 = b_H_D / b_He_D
    Phi_DL_D = b_H_D / H_D
    Phi_DL_He = b_H_He / H_He
    x_He = N_He / (N_H + N_He + N_D)
    f_He = N_He / N_H
    f_D = N_D / N_H
    num = Phi_DL_He * x_He - Phi_DL_D + Phi_H + alpha_3 * Phi_He
    denom = 1 + alpha_3 * f_He

    return max(0, f_D * num / denom)
