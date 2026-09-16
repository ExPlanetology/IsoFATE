# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX/diffrax-native reimplementations of `isofunks.py` functions.

Home for jit-compatible versions of functions that are otherwise plain Python/NumPy (and so
cannot be traced under `jax.jit`), as they're ported over. See `make_atmosphere_descent_jax`'s
docstring for the first example and the reasoning behind it.
"""

from typing import Literal

import diffrax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from isofate.constants import const
from isofate.isofunks import R_core


def _atmosphere_descent_vector_field(
    r: ArrayLike,
    y: tuple[ArrayLike, ArrayLike],
    args: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
) -> tuple[Array, Array]:
    """RHS of the atmosphere-descent ODE system; see `make_atmosphere_descent_jax`'s docstring
    for the derivation of this from `make_atmosphere_descent`'s original discrete update."""
    p, _matm = y
    Tem, mu, Mc, K, pem = args

    T = Tem * (p / pem) ** K  # dry adiabat: T is algebraic in p, not itself integrated
    rho = p * mu / (const.kb * T)
    g = const.G * Mc / r**2

    dp_dr = -g * rho
    dmatm_dr = -4 * jnp.pi * r**2 * rho

    return dp_dr, dmatm_dr


def make_atmosphere_descent_jax(
    Tem: ArrayLike,
    mu: ArrayLike,
    rplanet: ArrayLike,
    Mc: ArrayLike,
    gamma: ArrayLike,
    output_mode: Literal[1, 2],
) -> Array | tuple[Array, Array]:
    """JAX/diffrax-jittable equivalent of `isofunks.make_atmosphere_descent`.

    `make_atmosphere_descent`'s three per-step array updates reduce to a 2-state ODE in `r`:
    `T` is not actually integrated - it's read off the dry adiabat algebraically from `p`
    (`T = Tem*(p/pem)**K`), and `rho` is a pure function of `p` too, so the only genuine ODE
    states are `p(r)` and the accumulated `Matm(r)`. This integrates that system with
    `diffrax.Euler()` at the same fixed step size and step count (249 steps over a 250-point
    grid) as the original NumPy loop, so it should reproduce it to near machine precision -
    see the validation script (not part of this module) comparing the two directly.

    Unlike the original, `output_mode=0` (the full radial `T`/`p` profile) is not implemented,
    since nothing in the codebase actually calls it that way.

    Args:
        Tem: Emission temperature (K)
        mu: Atomic mass (kg)
        rplanet: Planetary radius (m)
        Mc: Planet core mass (kg)
        gamma: Adiabatic index (dimensionless)
        output_mode: ``1`` for `Matm` only, ``2`` for `(Tsurf, psurf)`. Must be a static
            (non-traced) Python value, exactly as in the original.

    Returns:
        See `output_mode` above.
    """
    nr = 250
    rc = R_core(Mc)
    pem = 0.2e5
    K = (gamma - 1) / gamma
    dr = (rplanet - rc) / (nr - 1)

    gem = const.G * Mc / rplanet**2
    matm0 = 4 * jnp.pi * rplanet**2 * pem / gem

    term = diffrax.ODETerm(_atmosphere_descent_vector_field)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Euler(),
        t0=rplanet,
        t1=rc,
        dt0=-dr,
        y0=(pem, matm0),
        args=(Tem, mu, Mc, K, pem),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(t1=True),
    )
    p_final = sol.ys[0][-1]
    matm_final = sol.ys[1][-1]

    if output_mode == 1:
        return matm_final
    elif output_mode == 2:
        Tsurf = Tem * (p_final / pem) ** K
        return Tsurf, p_final
    else:
        raise NotImplementedError(
            "output_mode=0 (full radial profile) is not implemented in the jax version"
        )
