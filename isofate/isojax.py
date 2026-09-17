# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX/diffrax-native reimplementations of `isofunks.py` functions.

Home for jit-compatible versions of functions that are otherwise plain Python/NumPy (and so
cannot be traced under `jax.jit`), as they're ported over. See `make_atmosphere_descent_jax`'s
docstring for the first example and the reasoning behind it.
"""

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
    for the derivation of this from the original discrete update it replaces (formerly
    `isofunks.make_atmosphere_descent`, removed as dead code once this JAX port replaced its one
    live call site).
    """
    p, _matm = y
    Tem, mu, Mc, K, pem = args

    T = Tem * (p / pem) ** K  # dry adiabat: T is algebraic in p, not itself integrated

    # Below is intentionally kept (but was always commented out) in the original code, to
    # presumably test a deep isothermal layer.  It will not work as is (not JAX compliant), but is
    # left here for reference.
    #
    # include a deep isothermal layer?
    # if p[i-1]>100*1e5:
    #    T[i-1]=T[i]
    # else:
    #    T[i-1]=T[i]/(p[i]/p[i-1])**K

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
) -> tuple[Array, Array, Array]:
    """JAX/diffrax-jittable atmosphere-descent integration: returns `Matm`, `Tsurf`, `Psurf`.

    The original NumPy implementation's three per-step array updates reduce to a 2-state ODE in
    `r`: `T` is not actually integrated - it's read off the dry adiabat algebraically from `p`
    (`T = Tem*(p/pem)**K`), and `rho` is a pure function of `p` too, so the only genuine ODE
    states are `p(r)` and the accumulated `Matm(r)`. This integrates that system with
    `diffrax.Euler()` at the same fixed step size and step count (249 steps over a 250-point
    grid) as the original NumPy loop, so it reproduces it to near machine precision (verified
    directly against the original before it was removed: ~1e-12 to 1e-14 relative agreement).

    All three outputs are always computed and returned together - the original's `output_mode`
    switch (`1` for `Matm` only, `2` for `(Tsurf, psurf)`, `0` for the full radial profile, never
    used anywhere) added no real savings here, since `Tsurf`/`Psurf` are cheap algebraic
    byproducts of the same integration. Callers that only need a subset should just discard the
    rest (e.g. `_, Tsurf, Psurf = make_atmosphere_descent_jax(...)`).

    Args:
        Tem: Emission temperature (K)
        mu: Atomic mass (kg)
        rplanet: Planetary radius (m)
        Mc: Planet core mass (kg)
        gamma: Adiabatic index (dimensionless)

    Returns:
        `(Matm, Tsurf, Psurf)`
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
    Tsurf = Tem * (p_final / pem) ** K

    return matm_final, Tsurf, p_final
