# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Atmosphere model for IsoFATE."""

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from isofate.constants import const
from isofate.utils import gravitational_acceleration


class AtmosphereModel(eqx.Module):
    """Atmosphere model

    Args:
        adiabatic_index: Adiabatic index. Defaults to ``7/5`` (diatomic ideal gas).
        emission_pressure: Pressure at the emission level/top of atmosphere (Pa). Defaults to
            ``0.2e5`` Pa (0.2 bar).
    """

    adiabatic_index: ArrayLike = 7 / 5
    emission_pressure: float = 0.2e5

    def _atmosphere_descent_vector_field(
        self,
        r: ArrayLike,
        y: tuple[ArrayLike, ArrayLike],
        args: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    ) -> tuple[ArrayLike, Array]:
        """RHS of the atmosphere-descent ODE system.

        Integrated downward in `r` (diffrax's independent/"time" variable here) from the planetary
        radius `rplanet` to the rocky-component radius `r_rocky`, hydrostatically building up the
        atmospheric column on a dry adiabat.

        Args:
            r: Radial coordinate (m) - the integration variable, decreasing from `rplanet` to
                `r_rocky`.
            y: `(p, matm)` - pressure at `r` (Pa) and the atmospheric mass (kg) accumulated so far,
                integrated from the top of the atmosphere down to `r`.
            args: `(emission_temperature, mu, Mc, K)` - emission temperature (K), mean molecular
                mass (kg), planet core mass (kg), and adiabatic exponent
                `(adiabatic_index-1)/adiabatic_index` (dimensionless).

        Returns:
            `(dp_dr, dmatm_dr)`: the hydrostatic pressure gradient and the rate of atmospheric mass
            accumulation, both with respect to `r`.
        """
        p, _ = y
        emission_temperature, mu, Mc, K = args

        # dry adiabat: T is algebraic in p, not itself integrated
        T: ArrayLike = emission_temperature * (p / self.emission_pressure) ** K

        # Below is intentionally kept from the original code (but was always commented out), to
        # presumably test a deep isothermal layer.  It requires rewriting since it is not JAX
        # compliant.
        #
        # include a deep isothermal layer?
        # if p[i-1]>100*1e5:
        #    T[i-1]=T[i]
        # else:
        #    T[i-1]=T[i]/(p[i]/p[i-1])**K

        rho: ArrayLike = p * mu / (const.kb * T)
        g: ArrayLike = gravitational_acceleration(Mc, r)

        dp_dr: ArrayLike = -1 * g * rho
        dmatm_dr: Array = -4 * jnp.pi * jnp.square(r) * rho

        return dp_dr, dmatm_dr

    def make_atmosphere_descent(
        self,
        emission_temperature: ArrayLike,
        mu: ArrayLike,
        rplanet: ArrayLike,
        Mc: ArrayLike,
        r_rocky: ArrayLike,
    ) -> tuple[ArrayLike, Array, ArrayLike]:
        """Atmosphere-descent integration: returns `M_atm`, `T_surf`, `P_surf`.

        The original NumPy implementation's three per-step array updates reduce to a 2-state ODE in
        `r`: `T` is not actually integrated - it's read off the dry adiabat algebraically from `p`
        (`T = emission_temperature*(p/emission_pressure)**K`), and `rho` is a pure function of `p`
        too, so the only genuine ODE states are `p(r)` and the accumulated `M_atm(r)`.

        This integrates with `diffrax.Tsit5()` (5th-order explicit Runge-Kutta) and adaptive step
        sizing (`PIDController`).

        All three outputs are always computed and returned together. Callers that only need a subset
        should just discard the rest (e.g. `_, T_surf, P_surf = make_atmosphere_descent(...)`).

        Args:
            emission_temperature: Emission temperature (K)
            mu: Atomic mass (kg)
            rplanet: Planetary radius (m)
            Mc: Planet core mass (kg)
            r_rocky: Planet rocky-component radius (m) - the integration's lower bound in `r`.

        Returns:
            `(M_atm, T_surf, P_surf)`
        """
        K: ArrayLike = (self.adiabatic_index - 1) / self.adiabatic_index

        gem: ArrayLike = gravitational_acceleration(Mc, rplanet)
        matm0: ArrayLike = 4 * jnp.pi * rplanet**2 * self.emission_pressure / gem

        term = diffrax.ODETerm(self._atmosphere_descent_vector_field)
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=rplanet,  # pyright: ignore[reportArgumentType]
            t1=r_rocky,  # pyright: ignore[reportArgumentType]
            dt0=None,  # let the PIDController choose an initial step adaptively
            y0=(self.emission_pressure, matm0),
            args=(emission_temperature, mu, Mc, K),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
            saveat=diffrax.SaveAt(t1=True),
        )
        P_surf: ArrayLike = sol.ys[0][-1]  # pyright: ignore[reportOptionalSubscript]
        M_atm: ArrayLike = sol.ys[1][-1]  # pyright: ignore[reportOptionalSubscript]
        T_surf: Array = jnp.power((P_surf / self.emission_pressure), K) * emission_temperature

        return M_atm, T_surf, P_surf
