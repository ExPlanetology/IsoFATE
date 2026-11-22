# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preset Star/Planet instances for known and example systems."""

from isofate.constants import const
from isofate.system import Planet, Star

STARS: dict[str, Star] = {
    "Sun-like": Star(radius=const.Rs, mass=const.Ms, temperature=6000),
    "M1": Star(radius=0.5 * const.Rs, mass=0.5 * const.Ms, temperature=3600),
    "K5": Star(radius=0.7 * const.Rs, mass=0.7 * const.Ms, temperature=4440),
    "LHS 1140": Star(
        radius=0.22 * const.Rs, mass=0.18 * const.Ms, temperature=3096
    ),  # , luminosity=0.0038 * const.Ls
    "Kepler-138": Star(radius=0.535 * const.Rs, mass=0.535 * const.Ms, temperature=3841),
    "K2-3": Star(radius=0.546 * const.Rs, mass=0.549 * const.Ms, temperature=3844),
    "GJ 3090": Star(radius=0.516 * const.Rs, mass=0.519 * const.Ms, temperature=3556),
    "K2-18": Star(radius=0.469 * const.Rs, mass=0.495 * const.Ms, temperature=3500),
    "HAT-P-11": Star(radius=0.752 * const.Rs, mass=0.809 * const.Ms, temperature=4780),  # K4V
    "TRAPPIST-1": Star(radius=0.1192 * const.Rs, mass=0.0898 * const.Ms, temperature=2566),
}
"""Catalogue of the preset stars, by name: generic stellar types and known host stars.

Select one with e.g. ``STARS["LHS 1140"]``, or list the names with ``list(STARS)``.
"""

PLANETS: dict[str, Planet] = {
    # f_atm, F_final in the comments are the run settings or data used with each planet
    # (not Planet fields), kept for reference
    "O2 planet": Planet(
        mass=10.110718251912926 * const.Me, period=1.1579370531867088 / const.s2day
    ),  # f_atm = 1.46472048e-01
    "O2 world": Planet(
        mass=12.408678941478096 * const.Me, period=2.28472886 / const.s2day
    ),  # f_atm = 2.17823165e-02
    "K2-18 b": Planet(mass=8.63 * const.Me, period=33 / const.s2day),  # f_atm = 0.001
    "LHS 1140 b": Planet(mass=5.6 * const.Me, period=24.74 / const.s2day),  # f_atm = 0.00085
    "GJ 3090 b": Planet(mass=3.34 * const.Me, period=2.9 / const.s2day),  # f_atm = 0.03
    "TRAPPIST-1 c": Planet(
        mass=1.308 * const.Me, period=2.42 / const.s2day
    ),  # f_atm = 0.0265, F_final = 1e2*Fe_xuv
    "TRAPPIST-1 f": Planet(
        mass=1.039 * const.Me, period=9.21 / const.s2day
    ),  # f_atm = 0.005, F_final = 1e2*Fe_xuv
    "TRAPPIST-1 g": Planet(mass=1.321 * const.Me, period=12.35 / const.s2day),  # f_atm = 0.005
    "TRAPPIST-1 h": Planet(mass=0.326 * const.Me, period=18.77 / const.s2day),  # f_atm = 0.02
    "N2 dynamic test planet": Planet(
        mass=2.0063872276215733 * const.Me, period=2.7202404227417087 / const.s2day
    ),  # f_atm = 0.0398
    "Kepler-138 d": Planet(mass=2.1 * const.Me, period=23 / const.s2day),  # f_atm = 0.012
    "K2-3 c": Planet(mass=2.68 * const.Me, period=24.6 / const.s2day),  # f_atm = 0.015
}
"""Catalogue of the preset planets, by name: known planets and hypothetical test cases.

Select one with e.g. ``PLANETS["LHS 1140 b"]``, or list the names with ``list(PLANETS)``. The
period fixes the orbit only together with a star (see ``STARS``).
"""

# TODO: not yet in PLANETS
# IL world
# f_atm = 0.001049211388465194
# Mp = 1.2474905792906028*Me
# P = 86.23173213822659
# f_atm = 2.00655135e-03
# Mp = 1.3615101475752467*Me
# P = 51.53009252/s2day

# HAT-P-11 b
# f_atm = 0.13
# Mp = 0.081*Mjup
# Mp = 0.073*Mjup
# P = 4.8878162/s2day
# P = Period(0.0425*au2m, M_star) # perihelion
# P = Period(0.0635*au2m, M_star) # aphelion
