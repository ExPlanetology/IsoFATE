# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Canonical registry of the elements tracked by IsoFATE's escape/interior model.

Several places in isocalc()/isofunks.py need the same 7 species, in the same order, either as a
list of atomic masses or as a list of symbols for name-based lookups (e.g. get_binary_diffusion_coeff).
Previously each of those sites hand-typed its own copy of the list; this module is the single
source of truth they should all import from instead.
"""

from dataclasses import dataclass

from isofate.constants import const


@dataclass(frozen=True)
class Element:
    symbol: str
    mass: float  # atomic mass [kg/atom]


ELEMENTS: tuple[Element, ...] = (
    Element("H", const.mu_H),
    Element("He", const.mu_He),
    Element("D", const.mu_D),
    Element("O", const.mu_O),
    Element("C", const.mu_C),
    Element("N", const.mu_N),
    Element("S", const.mu_S),
)

SYMBOLS: tuple[str, ...] = tuple(e.symbol for e in ELEMENTS)
ATOMIC_MASSES: tuple[float, ...] = tuple(e.mass for e in ELEMENTS)
MASS_BY_SYMBOL: dict[str, float] = {e.symbol: e.mass for e in ELEMENTS}
