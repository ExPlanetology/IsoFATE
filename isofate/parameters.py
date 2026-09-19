# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import equinox as eqx

from isofate.escape import EscapeMechanism
from isofate.escape_core import EscapeNumberFlux
from isofate.options import IsocalcOptions
from isofate.system import System


class Parameters(eqx.Module):
    system: System
    escape_mechanism: EscapeMechanism
    escape_number_flux: EscapeNumberFlux = EscapeNumberFlux()
    isocalc_options: IsocalcOptions = IsocalcOptions()
