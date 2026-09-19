# SPDX-FileCopyrightText: 2026 Collin Cherubim <collinc@uchicago.edu>
# SPDX-FileCopyrightText: 2026 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IsoFATE API"""

import logging

import equinox as eqx

from isofate.parameters import Parameters

logger: logging.Logger = logging.getLogger(__name__)


class IsoFATEModel(eqx.Module):
    """IsoFATE model for simulating atmospheric escape and evolution.

    Args:
        parameters: Parameters object containing system, escape mechanism, and options.
    """

    parameters: Parameters

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        logger.info("IsoFATE model initialized with provided parameters.")

    def run_simulation(self):
        """Run the IsoFATE simulation based on the provided parameters."""
        logger.info("Starting simulation...")
        # Simulation logic would go here
        logger.info("Simulation completed.")
