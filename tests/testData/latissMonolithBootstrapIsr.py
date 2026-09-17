# This file is part of donut_viz.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Config override for running LatissMonolithTask against ts_wep's gen3TestRepo,
# which carries only the curated LATISS calibrations (camera, defects,
# crosstalk). The production configuration needs bias, dark, flat, linearizer
# and ptc (~320 MB), so the test runs IsrTaskLSST in bootstrap mode instead:
# no gains, no bias/dark/flat, and the amp-level checks that require a PTC
# switched off. Defect masking and crosstalk correction stay on, which is what
# donut detection depends on. Zernikes differ from the production ISR by a few
# tens of nm; the test pins the bootstrap values.
from typing import Any

import numpy as np

# pex_config runs this file as exec(code, {"__file__": ...}, {"config": cfg}),
# so `config` is a pre-bound *local*. Rebinding it through locals() keeps
# ruff (F821) and mypy (name-defined) from flagging the injected name.
config: Any = locals()["config"]

config.isrTask.doBootstrap = True
config.isrTask.doApplyGains = False
config.isrTask.doCorrectGains = False
config.isrTask.ampNoiseThreshold = np.inf
config.isrTask.serialOverscanMedianShiftSigmaThreshold = np.inf
config.isrTask.doBias = False
config.isrTask.doDark = False
config.isrTask.doFlat = False
config.isrTask.doLinearize = False
