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

from typing import Any, cast

import lsst.daf.base as dafBase
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.utils.timer import timeMethod

from .aggregate_donut_stamps import AggregateDonutStampsTaskConfig

__all__ = [
    "AggregateDonutStampsUnpairedTaskConnections",
    "AggregateDonutStampsUnpairedTaskConfig",
    "AggregateDonutStampsUnpairedTask",
]


class AggregateDonutStampsUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    donutStampsIn = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
        multiple=True,
        deferGraphConstraint=True,
    )
    qualityTables = ct.Input(
        doc="Donut quality tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
        multiple=True,
        deferGraphConstraint=True,
    )
    donutStampsUnpairedVisit = ct.Output(
        doc="All Donut Stamps for unpaired estimation",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsUnpairedVisit",
    )


class AggregateDonutStampsUnpairedTaskConfig(
    AggregateDonutStampsTaskConfig,
    pipelineConnections=AggregateDonutStampsUnpairedTaskConnections,  # type: ignore
):
    pass


class AggregateDonutStampsUnpairedTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutStampsUnpairedTaskConfig
    _DefaultName = "AggregateDonutStampsUnpaired"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: AggregateDonutStampsUnpairedTaskConfig = cast(
            AggregateDonutStampsUnpairedTaskConfig,
            self.config,
        )

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        stampsIn = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.donutStampsIn}
        qualityTables = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.qualityTables}
        stampsOut = self.run(stampsIn, qualityTables)

        butlerQC.put(
            stampsOut.stamps,
            outputRefs.donutStampsUnpairedVisit,
        )

    @timeMethod
    def run(
        self,
        stampsIn: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut stamps for a set of visits.

        Parameters
        ----------
        stampsIn : list of DonutStamps
            List of donut stamps.
        qualityTables : list of `astropy.table.Table`
            List of donut quality tables.

        Returns
        -------
        struct
            Struct with `stamps` donut stamps.
        """
        stampsList = []
        stampsMetadata = None

        for detId in stampsIn.keys():
            # Skip if quality table is empty.
            stamps = stampsIn.get(detId)
            quality = qualityTables.get(detId)
            # If the detector doesn't have stamps or a quality table move on
            if stamps is None:
                self.log.warning(f"Missing stamps for detector {detId}, skipping.")
                continue
            if quality is None:
                self.log.warning(f"Missing quality table for detector {detId}, skipping.")
                continue
            # If no quality sources in quality table move on
            if len(quality) == 0:
                continue

            # Load the quality table and determine which donuts were selected
            qualitySelect = quality["FINAL_SELECT"]

            # Select donuts used in Zernike estimation
            stampsSelect = DonutStamps([stamps[i] for i in range(len(stamps)) if qualitySelect[i]])

            if stampsMetadata is None:
                # Create metadata for stamps
                # Only keep the visit level data
                # For stamp-level metadata look at the
                # metadata of the individual stamps
                stampsMetadata = dafBase.PropertyList()
                visitKeys = [
                    "VISIT",
                    "BORESIGHT_ROT_ANGLE_RAD",
                    "BORESIGHT_PAR_ANGLE_RAD",
                    "BORESIGHT_ALT_RAD",
                    "BORESIGHT_AZ_RAD",
                    "BORESIGHT_RA_RAD",
                    "BORESIGHT_DEC_RAD",
                    "MJD",
                    "BANDPASS",
                ]
                for key in visitKeys:
                    stampsMetadata[key] = stamps.metadata[key]

            # Append the requested number of donuts
            stampsList.append(stampsSelect[: self.config.maxDonutsPerDetector])

        stampsListRavel = [stamp for stampList in stampsList for stamp in stampList]

        stampsRavel = DonutStamps(stampsListRavel, metadata=stampsMetadata)

        return pipeBase.Struct(stamps=stampsRavel)
