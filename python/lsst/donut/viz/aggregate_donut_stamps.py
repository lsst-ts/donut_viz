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

import typing
from typing import Any, cast

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.utils.timer import timeMethod

from .utilities import intra_focal_ids

__all__ = [
    "AggregateDonutStampsTaskConnections",
    "AggregateDonutStampsTaskConfig",
    "AggregateDonutStampsTask",
]


class AggregateDonutStampsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    donutStampsIntra = ct.Input(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
        deferGraphConstraint=True,
    )
    donutStampsExtra = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
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
    donutStampsIntraVisit = ct.Output(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraVisit",
    )
    donutStampsExtraVisit = ct.Output(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtraVisit",
    )

    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """This will raise if any inputs have an intra-focal data id."""
        for data_id in adjuster.iter_data_ids():
            inputs = adjuster.get_inputs(data_id)
            for connection_name, input_data_ids in inputs.items():
                for input_data_id in input_data_ids:
                    detector = input_data_id.get("detector")
                    if detector is not None and detector in intra_focal_ids:
                        raise RuntimeError(
                            f"Input data id {input_data_id} has intra-focal detector; "
                            "this is not allowed for AggregateDonutStampsTask."
                        )


class AggregateDonutStampsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutStampsTaskConnections,  # type: ignore
):
    maxDonutsPerDetector: pexConfig.Field = pexConfig.Field(
        doc="Maximum number of donuts to use per detector",
        default=1,
        dtype=int,
    )

    def validate(self) -> None:
        if self.maxDonutsPerDetector < 1:
            raise pexConfig.FieldValidationError(
                self, "maxDonutsPerDetector", "maxDonutsPerDetector must be at least 1"
            )


class AggregateDonutStampsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutStampsTaskConfig
    _DefaultName = "AggregateDonutStamps"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: AggregateDonutStampsTaskConfig = cast(AggregateDonutStampsTaskConfig, self.config)

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Make robust to missing data: if any of the inputs are missing
        # for a given detector, that detector will be skipped.
        intraStampsDict = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.donutStampsIntra}
        extraStampsDict = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.donutStampsExtra}
        qualityTablesDict = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.qualityTables}
        intersectingDetectors = (
            set(intraStampsDict.keys()) & set(extraStampsDict.keys()) & set(qualityTablesDict.keys())
        )
        stampsOut = self.run(
            [intraStampsDict[detector] for detector in intersectingDetectors],
            [extraStampsDict[detector] for detector in intersectingDetectors],
            [qualityTablesDict[detector] for detector in intersectingDetectors],
        )

        butlerQC.put(
            stampsOut.intra,
            outputRefs.donutStampsIntraVisit,
        )

        butlerQC.put(
            stampsOut.extra,
            outputRefs.donutStampsExtraVisit,
        )

    @timeMethod
    def run(
        self,
        intraStamps: typing.List,
        extraStamps: typing.List,
        qualityTables: typing.List,
    ) -> pipeBase.Struct:
        """Aggregate donut stamps for a set of visits.

        Parameters
        ----------
        intraStamps : list of DonutStamps
            List of intrafocal donut stamps.
        extraStamps : list of DonutStamps
            List of extrafocal donut stamps.
        qualityTables : list of `astropy.table.Table`
            List of donut quality tables.

        Returns
        -------
        struct
            Struct with `intra` and `extra` donut stamps.
        """
        intraStampsList = []
        extraStampsList = []
        intraStampsMetadata = None
        extraStampsMetadata = None
        for intra, extra, quality in zip(intraStamps, extraStamps, qualityTables):
            # Skip if quality table is empty.
            if len(quality) == 0:
                continue

            # Load the quality table and determine which donuts were selected
            intraQualitySelect = quality[quality["DEFOCAL_TYPE"] == "intra"]["FINAL_SELECT"]
            extraQualitySelect = quality[quality["DEFOCAL_TYPE"] == "extra"]["FINAL_SELECT"]

            # Select donuts used in Zernike estimation
            intraStampsSelect = DonutStamps([intra[i] for i in range(len(intra)) if intraQualitySelect[i]])
            extraStampsSelect = DonutStamps([extra[i] for i in range(len(extra)) if extraQualitySelect[i]])

            if intraStampsMetadata is None or extraStampsMetadata is None:
                # Create metadata for stamps
                # Only keep the visit level data
                # For stamp-level metadata look at the
                # metadata of the individual stamps
                intraStampsMetadata = dafBase.PropertyList()
                extraStampsMetadata = dafBase.PropertyList()
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
                    intraStampsMetadata[key] = intra.metadata[key]
                    extraStampsMetadata[key] = extra.metadata[key]

            # Append the requested number of donuts
            maxKeep = min(len(intraStampsSelect), len(extraStampsSelect), self.config.maxDonutsPerDetector)
            intraStampsList.append(intraStampsSelect[:maxKeep])
            extraStampsList.append(extraStampsSelect[:maxKeep])

        intraStampsListRavel = [stamp for stampList in intraStampsList for stamp in stampList]
        extraStampsListRavel = [stamp for stampList in extraStampsList for stamp in stampList]

        intraStampsRavel = DonutStamps(intraStampsListRavel, metadata=intraStampsMetadata)
        extraStampsRavel = DonutStamps(extraStampsListRavel, metadata=extraStampsMetadata)

        return pipeBase.Struct(intra=intraStampsRavel, extra=extraStampsRavel)
