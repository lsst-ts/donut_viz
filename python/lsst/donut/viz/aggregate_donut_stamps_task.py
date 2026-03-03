import typing
from typing import Any, cast

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateDonutStampsTaskConnections",
    "AggregateDonutStampsTaskConfig",
    "AggregateDonutStampsTask",
    "AggregateDonutStampsUnpairedTaskConnections",
    "AggregateDonutStampsUnpairedTaskConfig",
    "AggregateDonutStampsUnpairedTask",
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
        stampsOut = self.run(
            butlerQC.get(inputRefs.donutStampsIntra),
            butlerQC.get(inputRefs.donutStampsExtra),
            butlerQC.get(inputRefs.qualityTables),
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
            intraStampsList.append(intraStampsSelect[: self.config.maxDonutsPerDetector])
            extraStampsList.append(extraStampsSelect[: self.config.maxDonutsPerDetector])

        intraStampsListRavel = [stamp for stampList in intraStampsList for stamp in stampList]
        extraStampsListRavel = [stamp for stampList in extraStampsList for stamp in stampList]

        intraStampsRavel = DonutStamps(intraStampsListRavel, metadata=intraStampsMetadata)
        extraStampsRavel = DonutStamps(extraStampsListRavel, metadata=extraStampsMetadata)

        return pipeBase.Struct(intra=intraStampsRavel, extra=extraStampsRavel)


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
        stampsOut = self.run(
            butlerQC.get(inputRefs.donutStampsIn),
            butlerQC.get(inputRefs.qualityTables),
        )

        butlerQC.put(
            stampsOut.stamps,
            outputRefs.donutStampsUnpairedVisit,
        )

    @timeMethod
    def run(
        self,
        stampsIn: typing.List,
        qualityTables: typing.List,
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
        for stamps, quality in zip(stampsIn, qualityTables):
            # Skip if quality table is empty.
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
