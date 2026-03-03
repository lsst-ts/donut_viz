from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.table import vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.geom import Point2D, radians
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.pairTask import ExposurePairer
from lsst.ts.wep.utils import convertDictToVisitInfo
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateDonutTablesTaskConnections",
    "AggregateDonutTablesTaskConfig",
    "AggregateDonutTablesTask",
    "AggregateDonutTablesUnpairedCwfsTaskConfig",
    "AggregateDonutTablesUnpairedCwfsTask",
    "AggregateDonutTablesCwfsTaskConnections",
    "AggregateDonutTablesCwfsTaskConfig",
    "AggregateDonutTablesCwfsTask",
]


# Note: cannot make visit a dimension because we have not yet paired visits.
class AggregateDonutTablesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),  # type: ignore
):
    donut_visit_pair_table = ct.Input(
        doc="Visit pair table",
        dimensions=("instrument",),
        storageClass="AstropyTable",
        name="donut_visit_pair_table",
    )
    donutTables = ct.Input(
        doc="Donut tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutTable",
        multiple=True,
    )
    qualityTables = ct.Input(
        doc="Donut quality tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
        multiple=True,
        deferGraphConstraint=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    aggregateDonutTable = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
        multiple=True,
    )
    dummyExposureJoiner = ct.Input(
        name="raw",
        doc=(
            "A dummy connection (datasets are never actually loaded) "
            "that adds the 'exposure' dimension to the QG generation query "
            "in order to relate 'visit' and 'group'."
        ),
        dimensions=("exposure", "detector"),
        storageClass="Exposure",
        deferLoad=True,
        multiple=True,
    )

    def __init__(self, *, config: "AggregateDonutTablesTaskConfig | None" = None) -> None:
        super().__init__(config=config)
        if config is None:
            return
        if not config.pairer.target._needsPairTable:
            del self.donut_visit_pair_table
        if config.pairer.target._needsGroupDimension:
            self.dimensions.add("group")


class AggregateDonutTablesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesTaskConnections,  # type: ignore
):
    pairer: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=ExposurePairer,
        doc="Task to pair up intra- and extra-focal exposures",
    )


class AggregateDonutTablesTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutTablesTaskConfig
    _DefaultName = "AggregateDonutTables"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config: AggregateDonutTablesTaskConfig = cast(AggregateDonutTablesTaskConfig, self.config)
        self.pairer = self.config.pairer
        self.makeSubtask("pairer")

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        camera = butlerQC.get(inputRefs.camera)

        # Only need one visitInfo per exposure.
        # And the pairer only works with uniquified visitInfos.
        visitInfoDict = dict()
        for donutTableRef in inputRefs.donutTables:
            table = butlerQC.get(donutTableRef)
            visit_id = table.meta["visit_info"]["visit_id"]
            if visit_id in visitInfoDict:
                continue
            visitInfoDict[visit_id] = convertDictToVisitInfo(table.meta["visit_info"])

        if hasattr(inputRefs, "donut_visit_pair_table"):
            pairs = self.pairer.run(visitInfoDict, butlerQC.get(inputRefs.donut_visit_pair_table))
        else:
            pairs = self.pairer.run(visitInfoDict)

        # Make dictionaries to match visits and detectors
        donutTables = {
            (ref.dataId["visit"], ref.dataId["detector"]): butlerQC.get(ref) for ref in inputRefs.donutTables
        }
        qualityTables = {
            (ref.dataId["visit"], ref.dataId["detector"]): butlerQC.get(ref)
            for ref in inputRefs.qualityTables
        }

        pairTables = self.run(camera, visitInfoDict, pairs, donutTables, qualityTables)

        # Put pairTables in butler
        for pairTableRef in outputRefs.aggregateDonutTable:
            refVisit = pairTableRef.dataId["visit"]
            if refVisit in pairTables.pairTables.keys():
                butlerQC.put(pairTables.pairTables[refVisit], pairTableRef)

    @timeMethod
    def run(
        self,
        camera: Camera,
        visitInfoDict: dict,
        pairs: list,
        donutTables: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut tables for a set of visits.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.Camera
            The camera object.
        visitInfoDict : dict
            Dictionary of visit info objects keyed by visit ID.
        pairs : list
            List of visit pairs.
        donutTables : dict
            Dictionary of donut tables keyed by (visit, detector).
        qualityTables : dict
            Dictionary of quality tables keyed by (visit, detector).

        Returns
        -------
        struct
            Struct of aggregated donut tables, keyed on extra-focal visit.
        """
        # Find common (visit, detector) extra-focal pairs
        # DonutQualityTables only saved under extra-focal ids
        extra_keys = set(donutTables) & set(qualityTables)

        # Raise error if there's no matches
        if len(extra_keys) == 0:
            raise RuntimeError("No (visit, detector) matches found between the donut and quality tables")

        pairTables = {}
        for pair in pairs:
            intraVisitInfo = visitInfoDict[pair.intra]
            extraVisitInfo = visitInfoDict[pair.extra]

            tables = []

            # Iterate over the common (visit, detector) pairs
            for visit, detector in extra_keys:
                # Check if this extra-focal visit is in this pair.
                if visit != pair.extra:
                    # This visit isn't in this pair so we will skip for now
                    continue

                # Get pixels -> field angle transform for this detector
                det = camera[detector]
                tform = det.getTransform(PIXELS, FIELD_ANGLE)

                # Load the donut catalog table, and the donut quality table
                intraDonutTable = donutTables[(pair.intra, detector)]
                extraDonutTable = donutTables[(pair.extra, detector)]
                qualityTable = qualityTables[(pair.extra, detector)]

                # Get rows of quality table for this exposure
                intraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "intra"]
                extraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "extra"]

                if (len(extraQualityTable) == 0) or (len(intraQualityTable) == 0):
                    continue

                for donutTable, qualityTable in zip(
                    [intraDonutTable, extraDonutTable],
                    [intraQualityTable, extraQualityTable],
                ):
                    # Select donuts used in Zernike estimation
                    table = donutTable[qualityTable["FINAL_SELECT"]]

                    # Add focusZ to donut table
                    table["focusZ"] = table.meta["visit_info"]["focus_z"]

                    # Add SN from quality table to the donut table
                    table["snr"] = qualityTable["SN"][qualityTable["FINAL_SELECT"]]

                    # Add field angle in CCS to the table
                    pts = tform.applyForward(
                        [Point2D(x, y) for x, y in zip(table["centroid_x"], table["centroid_y"])]
                    )
                    table["thx_CCS"] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
                    table["thy_CCS"] = [pt.x for pt in pts]
                    table["detector"] = det.getName()

                    tables.append(table)

            # Don't attempt to stack metadata
            for table in tables:
                table.meta = {}

            out = vstack(tables)

            # Add metadata for extra and intra focal exposures
            # TODO: Swap parallactic angle for pseudo parallactic angle.
            #       See SMTN-019 for details.
            out.meta["extra"] = {
                "visit": pair.extra,
                "focusZ": extraVisitInfo.focusZ,
                "parallacticAngle": extraVisitInfo.boresightParAngle.asRadians(),
                "rotAngle": extraVisitInfo.boresightRotAngle.asRadians(),
                "rotTelPos": extraVisitInfo.boresightParAngle.asRadians()
                - extraVisitInfo.boresightRotAngle.asRadians()
                - np.pi / 2,
                "ra": extraVisitInfo.boresightRaDec.getRa().asRadians(),
                "dec": extraVisitInfo.boresightRaDec.getDec().asRadians(),
                "az": extraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
                "alt": extraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
                "mjd": extraVisitInfo.date.toAstropy().mjd,
            }
            out.meta["intra"] = {
                "visit": pair.intra,
                "focusZ": intraVisitInfo.focusZ,
                "parallacticAngle": intraVisitInfo.boresightParAngle.asRadians(),
                "rotAngle": intraVisitInfo.boresightRotAngle.asRadians(),
                "rotTelPos": intraVisitInfo.boresightParAngle.asRadians()
                - intraVisitInfo.boresightRotAngle.asRadians()
                - np.pi / 2,
                "ra": intraVisitInfo.boresightRaDec.getRa().asRadians(),
                "dec": intraVisitInfo.boresightRaDec.getDec().asRadians(),
                "az": intraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
                "alt": intraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
                "mjd": intraVisitInfo.date.toAstropy().mjd,
            }

            # Carefully average angles in meta
            out.meta["average"] = {}
            for k in (
                "parallacticAngle",
                "rotAngle",
                "rotTelPos",
                "ra",
                "dec",
                "az",
                "alt",
            ):
                a1 = out.meta["extra"][k] * radians
                a2 = out.meta["intra"][k] * radians
                a2 = a2.wrapNear(a1)
                out.meta["average"][k] = ((a1 + a2) / 2).wrapCtr().asRadians()

            # Easier to average the MJDs
            out.meta["average"]["mjd"] = 0.5 * (out.meta["extra"]["mjd"] + out.meta["intra"]["mjd"])

            # Calculate coordinates in different reference frames
            q = out.meta["average"]["parallacticAngle"]
            rtp = out.meta["average"]["rotTelPos"]
            out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
            out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
            out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
            out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

            pairTables[pair.extra] = out

        return pipeBase.Struct(pairTables=pairTables)


class AggregateDonutTablesCwfsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    donutTables = ct.Input(
        doc="Donut tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutTable",
        multiple=True,
    )
    qualityTables = ct.Input(
        doc="Donut quality tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
        multiple=True,
        deferGraphConstraint=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    aggregateDonutTable = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
        multiple=False,
    )


class AggregateDonutTablesCwfsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesCwfsTaskConnections,  # type: ignore
):
    pass


class AggregateDonutTablesCwfsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutTablesCwfsTaskConfig
    _DefaultName = "AggregateDonutTablesCwfs"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        camera = butlerQC.get(inputRefs.camera)

        # Make dictionaries to match detectors
        donutTables = {(ref.dataId["detector"]): butlerQC.get(ref) for ref in inputRefs.donutTables}
        qualityTables = {(ref.dataId["detector"]): butlerQC.get(ref) for ref in inputRefs.qualityTables}

        aggTable = self.run(camera, donutTables, qualityTables)
        butlerQC.put(aggTable.aggregateDonutTable, outputRefs.aggregateDonutTable)

    @timeMethod
    def run(
        self,
        camera: Camera,
        donutTables: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut tables for a set of visits.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.Camera
            The camera object.
        donutTables : dict
            Dictionary of donut tables keyed by detector.
        qualityTables : dict
            Dictionary of quality tables keyed by detector.

        Returns
        -------
        struct
            Struct of aggregated donut tables, keyed on extra-focal visit.
        """
        tables = []
        extraDetectorIds = [191, 195, 199, 203]

        for detector in donutTables.keys():
            if detector not in extraDetectorIds:
                continue
            det_extra = camera[detector]
            det_intra = camera[detector + 1]
            # Catch a case of incomplete corner ingestion (intra-focal missing)
            if detector + 1 not in donutTables.keys():
                self.log.warning(f"{detector + 1} is  not in donutTables, skipping that corner.")
                continue
            # Load the donut catalog table, and the donut quality table
            extraDonutTable = donutTables[detector]
            intraDonutTable = donutTables[detector + 1]
            qualityTable = qualityTables[detector]

            if len(qualityTable) == 0:
                continue

            # Get rows of quality table for this exposure
            intraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "intra"]
            extraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "extra"]

            for donutTable, qualityTable, det in zip(
                [extraDonutTable, intraDonutTable],
                [extraQualityTable, intraQualityTable],
                [det_extra, det_intra],
            ):
                # Select donuts used in Zernike estimation
                table = donutTable[qualityTable["FINAL_SELECT"]]

                # Add focusZ to donut table
                offset = 1.5 if det.getId() in extraDetectorIds else -1.5
                table["focusZ"] = table.meta["visit_info"]["focus_z"] + offset * u.mm

                # Add SN from quality table to the donut table
                table["snr"] = qualityTable["SN"][qualityTable["FINAL_SELECT"]]

                # Get pixels -> field angle transform for this detector
                tform = det.getTransform(PIXELS, FIELD_ANGLE)

                # Add field angle in CCS to the table
                pts = tform.applyForward(
                    [Point2D(x, y) for x, y in zip(table["centroid_x"], table["centroid_y"])]
                )
                table["thx_CCS"] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
                table["thy_CCS"] = [pt.x for pt in pts]
                table["detector"] = det.getName()

                tables.append(table)

        # Grab visitInfo. The last one will do since all should be the same.
        visitInfo = convertDictToVisitInfo(table.meta["visit_info"])

        # Don't attempt to stack metadata
        for table in tables:
            table.meta = {}

        out = vstack(tables)

        # Add metadata for extra and intra focal exposures
        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        out.meta["visitInfo"] = {
            "visit": visitInfo.id,
            "focusZ": visitInfo.focusZ,
            "parallacticAngle": visitInfo.boresightParAngle.asRadians(),
            "rotAngle": visitInfo.boresightRotAngle.asRadians(),
            "rotTelPos": visitInfo.boresightParAngle.asRadians()
            - visitInfo.boresightRotAngle.asRadians()
            - np.pi / 2,
            "ra": visitInfo.boresightRaDec.getRa().asRadians(),
            "dec": visitInfo.boresightRaDec.getDec().asRadians(),
            "az": visitInfo.boresightAzAlt.getLongitude().asRadians(),
            "alt": visitInfo.boresightAzAlt.getLatitude().asRadians(),
            "mjd": visitInfo.date.toAstropy().mjd,
        }

        # Calculate coordinates in different reference frames
        q = out.meta["visitInfo"]["parallacticAngle"]
        rtp = out.meta["visitInfo"]["rotTelPos"]
        out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
        out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
        out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
        out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

        return pipeBase.Struct(aggregateDonutTable=out)


class AggregateDonutTablesUnpairedCwfsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesCwfsTaskConnections,  # type: ignore
):
    pass


class AggregateDonutTablesUnpairedCwfsTask(AggregateDonutTablesCwfsTask):
    ConfigClass = AggregateDonutTablesUnpairedCwfsTaskConfig  # type: ignore[assignment]
    _DefaultName = "AggregateDonutTablesUnpairedCwfs"

    @timeMethod
    def run(
        self,
        camera: Camera,
        donutTables: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut tables for a set of visits.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.Camera
            The camera object.
        donutTables : dict
            Dictionary of donut tables keyed by detector.
        qualityTables : dict
            Dictionary of quality tables keyed by detector.

        Returns
        -------
        struct
            Struct of aggregated donut tables, keyed on extra-focal visit.
        """
        tables = []
        extraDetectorIds = [191, 195, 199, 203]

        for detector in donutTables.keys():
            if detector not in qualityTables.keys():
                continue

            det = camera[detector]

            # Load the donut catalog table, and the donut quality table
            donutTable = donutTables[detector]
            qualityTable = qualityTables[detector]

            if len(qualityTable) == 0:
                continue

            table = donutTable[qualityTable["FINAL_SELECT"]]

            # Add focusZ to donut table
            offset = 1.5 if det.getId() in extraDetectorIds else -1.5
            table["focusZ"] = table.meta["visit_info"]["focus_z"] + offset * u.mm

            # Add SN from quality table to the donut table
            table["snr"] = qualityTable["SN"][qualityTable["FINAL_SELECT"]]

            # Get pixels -> field angle transform for this detector
            tform = det.getTransform(PIXELS, FIELD_ANGLE)

            # Add field angle in CCS to the table
            pts = tform.applyForward(
                [Point2D(x, y) for x, y in zip(table["centroid_x"], table["centroid_y"])]
            )
            table["thx_CCS"] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
            table["thy_CCS"] = [pt.x for pt in pts]
            table["detector"] = det.getName()

            tables.append(table)

        # Grab visitInfo. The last one will do since all should be the same.
        visitInfo = convertDictToVisitInfo(table.meta["visit_info"])

        # Don't attempt to stack metadata
        for table in tables:
            table.meta = {}

        out = vstack(tables)

        # Add metadata for extra and intra focal exposures
        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        out.meta["visitInfo"] = {
            "visit": visitInfo.id,
            "focusZ": visitInfo.focusZ,
            "parallacticAngle": visitInfo.boresightParAngle.asRadians(),
            "rotAngle": visitInfo.boresightRotAngle.asRadians(),
            "rotTelPos": visitInfo.boresightParAngle.asRadians()
            - visitInfo.boresightRotAngle.asRadians()
            - np.pi / 2,
            "ra": visitInfo.boresightRaDec.getRa().asRadians(),
            "dec": visitInfo.boresightRaDec.getDec().asRadians(),
            "az": visitInfo.boresightAzAlt.getLongitude().asRadians(),
            "alt": visitInfo.boresightAzAlt.getLatitude().asRadians(),
            "mjd": visitInfo.date.toAstropy().mjd,
        }

        # Calculate coordinates in different reference frames
        q = out.meta["visitInfo"]["parallacticAngle"]
        rtp = out.meta["visitInfo"]["rotTelPos"]
        out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
        out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
        out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
        out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

        return pipeBase.Struct(aggregateDonutTable=out)
