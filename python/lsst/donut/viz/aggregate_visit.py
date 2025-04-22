import typing

import galsim
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy import units as u
from astropy.table import QTable, Table, vstack
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.geom import Point2D, radians
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.generateDonutCatalogUtils import convertDictToVisitInfo
from lsst.ts.wep.task.pairTask import ExposurePairer
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateZernikeTablesTaskConnections",
    "AggregateZernikeTablesTaskConfig",
    "AggregateZernikeTablesTask",
    "AggregateDonutTablesTaskConnections",
    "AggregateDonutTablesTaskConfig",
    "AggregateDonutTablesTask",
    "AggregateDonutTablesCwfsTaskConnections",
    "AggregateDonutTablesCwfsTaskConfig",
    "AggregateDonutTablesCwfsTask",
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
    "AggregateAOSVisitTableCwfsTask",
    "AggregateDonutStampsTaskConnections",
    "AggregateDonutStampsTaskConfig",
    "AggregateDonutStampsTask",
]


class AggregateZernikeTablesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),
):
    zernikeTable = ct.Input(
        doc="Zernike Coefficients from all donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="zernikes",
        multiple=True,
        deferGraphConstraint=True,
    )
    aggregateZernikesRaw = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
    )
    aggregateZernikesAvg = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
    )


class AggregateZernikeTablesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateZernikeTablesTaskConnections,
):
    pass


class AggregateZernikeTablesTask(pipeBase.PipelineTask):
    ConfigClass = AggregateZernikeTablesTaskConfig
    _DefaultName = "AggregateZernikeTables"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):

        zernike_tables = butlerQC.get(inputRefs.zernikeTable)
        out_raw, out_avg = self.run(zernike_tables)

        # Find the right output references
        butlerQC.put(out_raw, outputRefs.aggregateZernikesRaw)
        butlerQC.put(out_avg, outputRefs.aggregateZernikesAvg)

    @timeMethod
    def run(self, zernike_tables: typing.List[QTable]) -> tuple[Table, Table]:

        raw_tables = []
        avg_tables = []
        table_meta = None

        for zernike_table in zernike_tables:
            if len(zernike_table) == 0:
                continue
            raw_table = Table()
            zernikes_merged = []
            noll_indices = []
            for col_name in zernike_table.colnames:
                # Grab zernike output columns
                if col_name.startswith("Z"):
                    zernikes_merged.append(zernike_table[col_name].to(u.um).value)
                    noll_indices.append(int(col_name[1:]))
            zernikes_merged = np.array(zernikes_merged).T
            noll_indices = np.array(noll_indices)
            raw_table["zk_CCS"] = np.atleast_2d(zernikes_merged[1:])
            raw_table["detector"] = zernike_table.meta["extra"]["det_name"]
            raw_tables.append(raw_table)
            avg_table = Table()
            avg_table["zk_CCS"] = np.atleast_2d(zernikes_merged[0])
            avg_table["detector"] = zernike_table.meta["extra"]["det_name"]
            avg_tables.append(avg_table)
            # just get any one, they're all the same
            if table_meta is None:
                table_meta = zernike_table.meta
        out_raw = vstack(raw_tables)
        out_avg = vstack(avg_tables)

        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        meta = {}
        meta["visit"] = table_meta["extra"]["visit"]
        meta["parallacticAngle"] = table_meta["extra"]["boresight_par_angle_rad"]
        meta["rotAngle"] = table_meta["extra"]["boresight_rot_angle_rad"]
        rtp = (
            meta["parallacticAngle"] * radians
            - meta["rotAngle"] * radians
            - (np.pi / 2 * radians)
        ).asRadians()
        meta["rotTelPos"] = rtp
        meta["ra"] = table_meta["extra"]["boresight_ra_rad"]
        meta["dec"] = table_meta["extra"]["boresight_dec_rad"]
        meta["az"] = table_meta["extra"]["boresight_az_rad"]
        meta["alt"] = table_meta["extra"]["boresight_alt_rad"]
        # Average mjds
        meta["mjd"] = 0.5 * (table_meta["extra"]["mjd"] + table_meta["intra"]["mjd"])
        meta["nollIndices"] = noll_indices

        q = meta["parallacticAngle"]
        rtp = meta["rotTelPos"]

        jmin = np.min(noll_indices)
        jmax = np.max(noll_indices)
        rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:, 4:]
        rot_NW = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:, 4:]
        for cat in (out_raw, out_avg):
            cat.meta = meta
            full_zk_ccs = np.zeros((len(cat), jmax - jmin + 1))
            full_zk_ccs[:, noll_indices - 4] = cat["zk_CCS"]
            cat["zk_OCS"] = full_zk_ccs @ rot_OCS
            cat["zk_NW"] = full_zk_ccs @ rot_NW
            cat["zk_OCS"] = cat["zk_OCS"][:, noll_indices - 4]
            cat["zk_NW"] = cat["zk_NW"][:, noll_indices - 4]

        return out_raw, out_avg


# Note: cannot make visit a dimension because we have not yet paired visits.
class AggregateDonutTablesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
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

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.pairer.target._needsPairTable:
            del self.donut_visit_pair_table
        if config.pairer.target._needsGroupDimension:
            self.dimensions.add("group")


class AggregateDonutTablesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesTaskConnections,
):
    pairer = pexConfig.ConfigurableField(
        target=ExposurePairer,
        doc="Task to pair up intra- and extra-focal exposures",
    )


class AggregateDonutTablesTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutTablesTaskConfig
    _DefaultName = "AggregateDonutTables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("pairer")

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
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
            pairs = self.pairer.run(
                visitInfoDict, butlerQC.get(inputRefs.donut_visit_pair_table)
            )
        else:
            pairs = self.pairer.run(visitInfoDict)

        # Make dictionaries to match visits and detectors
        donutTables = {
            (ref.dataId["visit"], ref.dataId["detector"]): butlerQC.get(ref)
            for ref in inputRefs.donutTables
        }
        qualityTables = {
            (ref.dataId["visit"], ref.dataId["detector"]): butlerQC.get(ref)
            for ref in inputRefs.qualityTables
        }

        pairTables = self.run(camera, visitInfoDict, pairs, donutTables, qualityTables)

        # Put pairTables in butler
        for pairTableRef in outputRefs.aggregateDonutTable:
            refVisit = pairTableRef.dataId["visit"]
            if refVisit in pairTables.keys():
                butlerQC.put(pairTables[refVisit], pairTableRef)

    @timeMethod
    def run(
        self,
        camera,
        visitInfoDict: dict,
        pairs: list,
        donutTables: dict,
        qualityTables: dict,
    ) -> typing.List[QTable]:
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
        dict of astropy.table.QTable
            Dict of aggregated donut tables, keyed on extra-focal visit.
        """
        # Find common (visit, detector) extra-focal pairs
        # DonutQualityTables only saved under extra-focal ids
        extra_keys = set(donutTables) & set(qualityTables)

        # Raise error if there's no matches
        if len(extra_keys) == 0:
            raise RuntimeError(
                "No (visit, detector) matches found between "
                "the donut and quality tables"
            )

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
                intraQualityTable = qualityTable[
                    qualityTable["DEFOCAL_TYPE"] == "intra"
                ]
                extraQualityTable = qualityTable[
                    qualityTable["DEFOCAL_TYPE"] == "extra"
                ]

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

                    # Add field angle in CCS to the table
                    pts = tform.applyForward(
                        [
                            Point2D(x, y)
                            for x, y in zip(table["centroid_x"], table["centroid_y"])
                        ]
                    )
                    table["thx_CCS"] = [
                        pt.y for pt in pts
                    ]  # Transpose from DVCS to CCS
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
            out.meta["average"]["mjd"] = 0.5 * (
                out.meta["extra"]["mjd"] + out.meta["intra"]["mjd"]
            )

            # Calculate coordinates in different reference frames
            q = out.meta["average"]["parallacticAngle"]
            rtp = out.meta["average"]["rotTelPos"]
            out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
            out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
            out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
            out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

            pairTables[pair.extra] = out

        return pairTables


class AggregateDonutTablesCwfsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),
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
    pipelineConnections=AggregateDonutTablesCwfsTaskConnections,
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
    ):
        camera = butlerQC.get(inputRefs.camera)

        # Make dictionaries to match detectors
        donutTables = {
            (ref.dataId["detector"]): butlerQC.get(ref) for ref in inputRefs.donutTables
        }
        qualityTables = {
            (ref.dataId["detector"]): butlerQC.get(ref)
            for ref in inputRefs.qualityTables
        }

        aggTable = self.run(camera, donutTables, qualityTables)
        butlerQC.put(aggTable, outputRefs.aggregateDonutTable)

    @timeMethod
    def run(
        self,
        camera,
        donutTables: dict,
        qualityTables: dict,
    ) -> typing.List[QTable]:
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
        dict of astropy.table.QTable
            Dict of aggregated donut tables, keyed on extra-focal visit.
        """
        tables = []
        extraDetectorIds = [191, 195, 199, 203]

        for detector in donutTables.keys():
            if detector not in extraDetectorIds:
                continue

            det_extra = camera[detector]
            det_intra = camera[detector + 1]

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

                # Get pixels -> field angle transform for this detector
                tform = det.getTransform(PIXELS, FIELD_ANGLE)

                # Add field angle in CCS to the table
                pts = tform.applyForward(
                    [
                        Point2D(x, y)
                        for x, y in zip(table["centroid_x"], table["centroid_y"])
                    ]
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

        return out


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=(
        "visit",
        "instrument",
    ),
):
    aggregateDonutTable = ct.Input(
        doc="Visit-level table of donuts",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
        deferGraphConstraint=True,
    )
    aggregateZernikesRaw = ct.Input(
        doc="Visit-level table of raw Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
        deferGraphConstraint=True,
    )
    aggregateZernikesAvg = ct.Input(
        doc="Visit-level table of average Zernikes by detector",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
        deferGraphConstraint=True,
    )
    aggregateAOSRaw = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    aggregateAOSAvg = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
    )


class AggregateAOSVisitTableTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateAOSVisitTableTaskConnections,
):
    pass


class AggregateAOSVisitTableTask(pipeBase.PipelineTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTable"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        adt = butlerQC.get(inputRefs.aggregateDonutTable)
        azr = butlerQC.get(inputRefs.aggregateZernikesRaw)
        aza = butlerQC.get(inputRefs.aggregateZernikesAvg)

        avg_table, raw_table = self.run(adt, azr, aza)

        butlerQC.put(avg_table, outputRefs.aggregateAOSAvg)
        butlerQC.put(raw_table, outputRefs.aggregateAOSRaw)

    @timeMethod
    def run(
        self, adt: typing.List[Table], azr: typing.List[Table], aza: typing.List[Table]
    ) -> tuple[Table, Table]:
        dets = np.unique(adt["detector"])
        avg_table = aza.copy()
        avg_keys = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "thx_CCS",
            "thy_CCS",
            "thx_OCS",
            "thy_OCS",
            "th_N",
            "th_W",
        ]
        for k in avg_keys:
            avg_table[k] = np.nan  # Allocate

        for det in dets:
            w = avg_table["detector"] == det
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][adt["detector"] == det])

        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            fzmin = adt[wadt]["focusZ"].min()
            fzmax = adt[wadt]["focusZ"].max()
            if fzmin == fzmax:  # single-sided Zernike estimates
                for k in avg_keys:
                    raw_table[k][w] = adt[k][wadt]
            else:  # double-sided Zernike estimates
                wintra = adt[wadt]["focusZ"] == fzmin
                wextra = adt[wadt]["focusZ"] == fzmax
                for k in avg_keys:
                    # If one table has more rows than the other,
                    # trim the longer one
                    if wintra.sum() > wextra.sum():
                        wintra[wintra] = [True] * wextra.sum() + [False] * (
                            wintra.sum() - wextra.sum()
                        )
                    elif wextra.sum() > wintra.sum():
                        wextra[wextra] = [True] * wintra.sum() + [False] * (
                            wextra.sum() - wintra.sum()
                        )
                    # ought to be the same length now
                    raw_table[k][w] = 0.5 * (
                        adt[k][wadt][wintra] + adt[k][wadt][wextra]
                    )
                    if k + "_intra" not in raw_table.colnames:
                        raw_table[k + "_intra"] = np.nan
                        raw_table[k + "_extra"] = np.nan
                    raw_table[k + "_intra"][w] = adt[k][wadt][wintra]
                    raw_table[k + "_extra"][w] = adt[k][wadt][wextra]

        return avg_table, raw_table


class AggregateAOSVisitTableCwfsTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableCwfs"

    @timeMethod
    def run(
        self, adt: typing.List[Table], azr: typing.List[Table], aza: typing.List[Table]
    ) -> tuple[Table, Table]:
        extraDetectorNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        intraDetectorNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]
        # Only take extra focal detector names
        avg_table = aza.copy()
        avg_keys = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "thx_CCS",
            "thy_CCS",
            "thx_OCS",
            "thy_OCS",
            "th_N",
            "th_W",
        ]
        for k in avg_keys:
            avg_table[k] = np.nan  # Allocate

        # Process average table
        for det_extra, det_intra in zip(extraDetectorNames, intraDetectorNames):
            w = avg_table["detector"] == det_extra
            wextra = adt["detector"] == det_extra
            wintra = adt["detector"] == det_intra
            # Combine extra and intra detector masks
            wadt = np.logical_or(wextra, wintra)
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][wadt])

        # Process raw table
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det_extra, det_intra in zip(extraDetectorNames, intraDetectorNames):
            w = raw_table["detector"] == det_extra
            wextra = adt["detector"] == det_extra
            wintra = adt["detector"] == det_intra
            # Combine extra and intra detector masks
            wadt = np.logical_or(wextra, wintra)
            # Check if there are any matching rows
            if not np.any(wadt):
                continue

            for k in avg_keys:
                # If one table has more rows than the other,
                # trim the longer one
                if wintra.sum() > wextra.sum():
                    wintra[wintra] = [True] * wextra.sum() + [False] * (
                        wintra.sum() - wextra.sum()
                    )
                elif wextra.sum() > wintra.sum():
                    wextra[wextra] = [True] * wintra.sum() + [False] * (
                        wextra.sum() - wintra.sum()
                    )
                # ought to be the same length now
                raw_table[k][w] = 0.5 * (adt[k][wintra] + adt[k][wextra])
                if k + "_intra" not in raw_table.colnames:
                    raw_table[k + "_intra"] = np.nan
                    raw_table[k + "_extra"] = np.nan
                raw_table[k + "_intra"][w] = adt[k][wintra]
                raw_table[k + "_extra"][w] = adt[k][wextra]

        return avg_table, raw_table


class AggregateDonutStampsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),
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
    pipelineConnections=AggregateDonutStampsTaskConnections,
):
    maxDonutsPerDetector = pexConfig.Field[int](
        doc="Maximum number of donuts to use per detector",
        default=1,
    )

    def validate(self):
        if self.maxDonutsPerDetector < 1:
            raise pexConfig.FieldValidationError(
                "maxDonutsPerDetector must be at least 1"
            )


class AggregateDonutStampsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutStampsTaskConfig
    _DefaultName = "AggregateDonutStamps"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:

        intraStampsOut, extraStampsOut = self.run(
            butlerQC.get(inputRefs.donutStampsIntra),
            butlerQC.get(inputRefs.donutStampsExtra),
            butlerQC.get(inputRefs.qualityTables),
        )

        butlerQC.put(
            intraStampsOut,
            outputRefs.donutStampsIntraVisit,
        )

        butlerQC.put(
            extraStampsOut,
            outputRefs.donutStampsExtraVisit,
        )

    @timeMethod
    def run(
        self,
        intraStamps: typing.List,
        extraStamps: typing.List,
        qualityTables: typing.List,
    ) -> tuple[typing.List, typing.List, dafBase.PropertyList, dafBase.PropertyList]:
        intraStampsList = []
        extraStampsList = []
        intraMetaAll = None
        extraMetaAll = None
        for intra, extra, quality in zip(intraStamps, extraStamps, qualityTables):
            # Skip if quality table is empty.
            if len(quality) == 0:
                continue

            # Load the quality table and determine which donuts were selected
            intraQualitySelect = quality[quality["DEFOCAL_TYPE"] == "intra"][
                "FINAL_SELECT"
            ]
            extraQualitySelect = quality[quality["DEFOCAL_TYPE"] == "extra"][
                "FINAL_SELECT"
            ]

            # Extract metadata dictionaries
            intraMeta = intra.metadata.toDict().copy()
            extraMeta = extra.metadata.toDict().copy()
            # The metadata for donutStamps is a PropertyList.
            # If there is only a single donutStamp then the PropertyList
            # converts any lists of length 1 into scalars.
            # Therefore, we check both just in case one metadata value
            # is a list and one is a scalar for the same key.
            listKeys_intra = [
                key
                for key, val in intraMeta.items()
                if (isinstance(val, list) and key != "COMMENT")
            ]
            listKeys_extra = [
                key
                for key, val in extraMeta.items()
                if (isinstance(val, list) and key != "COMMENT")
            ]
            listKeys = set(listKeys_extra).union(set(listKeys_intra))
            singleKeys = set(intraMeta) - set(listKeys)

            # Select donuts used in Zernike estimation
            intraStampsSelect = DonutStamps(
                [intra[i] for i in range(len(intra)) if intraQualitySelect[i]]
            )
            extraStampsSelect = DonutStamps(
                [extra[i] for i in range(len(extra)) if extraQualitySelect[i]]
            )

            # Copy over metadata
            if intraMetaAll is None:
                intraMetaAll = dict()
                extraMetaAll = dict()
                for key in singleKeys:
                    intraMetaAll[key] = intraMeta[key]
                    extraMetaAll[key] = extraMeta[key]
                for key in listKeys:
                    intraMetaAll[key] = list()
                    extraMetaAll[key] = list()

            for key in listKeys:
                intraMetaAll[key].append(
                    np.array(intra.metadata.getArray(key))[intraQualitySelect].tolist()[
                        : self.config.maxDonutsPerDetector
                    ]
                )
                extraMetaAll[key].append(
                    np.array(extra.metadata.getArray(key))[extraQualitySelect].tolist()[
                        : self.config.maxDonutsPerDetector
                    ]
                )

            # Append the requested number of donuts
            intraStampsList.append(
                intraStampsSelect[: self.config.maxDonutsPerDetector]
            )
            extraStampsList.append(
                extraStampsSelect[: self.config.maxDonutsPerDetector]
            )

        for key in listKeys:
            intraMetaAll[key] = [
                obj for detList in intraMetaAll[key] for obj in detList
            ]
            extraMetaAll[key] = [
                obj for detList in extraMetaAll[key] for obj in detList
            ]
        intraStampsSelect._metadata = intraStampsSelect.metadata.from_mapping(
            intraMetaAll
        )
        extraStampsSelect._metadata = extraStampsSelect.metadata.from_mapping(
            extraMetaAll
        )

        intraStampsListRavel = [
            stamp for stampList in intraStampsList for stamp in stampList
        ]
        extraStampsListRavel = [
            stamp for stampList in extraStampsList for stamp in stampList
        ]

        intraStampsRavel = DonutStamps(
            intraStampsListRavel, metadata=intraStampsSelect._metadata
        )
        extraStampsRavel = DonutStamps(
            extraStampsListRavel, metadata=extraStampsSelect._metadata
        )

        return intraStampsRavel, extraStampsRavel
