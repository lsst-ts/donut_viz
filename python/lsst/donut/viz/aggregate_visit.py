import galsim
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
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
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
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
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
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
        camera = butlerQC.get(inputRefs.camera)

        raw_tables = []
        avg_tables = []
        for zernikesRef in inputRefs.zernikeTable:
            zernike_table = butlerQC.get(zernikesRef)
            raw_table = Table()
            zernikes_merged = []
            for col_name in zernike_table.colnames:
                # Grab zernike output columns
                if col_name.startswith("Z"):
                    zernikes_merged.append(zernike_table[col_name].to(u.um).value)
            zernikes_merged = np.array(zernikes_merged).T
            raw_table["zk_CCS"] = np.atleast_2d(zernikes_merged[1:])
            raw_table["detector"] = camera[zernikesRef.dataId["detector"]].getName()
            raw_tables.append(raw_table)
            avg_table = Table()
            avg_table["zk_CCS"] = np.atleast_2d(zernikes_merged[0])
            avg_table["detector"] = camera[zernikesRef.dataId["detector"]].getName()
            avg_tables.append(avg_table)
        out_raw = vstack(raw_tables)
        out_avg = vstack(avg_tables)

        # just get the last one, they're all the same
        table_meta = zernike_table.meta

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

        q = meta["parallacticAngle"]
        rtp = meta["rotTelPos"]

        jmax = out_raw["zk_CCS"].shape[1] + 3
        rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:, 4:]
        rot_NW = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:, 4:]
        for cat in (out_raw, out_avg):
            cat.meta = meta
            cat["zk_OCS"] = cat["zk_CCS"] @ rot_OCS
            cat["zk_NW"] = cat["zk_CCS"] @ rot_NW

        # Find the right output references
        butlerQC.put(out_raw, outputRefs.aggregateZernikesRaw)
        butlerQC.put(out_avg, outputRefs.aggregateZernikesAvg)


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
        visitInfoDict = {}
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

        for pair in pairs:
            intraVisitInfo = visitInfoDict[pair.intra]
            extraVisitInfo = visitInfoDict[pair.extra]

            tables = []
            # Find all detectors...
            for donutTableRef in inputRefs.donutTables:
                dataId = donutTableRef.dataId
                if dataId["visit"] not in (pair.intra, pair.extra):
                    continue

                det = camera[dataId["detector"]]
                tform = det.getTransform(PIXELS, FIELD_ANGLE)

                table = butlerQC.get(donutTableRef)
                table["focusZ"] = (
                    intraVisitInfo.focusZ
                    if dataId["visit"] == pair.intra
                    else extraVisitInfo.focusZ
                )
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

            # Don't attempt to stack metadata
            for table in tables:
                table.meta = {}

            out = vstack(tables)

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

            q = out.meta["average"]["parallacticAngle"]
            rtp = out.meta["average"]["rotTelPos"]
            out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
            out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
            out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
            out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

            # Find the right output references
            for outRef in outputRefs.aggregateDonutTable:
                if outRef.dataId["visit"] == pair.extra:
                    butlerQC.put(out, outRef)
                    break
            else:
                raise ValueError(
                    f"Expected to find an output reference with visit {pair.extra}"
                )


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=(
        "visit",
        "instrument",
    ),
):
    aggregateDonutTable = ct.Input(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
    )
    aggregateZernikesRaw = ct.Input(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
    )
    aggregateZernikesAvg = ct.Input(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
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
        adc = butlerQC.get(inputRefs.aggregateDonutTable)
        azr = butlerQC.get(inputRefs.aggregateZernikesRaw)
        aza = butlerQC.get(inputRefs.aggregateZernikesAvg)

        dets = np.unique(adc["detector"])
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
                avg_table[k][w] = np.mean(adc[k][adc["detector"] == det])

        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det in dets:
            w = raw_table["detector"] == det
            wadc = adc["detector"] == det
            fzmin = adc[wadc]["focusZ"].min()
            fzmax = adc[wadc]["focusZ"].max()
            if fzmin == fzmax:  # single-sided Zernike estimates
                for k in avg_keys:
                    raw_table[k][w] = adc[k][wadc]
            else:  # double-sided Zernike estimates
                wintra = adc[wadc]["focusZ"] == fzmin
                wextra = adc[wadc]["focusZ"] == fzmax
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
                        adc[k][wadc][wintra] + adc[k][wadc][wextra]
                    )
                    if k + "_intra" not in raw_table.colnames:
                        raw_table[k + "_intra"] = np.nan
                        raw_table[k + "_extra"] = np.nan
                    raw_table[k + "_intra"][w] = adc[k][wadc][wintra]
                    raw_table[k + "_extra"][w] = adc[k][wadc][wextra]

        butlerQC.put(avg_table, outputRefs.aggregateAOSAvg)
        butlerQC.put(raw_table, outputRefs.aggregateAOSRaw)


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
    )
    donutStampsExtra = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
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
        intraStampsList = []
        extraStampsList = []
        for intraRef, extraRef in zip(
            inputRefs.donutStampsIntra, inputRefs.donutStampsExtra
        ):
            intra = butlerQC.get(intraRef)
            extra = butlerQC.get(extraRef)
            intraStampsList.append(intra[: self.config.maxDonutsPerDetector])
            extraStampsList.append(extra[: self.config.maxDonutsPerDetector])
        intraStampsListRavel = np.ravel(intraStampsList)
        extraStampsListRavel = np.ravel(extraStampsList)

        butlerQC.put(
            DonutStamps(intraStampsListRavel, metadata=intra.metadata),
            outputRefs.donutStampsIntraVisit,
        )

        butlerQC.put(
            DonutStamps(extraStampsListRavel, metadata=extra.metadata),
            outputRefs.donutStampsExtraVisit,
        )
