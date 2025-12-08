import typing
from copy import copy
from typing import Any, cast

import galsim
import numpy as np
from astropy import units as u
from astropy.table import QTable, Table, vstack

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.geom import Point2D, radians
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.pairTask import ExposurePairer
from lsst.ts.wep.utils import convertDictToVisitInfo
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateZernikeTablesTaskConnections",
    "AggregateZernikeTablesTaskConfig",
    "AggregateZernikeTablesTask",
    "AggregateDonutTablesTaskConnections",
    "AggregateDonutTablesTaskConfig",
    "AggregateDonutTablesTask",
    "AggregateDonutTablesUnpairedCwfsTaskConfig",
    "AggregateDonutTablesUnpairedCwfsTask",
    "AggregateDonutTablesCwfsTaskConnections",
    "AggregateDonutTablesCwfsTaskConfig",
    "AggregateDonutTablesCwfsTask",
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
    "AggregateAOSVisitTableCwfsTask",
    "AggregateAOSVisitTableUnpairedCwfsTask",
    "AggregateDonutStampsTaskConnections",
    "AggregateDonutStampsTaskConfig",
    "AggregateDonutStampsTask",
    "AggregateDonutStampsUnpairedTaskConnections",
    "AggregateDonutStampsUnpairedTaskConfig",
    "AggregateDonutStampsUnpairedTask",
]


class AggregateZernikeTablesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
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
    pipelineConnections=AggregateZernikeTablesTaskConnections,  # type: ignore
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
    ) -> None:
        zernike_tables = butlerQC.get(inputRefs.zernikeTable)
        aggregate_tables = self.run(zernike_tables)

        # Find the right output references
        butlerQC.put(aggregate_tables.raw, outputRefs.aggregateZernikesRaw)
        butlerQC.put(aggregate_tables.avg, outputRefs.aggregateZernikesAvg)

    @timeMethod
    def run(self, zernike_tables: typing.List[QTable]) -> pipeBase.Struct:
        """Aggregate Zernike tables for a visit.

        Parameters
        ----------
        zernike_tables : list of astropy.table.QTable
            List of Zernike tables for all detectors in a visit.

        Returns
        -------
        struct
            Struct of aggregated zernike tables with 'raw' and 'avg' entries.
        """
        raw_tables = []
        avg_tables = []
        table_meta: dict | None = None
        estimator_meta: dict = dict()

        for zernike_table in zernike_tables:
            if len(zernike_table) == 0:
                continue

            # Get the populated metadata dictionary
            intra_meta = zernike_table.meta["intra"]
            extra_meta = zernike_table.meta["extra"]

            # Check if both are empty
            if not intra_meta and not extra_meta:
                self.log.warning("Both intra and extra metadata are empty dictionaries. Skipping this table.")
                continue

            # Select metadata and determine if unpaired
            det_meta = extra_meta or intra_meta
            unpaired_det_type = not (intra_meta and extra_meta)

            # Create tables for raw and average zernikes
            raw_table = Table()
            avg_table = Table()

            # Save OPD coefficients
            zernikes_merged = []
            for col_name in zernike_table.meta["opd_columns"]:
                zernikes_merged.append(zernike_table[col_name].to(u.um).value)
            zernikes_merged = np.array(zernikes_merged).T
            raw_table["zk_CCS"] = np.atleast_2d(zernikes_merged[1:])
            avg_table["zk_CCS"] = np.atleast_2d(zernikes_merged[0])

            # Save intrinsic coefficients
            intrinsics_merged = []
            for col_name in zernike_table.meta["intrinsic_columns"]:
                intrinsics_merged.append(zernike_table[col_name].to(u.um).value)
            intrinsics_merged = np.array(intrinsics_merged).T
            raw_table["zk_intrinsic_CCS"] = np.atleast_2d(intrinsics_merged[1:])
            avg_table["zk_intrinsic_CCS"] = np.atleast_2d(intrinsics_merged[0])

            # Save wavefront deviation coefficients
            deviations_merged = []
            for col_name in zernike_table.meta["deviation_columns"]:
                deviations_merged.append(zernike_table[col_name].to(u.um).value)
            deviations_merged = np.array(deviations_merged).T
            raw_table["zk_deviation_CCS"] = np.atleast_2d(deviations_merged[1:])
            avg_table["zk_deviation_CCS"] = np.atleast_2d(deviations_merged[0])

            # Add some more metadata
            raw_table["used"] = zernike_table["used"][1:]
            raw_table["detector"] = det_meta["det_name"]
            avg_table["detector"] = det_meta["det_name"]

            raw_tables.append(raw_table)
            avg_tables.append(avg_table)

            # Save estimator metadata
            # (just get any one, they're all the same)
            if table_meta is None:
                table_meta = zernike_table.meta
            if "estimatorInfo" in zernike_table.meta.keys():
                for key, val in zernike_table.meta["estimatorInfo"].items():
                    if key not in estimator_meta:
                        estimator_meta[key] = []
                    estimator_meta[key] += val

        # Aggregate all tables
        out_raw = vstack(raw_tables)
        out_avg = vstack(avg_tables)

        # Metadata about pointing, rotation, etc.
        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        meta = {}
        meta["visit"] = det_meta["visit"]
        meta["parallacticAngle"] = det_meta["boresight_par_angle_rad"]
        meta["rotAngle"] = det_meta["boresight_rot_angle_rad"]
        rtp = (
            meta["parallacticAngle"] * radians - meta["rotAngle"] * radians - (np.pi / 2 * radians)
        ).asRadians()
        meta["rotTelPos"] = rtp
        meta["ra"] = det_meta["boresight_ra_rad"]
        meta["dec"] = det_meta["boresight_dec_rad"]
        meta["az"] = det_meta["boresight_az_rad"]
        meta["alt"] = det_meta["boresight_alt_rad"]
        meta["band"] = det_meta["band"]

        # Average mjds
        if table_meta is None:
            raise RuntimeError("No metadata found in input zernike tables.")
        if unpaired_det_type is False:
            meta["mjd"] = 0.5 * (table_meta["extra"]["mjd"] + table_meta["intra"]["mjd"])
        else:
            meta["mjd"] = det_meta["mjd"]

        # Noll indices corresponding to the Zernike coefficients
        noll_indices = np.array(zernike_table.meta["noll_indices"])
        meta["nollIndices"] = noll_indices

        # Transform Zernike coefficients to OCS and NW frames
        q = meta["parallacticAngle"]
        rtp = meta["rotTelPos"]

        jmin = np.min(noll_indices)
        jmax = np.max(noll_indices)
        rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:, 4:]
        rot_NW = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:, 4:]
        for cat in (out_raw, out_avg):
            cat.meta = copy(meta)

            # OPD coefficients
            full_zk_ccs = np.zeros((len(cat), jmax - jmin + 1))
            full_zk_ccs[:, noll_indices - 4] = cat["zk_CCS"]
            cat["zk_OCS"] = full_zk_ccs @ rot_OCS
            cat["zk_NW"] = full_zk_ccs @ rot_NW
            cat["zk_OCS"] = cat["zk_OCS"][:, noll_indices - 4]
            cat["zk_NW"] = cat["zk_NW"][:, noll_indices - 4]

            # Intrinsic coefficients
            full_zk_intrinsic_ccs = np.zeros((len(cat), jmax - jmin + 1))
            full_zk_intrinsic_ccs[:, noll_indices - 4] = cat["zk_intrinsic_CCS"]
            cat["zk_intrinsic_OCS"] = full_zk_intrinsic_ccs @ rot_OCS
            cat["zk_intrinsic_NW"] = full_zk_intrinsic_ccs @ rot_NW
            cat["zk_intrinsic_OCS"] = cat["zk_intrinsic_OCS"][:, noll_indices - 4]
            cat["zk_intrinsic_NW"] = cat["zk_intrinsic_NW"][:, noll_indices - 4]

            # Deviation coefficients
            full_zk_deviation_ccs = np.zeros((len(cat), jmax - jmin + 1))
            full_zk_deviation_ccs[:, noll_indices - 4] = cat["zk_deviation_CCS"]
            cat["zk_deviation_OCS"] = full_zk_deviation_ccs @ rot_OCS
            cat["zk_deviation_NW"] = full_zk_deviation_ccs @ rot_NW
            cat["zk_deviation_OCS"] = cat["zk_deviation_OCS"][:, noll_indices - 4]
            cat["zk_deviation_NW"] = cat["zk_deviation_NW"][:, noll_indices - 4]

        # Add wavefront estimation metadata for individual donuts
        # to the raw table.
        out_raw.meta["estimatorInfo"] = estimator_meta
        # Add average danish fwhm values into metadata of average table.
        if "fwhm" in out_raw.meta["estimatorInfo"].keys():
            out_avg.meta["estimatorInfo"] = dict()
            out_avg.meta["estimatorInfo"]["fwhm"] = np.nanmedian(out_raw.meta["estimatorInfo"]["fwhm"])

        return pipeBase.Struct(raw=out_raw, avg=out_avg)


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


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=(
        "visit",
        "instrument",
    ),  # type: ignore
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
    pipelineConnections=AggregateAOSVisitTableTaskConnections,  # type: ignore
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

        tables = self.run(adt, azr, aza)

        butlerQC.put(tables.avg, outputRefs.aggregateAOSAvg)
        butlerQC.put(tables.raw, outputRefs.aggregateAOSRaw)

    @timeMethod
    def run(self, adt: Table, azr: Table, aza: Table) -> pipeBase.Struct:
        """
        Create overall summary tables for the visit.

        Parameters
        ----------
        adt : `astropy.table.Table`
            Aggregated donut table.
        azr : `astropy.table.Table`
            Aggregated raw Zernike table.
        aza : `astropy.table.Table`
            Aggregated average Zernike table.

        Returns
        -------
        struct : `pipeBase.Struct`
            Struct with `avg` and `raw` tables:
            avg_table : `astropy.table.Table`
                Table with average donut and Zernike values by detector.
            raw_table : `astropy.table.Table`
                Table with donut and Zernike values from every
                source that went into the averages.
        """
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
        # Use focusZ to learn about contents of the table
        single_sided = False
        adt["focusZ"] = np.round(adt["focusZ"], decimals=4)
        visit_fzmin = adt["focusZ"].min()
        visit_fzmax = adt["focusZ"].max()
        # If entire donut table has a single focusZ value, then
        # we can assume single-sided Zernike estimates
        if visit_fzmin == visit_fzmax:
            single_sided = True
        # Create the final table
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            if single_sided:  # single-sided Zernike estimates
                for k in avg_keys:
                    raw_table[k][w] = adt[k][wadt]
            else:  # double-sided Zernike estimates
                wintra = adt[wadt]["focusZ"] == visit_fzmin
                wextra = adt[wadt]["focusZ"] == visit_fzmax
                for k in avg_keys:
                    # If one table has more rows than the other,
                    # trim the longer one
                    if wintra.sum() > wextra.sum():
                        wintra[wintra] = [True] * wextra.sum() + [False] * (wintra.sum() - wextra.sum())
                    elif wextra.sum() > wintra.sum():
                        wextra[wextra] = [True] * wintra.sum() + [False] * (wextra.sum() - wintra.sum())
                    # ought to be the same length now
                    raw_table[k][w] = 0.5 * (adt[k][wadt][wintra] + adt[k][wadt][wextra])
                    if k + "_intra" not in raw_table.colnames:
                        raw_table[k + "_intra"] = np.nan
                        raw_table[k + "_extra"] = np.nan
                    raw_table[k + "_intra"][w] = adt[k][wadt][wintra]
                    raw_table[k + "_extra"][w] = adt[k][wadt][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)


class AggregateAOSVisitTableCwfsTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableCwfs"

    @timeMethod
    def run(self, adt: Table, azr: Table, aza: Table) -> pipeBase.Struct:
        """
        Create overall summary tables for the visit.

        Parameters
        ----------
        adt : `astropy.table.Table`
            Aggregated donut table.
        azr : `astropy.table.Table`
            Aggregated raw Zernike table.
        aza : `astropy.table.Table`
            Aggregated average Zernike table.

        Returns
        -------
        struct : `pipeBase.Struct`
            Struct with `avg` and `raw` tables:
            avg_table : `astropy.table.Table`
                Table with average donut and Zernike values by detector.
            raw_table : `astropy.table.Table`
                Table with donut and Zernike values from every
                source that went into the averages.
        """
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
                    wintra[wintra] = [True] * wextra.sum() + [False] * (wintra.sum() - wextra.sum())
                elif wextra.sum() > wintra.sum():
                    wextra[wextra] = [True] * wintra.sum() + [False] * (wextra.sum() - wintra.sum())
                # ought to be the same length now
                raw_table[k][w] = 0.5 * (adt[k][wintra] + adt[k][wextra])
                if k + "_intra" not in raw_table.colnames:
                    raw_table[k + "_intra"] = np.nan
                    raw_table[k + "_extra"] = np.nan
                raw_table[k + "_intra"][w] = adt[k][wintra]
                raw_table[k + "_extra"][w] = adt[k][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)


class AggregateAOSVisitTableUnpairedCwfsTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableUnpairedCwfs"

    @timeMethod
    def run(self, adt: Table, azr: Table, aza: Table) -> pipeBase.Struct:
        """
        Create overall summary table for the visit.

        Parameters
        ----------
        adt : `astropy.table.Table`
            Aggregated donut table.
        azr : `astropy.table.Table`
            Aggregated raw Zernike table.
        aza : `astropy.table.Table`
            Aggregated average Zernike table.

        Returns
        -------
        struct : `pipeBase.Struct`
            Struct with `avg` and `raw` tables:
            avg_table : `astropy.table.Table`
                Table with average donut and Zernike values by detector.
            raw_table : `astropy.table.Table`
                Table with donut and Zernike values from every
                source that went into the averages.
        """
        dets = np.unique(adt["detector"])
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
            # Add keys into table. That way even if no donuts
            # are present, the column will exist.
            avg_table[k] = np.nan

        # Process average table
        for det in dets:
            w = avg_table["detector"] == det
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][adt["detector"] == det])

        # Process raw table
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            # Check if there are any matching rows
            if not np.any(wadt):
                continue

            for k in avg_keys:
                # ought to be the same length now
                raw_table[k][w] = adt[k][wadt]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)


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
