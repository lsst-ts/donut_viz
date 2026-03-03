import typing
from copy import copy

import galsim
import numpy as np
from astropy import units as u
from astropy.table import QTable, Table, vstack

import lsst.pipe.base as pipeBase
from lsst.geom import radians
from lsst.pipe.base import connectionTypes as ct
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateZernikeTablesTaskConnections",
    "AggregateZernikeTablesTaskConfig",
    "AggregateZernikeTablesTask",
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
