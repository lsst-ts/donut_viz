# This file is part of donut_viz.
#
# Developed for the LSST Data Management System.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "FormatBlitzTaskConnections",
    "FormatBlitzTaskConfig",
    "FormatBlitzTask",
]

import galsim
import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.afw.cameraGeom import Camera
from lsst.afw.image import VisitInfo
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
)
from lsst.utils.timer import timeMethod

# Extra-focal corner sensor detector ids; the paired estimate is
# labelled by its extra-focal side, matching
# AggregateAOSVisitTableCwfsTask.
_EXTRA_FOCAL_DET_IDS = frozenset({191, 195, 199, 203})

# Geometry columns carried through from the blitz per-donut catalog to
# the aggregate per-pair table. Each becomes a pair-averaged column plus
# ``_intra`` and ``_extra`` split columns, mirroring
# AggregateAOSVisitTableTask.
_GEOM_KEYS = (
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
    "snr",
)


class FormatBlitzTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
):
    """Pipeline connections for FormatBlitzTask."""

    blitzResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzMonolithTask containing "
            "selection metrics, fit results, and per-Noll Zernikes."
        ),
        name="donutBlitzResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
    )
    visitInfos = connectionTypes.Input(
        doc="Visit info from the raw corner wavefront sensor exposures.",
        name="raw.visitInfo",
        storageClass="VisitInfo",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera geometry.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    aggregateAOSRaw = connectionTypes.Output(
        doc="Visit-level table of paired donuts and Zernikes.",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    aggregateAOSAvg = connectionTypes.Output(
        doc="Visit-level table of per-detector average donuts and Zernikes.",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
    )


class FormatBlitzTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FormatBlitzTaskConnections,  # type: ignore
):
    """Configuration for FormatBlitzTask."""

    pass


class FormatBlitzTask(pipeBase.PipelineTask):
    """Convert a DonutBlitzMonolith catalog into aggregateAOSVisitTable
    format.

    ``DonutBlitzMonolithTask`` emits a per-donut catalog, whereas the
    donut_viz plotting/analysis path consumes the per-estimate
    ``aggregateAOSVisitTableRaw`` schema produced by
    ``AggregateAOSVisitTableTask``. This task rewrites the blitz catalog
    into that schema so blitz output can feed the existing plots without
    running the multi-step aggregate chain.

    Only the DonutBlitzMonolith default ``wfEstimationMode="paired"`` is
    handled: each wavefront-fit group (one intra + one extra donut) maps
    to one output row.
    """

    ConfigClass = FormatBlitzTaskConfig
    _DefaultName = "formatBlitz"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # Preserve astropy meta (noll_indices etc.), as
        # DonutBlitzPlotTask does.
        catalog = butlerQC.get(inputRefs.blitzResults).get(parameters={"strip_astropy_meta_yaml": False})
        camera = butlerQC.get(inputRefs.camera)

        # Any one raw supplies the (visit-constant) visit info; the visitInfo
        # component is read directly, so no pixels are loaded.
        visitInfo = butlerQC.get(inputRefs.visitInfos)[0]

        result = self.run(catalog, visitInfo, camera)

        butlerQC.put(result.raw, outputRefs.aggregateAOSRaw)
        butlerQC.put(result.avg, outputRefs.aggregateAOSAvg)

    @timeMethod
    def run(self, catalog: Table, visitInfo: VisitInfo, camera: Camera) -> pipeBase.Struct:
        """Convert a blitz catalog to aggregateAOSVisitTable raw/avg tables.

        Parameters
        ----------
        catalog : `astropy.table.Table`
            Per-donut catalog from ``DonutBlitzMonolithTask._buildCatalog``.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit info supplying boresight angles and MJD.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry, used to resolve the extra-focal detector name.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Struct with ``raw`` (one row per intra/extra pair) and
            ``avg`` (one row per detector, averaged over used pairs)
            tables.

        Notes
        -----
        Needs the astropy ``meta`` (``noll_indices``). ``ArrowAstropy``
        objects strip it on a plain ``butler.get``, so load with
        ``parameters={"strip_astropy_meta_yaml": False}``.
        """
        meta = self._buildMeta(catalog, visitInfo)
        nollIndices = meta["nollIndices"]

        pairs = self._collectPairs(catalog, camera)

        if len(nollIndices) == 0 or len(pairs) == 0:
            empty = Table()
            empty.meta = meta
            return pipeBase.Struct(raw=empty, avg=empty.copy())

        raw = self._buildRawTable(pairs, nollIndices, meta)
        avg = self._buildAvgTable(raw, nollIndices, meta)
        return pipeBase.Struct(raw=raw, avg=avg)

    def _buildMeta(self, catalog: Table, visitInfo: VisitInfo) -> dict:
        """Build the visit-level metadata dict for the output tables."""
        q = visitInfo.boresightParAngle.asRadians()
        rot = visitInfo.boresightRotAngle.asRadians()
        raDec = visitInfo.boresightRaDec
        azAlt = visitInfo.boresightAzAlt

        band = str(catalog["band"][0]) if "band" in catalog.colnames and len(catalog) else ""

        return {
            "visit": int(visitInfo.id),
            "parallacticAngle": float(q),
            "rotAngle": float(rot),
            "rotTelPos": float(q - rot - np.pi / 2),
            "ra": float(raDec.getRa().asRadians()),
            "dec": float(raDec.getDec().asRadians()),
            "az": float(azAlt.getLongitude().asRadians()),
            "alt": float(azAlt.getLatitude().asRadians()),
            "band": band,
            "mjd": float(visitInfo.date.toAstropy().mjd),
            "nollIndices": np.array(list(catalog.meta.get("noll_indices", [])), dtype=int),
        }

    def _collectPairs(self, catalog: Table, camera: Camera) -> list:
        """Group accepted paired donuts into (extra_row, intra_row) pairs.

        Returns a list of dicts, one per wavefront-fit group holding both an
        extra- and an intra-focal donut. Groups missing a side are skipped.
        """
        if len(catalog) == 0 or "group" not in catalog.colnames:
            return []

        accepted = catalog["accepted"] if "accepted" in catalog.colnames else np.ones(len(catalog), bool)

        groups: dict = {}
        for i, row in enumerate(catalog):
            if not bool(accepted[i]):
                continue
            grp = int(row["group"])
            if grp < 0:
                continue
            groups.setdefault(grp, {})[str(row["defocal"])] = row

        pairs = []
        for grp in sorted(groups):
            sides = groups[grp]
            if "extra" not in sides or "intra" not in sides:
                continue
            extra, intra = sides["extra"], sides["intra"]
            detId = int(extra["det_id"])
            try:
                detName = camera[detId].getName()
            except (KeyError, LookupError):
                detName = str(extra["sensor"])
            pairs.append({"group": grp, "extra": extra, "intra": intra, "detector": detName})
        return pairs

    def _buildRawTable(self, pairs: list, nollIndices: np.ndarray, meta: dict) -> Table:
        """Build the per-pair raw table with frame-transformed Zernikes."""
        rtp = meta["rotTelPos"]
        q = meta["parallacticAngle"]

        devCols = [f"Z{j}_dev" for j in nollIndices]
        intCols = [f"Z{j}_intrinsic" for j in nollIndices]

        # Per-pair Zernike arrays in CCS (µm). Deviation is shared across the
        # pair; intrinsic differs per side, so average the two.
        zkDevCCS = np.array([[float(p["extra"][c]) for c in devCols] for p in pairs])
        zkIntCCS = np.array(
            [[0.5 * (float(p["extra"][c]) + float(p["intra"][c])) for c in intCols] for p in pairs]
        )
        zkCCS = zkIntCCS + zkDevCCS

        rotOCS, rotNW = self._rotationMatrices(nollIndices, rtp, q)

        raw = Table()
        raw["zk_CCS"] = zkCCS
        raw["zk_OCS"] = self._rotateZk(zkCCS, nollIndices, rotOCS)
        raw["zk_NW"] = self._rotateZk(zkCCS, nollIndices, rotNW)
        raw["zk_intrinsic_CCS"] = zkIntCCS
        raw["zk_intrinsic_OCS"] = self._rotateZk(zkIntCCS, nollIndices, rotOCS)
        raw["zk_intrinsic_NW"] = self._rotateZk(zkIntCCS, nollIndices, rotNW)
        raw["zk_deviation_CCS"] = zkDevCCS
        raw["zk_deviation_OCS"] = self._rotateZk(zkDevCCS, nollIndices, rotOCS)
        raw["zk_deviation_NW"] = self._rotateZk(zkDevCCS, nollIndices, rotNW)

        raw["used"] = np.array([bool(p["extra"]["fit_success"]) for p in pairs])
        raw["detector"] = [p["detector"] for p in pairs]

        def _donutId(row):
            return f"{int(row['visit_id'])}_{int(row['det_id']):03d}_{int(row['source_id'])}"

        raw["extra_donut_id"] = [_donutId(p["extra"]) for p in pairs]
        raw["intra_donut_id"] = [_donutId(p["intra"]) for p in pairs]
        raw["donut_id_extra"] = raw["extra_donut_id"]
        raw["donut_id_intra"] = raw["intra_donut_id"]

        # Per-side geometry, then pair-average plus intra/extra split columns.
        geomExtra = self._sideGeometry([p["extra"] for p in pairs], rtp, q)
        geomIntra = self._sideGeometry([p["intra"] for p in pairs], rtp, q)
        for k in _GEOM_KEYS:
            raw[k] = 0.5 * (geomExtra[k] + geomIntra[k])
            raw[k + "_intra"] = geomIntra[k]
            raw[k + "_extra"] = geomExtra[k]

        raw.meta = dict(meta)
        raw.meta["estimatorInfo"] = self._estimatorInfo(pairs)
        return raw

    def _buildAvgTable(self, raw: Table, nollIndices: np.ndarray, meta: dict) -> Table:
        """Average the raw per-pair table by detector over used pairs."""
        zkKeys = [
            "zk_CCS",
            "zk_OCS",
            "zk_NW",
            "zk_intrinsic_CCS",
            "zk_intrinsic_OCS",
            "zk_intrinsic_NW",
            "zk_deviation_CCS",
            "zk_deviation_OCS",
            "zk_deviation_NW",
        ]
        detArr = np.array([str(d) for d in raw["detector"]])
        usedArr = np.array(raw["used"], bool)
        detectors = list(dict.fromkeys(detArr))

        avg = Table()
        avg["detector"] = detectors
        avg["used"] = np.ones(len(detectors), bool)
        for k in zkKeys:
            avg[k] = np.full((len(detectors), len(nollIndices)), np.nan)
        for k in _GEOM_KEYS:
            avg[k] = np.full(len(detectors), np.nan)

        for i, det in enumerate(detectors):
            w = (detArr == det) & usedArr
            if not np.any(w):
                w = detArr == det
            for k in zkKeys:
                avg[k][i] = np.nanmean(np.atleast_2d(raw[k][w]), axis=0)
            for k in _GEOM_KEYS:
                avg[k][i] = self._nanmean(np.asarray(raw[k][w], dtype=float))

        avg.meta = dict(meta)
        return avg

    @staticmethod
    def _nanmean(values: np.ndarray) -> float:
        """NaN-mean returning NaN (no warning) for an all-NaN slice."""
        if values.size == 0 or not np.any(np.isfinite(values)):
            return float("nan")
        return float(np.nanmean(values))

    @staticmethod
    def _rotationMatrices(nollIndices: np.ndarray, rtp: float, q: float) -> tuple:
        """Return (OCS, NW) Zernike rotation matrices, mirroring donut_viz."""
        jmax = int(np.max(nollIndices))
        rotOCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:, 4:]
        rotNW = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:, 4:]
        return rotOCS, rotNW

    @staticmethod
    def _rotateZk(zkSub: np.ndarray, nollIndices: np.ndarray, rotMat: np.ndarray) -> np.ndarray:
        """Rotate per-Noll Zernike coefficients into another frame.

        Follows AggregateZernikeTablesTask: scatter the fitted
        coefficients into a dense Noll 4..jmax array, apply the rotation
        matrix, then re-select the fitted indices.
        """
        jmin = int(np.min(nollIndices))
        jmax = int(np.max(nollIndices))
        full = np.zeros((len(zkSub), jmax - jmin + 1))
        full[:, nollIndices - 4] = zkSub
        return (full @ rotMat)[:, nollIndices - 4]

    @staticmethod
    def _sideGeometry(rows: list, rtp: float, q: float) -> dict:
        """Compute geometry columns for one defocal side of every pair."""
        thxCCS = np.array([float(r["fa_x_ccs"]) for r in rows])
        thyCCS = np.array([float(r["fa_y_ccs"]) for r in rows])
        return {
            "coord_ra": np.full(len(rows), np.nan),
            "coord_dec": np.full(len(rows), np.nan),
            "centroid_x": np.array([float(r["centroid_x_raw"]) for r in rows]),
            "centroid_y": np.array([float(r["centroid_y_raw"]) for r in rows]),
            "thx_CCS": thxCCS,
            "thy_CCS": thyCCS,
            "thx_OCS": np.cos(rtp) * thxCCS - np.sin(rtp) * thyCCS,
            "thy_OCS": np.sin(rtp) * thxCCS + np.cos(rtp) * thyCCS,
            "th_N": np.cos(q) * thxCCS - np.sin(q) * thyCCS,
            "th_W": np.sin(q) * thxCCS + np.cos(q) * thyCCS,
            "snr": np.array([float(r["snr"]) for r in rows]),
        }

    @staticmethod
    def _estimatorInfo(pairs: list) -> dict:
        """Build a per-pair estimatorInfo dict from blitz fit diagnostics."""

        def _col(row, key):
            return float(row[key]) if key in row.colnames else float("nan")

        return {
            "fwhm": [_col(p["extra"], "fit_fwhm") for p in pairs],
            "model_dx": [_col(p["extra"], "fit_dx") for p in pairs],
            "model_dy": [_col(p["extra"], "fit_dy") for p in pairs],
            "chi_square": [_col(p["extra"], "fit_cost") for p in pairs],
            "model_flux": [_col(p["extra"], "fit_flux") for p in pairs],
            "model_bkg": [_col(p["extra"], "fit_bkg") for p in pairs],
            "fit_success": [bool(p["extra"]["fit_success"]) for p in pairs],
        }
