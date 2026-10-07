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

import warnings

import astropy.units as u
import galsim
import numpy as np
from astropy.table import Row, Table

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
from lsst.ts.wep.blitz.utils import CORNER_DEFOCAL_BY_DET_NAME, CORNER_PAIRS
from lsst.utils.timer import timeMethod

# Extra-focal corner sensor detector ids; the paired estimate is
# labelled by its extra-focal side, matching
# AggregateAOSVisitTableCwfsTask.
_EXTRA_FOCAL_DET_IDS = frozenset({191, 195, 199, 203})

# Corner -> extra-focal (SW0) detector name, the label every output row
# carries. Derived from ts_wep's CORNER_PAIRS so the two cannot drift.
_SW0_BY_CORNER = {corner: sw0 for corner, (sw0, _) in CORNER_PAIRS.items()}

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

# The nine Zernike array columns, each (nrow, len(nollIndices)).
_ZK_KEYS = (
    "zk_CCS",
    "zk_OCS",
    "zk_NW",
    "zk_intrinsic_CCS",
    "zk_intrinsic_OCS",
    "zk_intrinsic_NW",
    "zk_deviation_CCS",
    "zk_deviation_OCS",
    "zk_deviation_NW",
)

# Danish fit diagnostics PlotDonutFitsTask expects in meta["estimatorInfo"].
_ESTIMATOR_KEYS = (
    "fwhm",
    "model_dx",
    "model_dy",
    "chi_square",
    "model_flux",
    "model_bkg",
    "fit_success",
)


class FormatBlitzTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
):
    """Pipeline connections for FormatBlitzTask."""

    blitzResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzCornerTask containing "
            "selection metrics, fit results, and Noll-indexed Zernike "
            "array columns."
        ),
        name="donutBlitzCornerResults",
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
    """Convert a DonutBlitzCorner catalog into aggregateAOSVisitTable
    format.

    ``DonutBlitzCornerTask`` emits a per-donut catalog, whereas the
    donut_viz plotting/analysis path consumes the per-estimate
    ``aggregateAOSVisitTableRaw`` schema produced by
    ``AggregateAOSVisitTableTask``. This task rewrites the blitz catalog
    into that schema so blitz output can feed the existing plots without
    running the multi-step aggregate chain.

    Only the default ``wfEstimationMode="paired"`` is handled: each
    wavefront-fit group (one intra + one extra donut) maps to one output
    row. Groups of any other size -- which the other fitting modes
    produce -- are skipped, with a count logged.
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
        catalog : `astropy.table.QTable`
            Per-donut catalog from ``DonutBlitzCornerTask``, as built by
            ``lsst.ts.wep.blitz.catalogBuilder._build_donut_catalog``.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit info supplying boresight angles and MJD.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry. Unused -- the catalog carries ``det_name``
            directly -- but kept so the connection, and the signature
            donut_viz' other aggregate tasks share, stay the same.

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
            # A fully-shaped table with zero rows, not a bare Table(): the
            # consumers index columns unconditionally (PlotAOSTask reads
            # aos_raw["zk_OCS"]), so a column-less table is a KeyError
            # rather than an empty plot.
            return pipeBase.Struct(
                raw=self._emptyTable(nollIndices, meta, withSplits=True),
                avg=self._emptyTable(nollIndices, meta, withSplits=False),
            )

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
        """Group paired donuts into (extra_row, intra_row) pairs.

        Returns a list of dicts, one per wavefront-fit group holding both an
        extra- and an intra-focal donut.

        Notes
        -----
        Follows ``lsst.ts.wep.blitz.zernikesTable._groups_by_corner``, which
        reads the same catalog. A group is identified by ``group_id``, and
        ``group_id == ""`` marks a donut no fit consumed -- either surplus
        (a candidate with no partner) or rejected by selection. Filtering on
        it therefore subsumes a ``candidate`` check, since a non-candidate
        donut can never hold a group.

        The side of focus is not a column: in corner mode it is a property
        of the detector, so it comes from ``det_name`` via
        ``CORNER_DEFOCAL_BY_DET_NAME`` (SW0 extra, SW1 intra).
        """
        if len(catalog) == 0 or "group_id" not in catalog.colnames:
            return []

        # Arrow round-trips do not give plain str, so cast before comparing.
        groupIds = np.asarray(catalog["group_id"], dtype=str)
        detNames = np.asarray(catalog["det_name"], dtype=str)
        sides = np.array([CORNER_DEFOCAL_BY_DET_NAME.get(n, "") for n in detNames])

        pairs = []
        nonPaired = 0
        for groupId in dict.fromkeys(groupIds):  # first-seen order
            if not groupId:
                continue
            inGroup = groupIds == groupId
            extraRows = catalog[inGroup & (sides == "extra")]
            intraRows = catalog[inGroup & (sides == "intra")]
            if len(extraRows) != 1 or len(intraRows) != 1:
                # Exactly one donut per side is what "paired" mode means;
                # the other fitting modes group differently and this task
                # has no row shape for them.
                nonPaired += 1
                continue
            # Label the pair by its extra-focal sensor, matching
            # AggregateAOSVisitTableCwfsTask.
            extraName = str(extraRows["det_name"][0])
            detName = _SW0_BY_CORNER.get(extraName[:3], extraName)
            pairs.append(
                {
                    "group": groupId,
                    "extra": extraRows[0],
                    "intra": intraRows[0],
                    "detector": detName,
                }
            )
        if nonPaired:
            self.log.warning(
                "Skipped %d wavefront-fit group(s) that were not one extra + one intra donut; "
                "this task only handles wfEstimationMode='paired'.",
                nonPaired,
            )
        return pairs

    def _buildRawTable(self, pairs: list, nollIndices: np.ndarray, meta: dict) -> Table:
        """Build the per-pair raw table with frame-transformed Zernikes."""
        rtp = meta["rotTelPos"]
        q = meta["parallacticAngle"]

        # Per-pair Zernike arrays in CCS (µm). Deviation is a property of the
        # joint fit and so is replicated onto both members of the pair;
        # intrinsic differs per side, so average the two.
        #
        # The catalog stores these as dense Noll-indexed array columns --
        # zk[:, j] is Noll j, with no j-4 offset -- so the fitted subset is a
        # plain take(). The two columns have different widths:
        # zk_deviation_ccs is max(noll_indices)+1 while zk_intrinsic_ccs is
        # the fixed ts_wep _ZK_JMAX+1 (67), so they must be sliced
        # separately and the deviation take() bounds-checked.
        zkDevCCS = np.array([self._takeNoll(p["extra"], "zk_deviation_ccs", nollIndices) for p in pairs])
        # Averaged NaN-aware: a donut whose intrinsic calibration was
        # missing carries NaN, and on real data that happens to one side of
        # a pair often enough that a plain mean would throw away the good
        # side. All-NaN stays NaN.
        zkIntCCS = np.array(
            [
                self._pairMean(
                    self._takeNoll(p["extra"], "zk_intrinsic_ccs", nollIndices),
                    self._takeNoll(p["intra"], "zk_intrinsic_ccs", nollIndices),
                )
                for p in pairs
            ]
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

        raw["used"] = np.array([bool(p["extra"]["group_fit_success"]) for p in pairs])
        raw["detector"] = [p["detector"] for p in pairs]

        def _donutId(row: Row) -> str:
            # Keeps AggregateDonutTablesCwfsTask's visit_det_source format.
            # Note blitz' donut_id is a reference-catalog source id, not the
            # sequential index the non-blitz path uses, so these strings will
            # not match DonutStamp.donut_id values; blitz emits no
            # donutStamps, so nothing in this pipeline cross-references them.
            return f"{int(row['visit_id'])}_{int(row['det_id']):03d}_{int(row['donut_id'])}"

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
        zkKeys = list(_ZK_KEYS)
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
    def _pairMean(extra: np.ndarray, intra: np.ndarray) -> np.ndarray:
        """Mean of the two sides, ignoring a side that is NaN.

        NaN where both sides are, so a term missing from both donuts stays
        missing rather than becoming zero.
        """
        stacked = np.vstack([extra, intra])
        allNaN = np.isnan(stacked).all(axis=0)
        out = np.full(stacked.shape[1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[~allNaN] = np.nanmean(stacked[:, ~allNaN], axis=0)
        return out

    @staticmethod
    def _rowValue(row: Row, column: str, unit: u.UnitBase | None = None) -> np.ndarray:
        """Read a row value, converting via the parent column's unit.

        ``ArrowAstropy`` comes back as a plain `~astropy.table.Table`, so
        units live on the column and a row lookup yields a bare float or
        ndarray. Wrapping that in a `~astropy.units.Quantity` would silently
        call it dimensionless, so take the unit from the column instead.
        """
        value = np.asarray(row[column], dtype=float)
        if unit is None:
            return value
        colUnit = row.table[column].unit
        if colUnit is None:
            return value
        return (value * colUnit).to_value(unit)

    @classmethod
    def _takeNoll(cls, row: Row, column: str, nollIndices: np.ndarray) -> np.ndarray:
        """Select the fitted Noll terms from a dense Zernike array column.

        The column is Noll-indexed (``zk[j]`` is Noll j) and carried in
        microns. Indices beyond its width -- possible for
        ``zk_deviation_ccs``, whose width is ``max(noll_indices) + 1`` --
        come back NaN rather than raising, mirroring
        ``lsst.ts.wep.blitz.zernikesTable.build_zernikes_tables``.
        """
        if column not in row.colnames:
            return np.full(len(nollIndices), np.nan)
        dense = cls._rowValue(row, column, u.micron)
        out = np.full(len(nollIndices), np.nan)
        inRange = nollIndices < len(dense)
        out[inRange] = dense[nollIndices[inRange]]
        return out

    @staticmethod
    def _emptyTable(nollIndices: np.ndarray, meta: dict, withSplits: bool) -> Table:
        """A correctly-shaped output table with zero rows."""
        table = Table()
        for key in _ZK_KEYS:
            table[key] = np.zeros((0, len(nollIndices)))
        table["used"] = np.zeros(0, bool)
        table["detector"] = np.zeros(0, dtype="<U8")
        if withSplits:
            for key in ("extra_donut_id", "intra_donut_id", "donut_id_extra", "donut_id_intra"):
                table[key] = np.zeros(0, dtype="<U32")
        for key in _GEOM_KEYS:
            table[key] = np.zeros(0)
            if withSplits:
                table[key + "_intra"] = np.zeros(0)
                table[key + "_extra"] = np.zeros(0)
        table.meta = dict(meta)
        if withSplits:
            table.meta["estimatorInfo"] = {key: [] for key in _ESTIMATOR_KEYS}
        return table

    @classmethod
    def _sideGeometry(cls, rows: list, rtp: float, q: float) -> dict:
        """Compute geometry columns for one defocal side of every pair.

        Catalog values carry units, so each is converted explicitly. Field
        angles stay in radians and centroids in pixels, matching what
        AggregateAOSVisitTableTask writes.
        """

        def _val(key: str, unit: u.UnitBase) -> np.ndarray:
            return np.array([float(cls._rowValue(r, key, unit)) for r in rows])

        thxCCS = _val("thx_ccs", u.rad)
        thyCCS = _val("thy_ccs", u.rad)
        return {
            # Real sky coordinates now, NaN only where no reference
            # catalogue backed the selection.
            "coord_ra": _val("coord_ra", u.deg),
            "coord_dec": _val("coord_dec", u.deg),
            "centroid_x": _val("x_det", u.pix),
            "centroid_y": _val("y_det", u.pix),
            "thx_CCS": thxCCS,
            "thy_CCS": thyCCS,
            "thx_OCS": np.cos(rtp) * thxCCS - np.sin(rtp) * thyCCS,
            "thy_OCS": np.sin(rtp) * thxCCS + np.cos(rtp) * thyCCS,
            "th_N": np.cos(q) * thxCCS - np.sin(q) * thyCCS,
            "th_W": np.sin(q) * thxCCS + np.cos(q) * thyCCS,
            "snr": np.array([float(r["snr"]) for r in rows]),
        }

    @classmethod
    def _estimatorInfo(cls, pairs: list) -> dict:
        """Build a per-pair estimatorInfo dict from blitz fit diagnostics.

        Keyed on the extra-focal row; the ``group_*`` diagnostics are
        replicated across the pair, so either side would answer.
        """

        def _col(row: Row, key: str, unit: u.UnitBase | None = None) -> float:
            if key not in row.colnames:
                return float("nan")
            return float(cls._rowValue(row, key, unit))

        def _bkg(row: Row, nbkg: int) -> np.ndarray:
            # Kept 2-D: PlotDonutFitsTask.getModel indexes
            # model_bkg.shape[1] and unpacks each row as a tuple of
            # galsim Zernike coefficients.
            if "fit_bkg" not in row.colnames:
                return np.full(nbkg, np.nan)
            return np.atleast_1d(np.asarray(row["fit_bkg"], dtype=float))

        nbkg = 1
        for pair in pairs:
            if "fit_bkg" in pair["extra"].colnames:
                nbkg = max(nbkg, np.atleast_1d(pair["extra"]["fit_bkg"]).size)

        return {
            "fwhm": [_col(p["extra"], "group_fwhm", u.arcsec) for p in pairs],
            "model_dx": [_col(p["extra"], "fit_dx", u.arcsec) for p in pairs],
            "model_dy": [_col(p["extra"], "fit_dy", u.arcsec) for p in pairs],
            "chi_square": [_col(p["extra"], "group_fit_cost") for p in pairs],
            "model_flux": [_col(p["extra"], "fit_flux") for p in pairs],
            "model_bkg": np.array([_bkg(p["extra"], nbkg) for p in pairs]),
            "fit_success": [bool(p["extra"]["group_fit_success"]) for p in pairs],
        }
