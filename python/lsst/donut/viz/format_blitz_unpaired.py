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

__all__ = [
    "FormatBlitzUnpairedTaskConnections",
    "FormatBlitzUnpairedTaskConfig",
    "FormatBlitzUnpairedTask",
]

import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.afw.cameraGeom import Camera
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
)
from lsst.utils.timer import timeMethod

from .format_blitz import _GEOM_KEYS, FormatBlitzTask


class FormatBlitzUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
):
    """Pipeline connections for FormatBlitzUnpairedTask.

    The same datasets as the paired task, minus the camera: an unpaired row
    is labelled by the donut's own detector, so no corner lookup is needed.
    """

    blitzResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzCornerTask run with "
            "wfEstimationMode='unpaired', containing selection metrics, fit "
            "results, and Noll-indexed Zernike array columns."
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
    aggregateAOSRaw = connectionTypes.Output(
        doc="Visit-level table of individual donuts and their Zernikes.",
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


class FormatBlitzUnpairedTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FormatBlitzUnpairedTaskConnections,  # type: ignore
):
    """Configuration for FormatBlitzUnpairedTask."""

    pass


class FormatBlitzUnpairedTask(FormatBlitzTask):
    """Convert an unpaired DonutBlitzCorner catalog into
    aggregateAOSVisitTable format.

    The unpaired counterpart of `FormatBlitzTask`, for
    ``DonutBlitzCornerTask`` run with ``wfEstimationMode="unpaired"``, where
    every wavefront-fit group holds a single donut and so one output row is
    one donut rather than one intra/extra pair. It stands to the paired task
    as ``AggregateAOSVisitTableUnpairedTask`` stands to
    ``AggregateAOSVisitTableTask``, and writes the same schema those two
    produce: the eleven geometry columns with **no** ``_intra``/``_extra``
    splits, and a flat ``donut_id`` in place of the paired id columns.

    Rows are labelled by the donut's own ``det_name``, so all eight corner
    sensors can appear -- unlike the paired task, which labels a pair by its
    extra-focal (SW0) side. In unpaired mode the side of focus is the only
    thing distinguishing two donuts of a corner, so collapsing it would
    discard the distinction.

    Groups of any size other than one are skipped with a count logged, the
    mirror of the paired task's guard: a catalog from any other fitting mode
    produces an empty table loudly rather than mislabelled rows quietly.
    """

    # Its own config, since the connections differ from the paired task's:
    # mypy flags the narrowing, which is the standard PipelineTask pattern.
    ConfigClass = FormatBlitzUnpairedTaskConfig  # type: ignore[assignment]
    _DefaultName = "formatBlitzUnpaired"

    # One donut per row, so there is no other side to split against.
    _rawHasSplits = False

    @timeMethod
    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # As the paired task, minus the camera: these connections do not
        # declare one, since an unpaired row is labelled by its own detector.
        catalog = butlerQC.get(inputRefs.blitzResults).get(parameters={"strip_astropy_meta_yaml": False})
        visitInfo = butlerQC.get(inputRefs.visitInfos)[0]

        result = self.run(catalog, visitInfo, None)

        butlerQC.put(result.raw, outputRefs.aggregateAOSRaw)
        butlerQC.put(result.avg, outputRefs.aggregateAOSAvg)

    @staticmethod
    def _emptyIdColumns(table: Table) -> None:
        """One flat ``donut_id``, as the unpaired aggregator writes."""
        table["donut_id"] = np.zeros(0, dtype="<U32")

    def _collectGroups(self, catalog: Table, camera: Camera) -> list:
        """Collect the single-donut fit groups.

        Returns a list of dicts, one per wavefront-fit group holding exactly
        one donut. The row is stored under the ``"extra"`` key so the
        diagnostics helper inherited from the paired task, which reads that
        key, works unchanged -- the name is about dict layout here, not about
        which side of focus the donut is on.

        Notes
        -----
        As in the paired task, a group is identified by ``group_id`` and
        ``group_id == ""`` marks a donut no fit consumed, so filtering on it
        subsumes a ``candidate`` check.
        """
        if len(catalog) == 0 or "group_id" not in catalog.colnames:
            return []

        # Arrow round-trips do not give plain str, so cast before comparing.
        groupIds = np.asarray(catalog["group_id"], dtype=str)

        donuts = []
        nonSingle = 0
        for groupId in dict.fromkeys(groupIds):  # first-seen order
            if not groupId:
                continue
            rows = catalog[groupIds == groupId]
            if len(rows) != 1:
                nonSingle += 1
                continue
            donuts.append(
                {
                    "group": groupId,
                    "extra": rows[0],
                    "detector": str(rows["det_name"][0]),
                }
            )
        if nonSingle:
            self.log.warning(
                "Skipped %d wavefront-fit group(s) holding more than one donut; this task "
                "only handles wfEstimationMode='unpaired'.",
                nonSingle,
            )
        return donuts

    def _buildRawTable(self, donuts: list, nollIndices: np.ndarray, meta: dict) -> Table:
        """Build the per-donut raw table with frame-transformed Zernikes."""
        rtp = meta["rotTelPos"]
        q = meta["parallacticAngle"]

        rows = [d["extra"] for d in donuts]

        # Per-donut Zernike arrays in CCS (µm), taken from the one row of the
        # group. Unlike the paired case there is no second side to average the
        # intrinsic over. See FormatBlitzTask._takeNoll for why the two array
        # columns are sliced separately.
        zkDevCCS = np.array([self._takeNoll(r, "zk_deviation_ccs", nollIndices) for r in rows])
        zkIntCCS = np.array([self._takeNoll(r, "zk_intrinsic_ccs", nollIndices) for r in rows])
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

        raw["used"] = np.array([bool(r["group_fit_success"]) for r in rows])
        raw["detector"] = [d["detector"] for d in donuts]
        raw["donut_id"] = [self._donutId(r) for r in rows]

        # Geometry straight from the single row: no pair average, no splits.
        geom = self._sideGeometry(rows, rtp, q)
        for key in _GEOM_KEYS:
            raw[key] = geom[key]

        raw.meta = dict(meta)
        raw.meta["estimatorInfo"] = self._estimatorInfo(donuts)
        return raw
