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

import inspect
import re
import unittest
from types import ModuleType

import numpy as np
from astropy.table import Table
from test_format_blitz import (
    CORNERS,
    NOLL_INDICES,
    _applyUnits,
    _makeCatalog,
    _makeDonutRow,
    _makeVisitInfo,
)

import lsst.utils.tests
from lsst.afw.image import VisitInfo
from lsst.donut.viz.format_blitz_unpaired import (
    FormatBlitzUnpairedTask,
    FormatBlitzUnpairedTaskConfig,
    FormatBlitzUnpairedTaskConnections,
)

# The eleven geometry columns the unpaired aggregate schema carries. Unlike
# the paired table these appear once each, with no _intra/_extra splits.
GEOM_KEYS = (
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

META_KEYS = {
    "alt",
    "az",
    "band",
    "dec",
    "mjd",
    "nollIndices",
    "parallacticAngle",
    "ra",
    "rotAngle",
    "rotTelPos",
    "visit",
}


def _makeUnpairedCatalog(surplus: bool = True) -> Table:
    """A catalog as DonutBlitzCornerTask emits it with wfEstimationMode
    set to "unpaired": one donut per fit group.

    Covers both sides of focus on both corners, so the output must keep all
    four detector names rather than collapsing each corner onto its SW0.
    """
    rows = []
    donutId = 5601386396281737472
    for grp, (extraDet, intraDet, extraName, intraName) in enumerate(CORNERS):
        for detId, detName in ((extraDet, extraName), (intraDet, intraName)):
            rows.append(
                _makeDonutRow(
                    f"{detName}_{donutId}",
                    detId,
                    detName,
                    donutId,
                    seed=grp * 10 + detId,
                )
            )
            donutId += 1

    if surplus:
        # A candidate donut no fit consumed; group_id "" must not reach the
        # output, exactly as on the paired path.
        rows.append(_makeDonutRow("", CORNERS[0][0], CORNERS[0][2], donutId, seed=99))

    catalog = _applyUnits(Table(rows))
    catalog.meta["noll_indices"] = list(NOLL_INDICES)
    return catalog


class TestFormatBlitzUnpairedTask(lsst.utils.tests.TestCase):
    # Built once in setUpClass; declared here so they are visible to mypy.
    visitInfo: VisitInfo
    task: FormatBlitzUnpairedTask
    catalog: Table

    @classmethod
    def setUpClass(cls) -> None:
        cls.visitInfo = _makeVisitInfo()

    def setUp(self) -> None:
        self.task = FormatBlitzUnpairedTask(config=FormatBlitzUnpairedTaskConfig())
        self.catalog = _makeUnpairedCatalog()

    def testRowCountAndColumns(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        # One row per donut a fit consumed: two sides x two corners.
        self.assertEqual(len(raw), 2 * len(CORNERS))

        for col in (
            "zk_CCS",
            "zk_OCS",
            "zk_NW",
            "zk_intrinsic_CCS",
            "zk_intrinsic_OCS",
            "zk_intrinsic_NW",
            "zk_deviation_CCS",
            "zk_deviation_OCS",
            "zk_deviation_NW",
            "used",
            "detector",
            "donut_id",
        ):
            self.assertIn(col, raw.colnames)
        for col in GEOM_KEYS:
            self.assertIn(col, raw.colnames)

        self.assertEqual(raw["zk_CCS"].shape, (2 * len(CORNERS), len(NOLL_INDICES)))

    def testNoSplitColumns(self) -> None:
        """A single donut has no other side, so no splits are written."""
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        splits = [c for c in raw.colnames if c.endswith(("_intra", "_extra"))]
        self.assertEqual(splits, [])
        # Nor the paired task's four donut-id variants.
        for col in ("extra_donut_id", "intra_donut_id", "donut_id_extra", "donut_id_intra"):
            self.assertNotIn(col, raw.colnames)

    def testDetectorKeepsIntraFocalName(self) -> None:
        """Rows are labelled by their own sensor, including SW1.

        The paired task relabels a pair onto its extra-focal SW0; doing that
        here would discard the only thing distinguishing the two donuts of a
        corner.
        """
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        dets = sorted({str(d) for d in raw["detector"]})
        self.assertEqual(dets, ["R00_SW0", "R00_SW1", "R04_SW0", "R04_SW1"])

    def testZernikeRelationship(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        np.testing.assert_allclose(raw["zk_CCS"], raw["zk_intrinsic_CCS"] + raw["zk_deviation_CCS"])
        # The deviation is the single donut's own, taken straight from its
        # array column rather than shared with a partner.
        catDev = np.asarray(self.catalog["zk_deviation_ccs"])
        groupIds = np.asarray(self.catalog["group_id"], dtype=str)
        donutIds = [str(d) for d in raw["donut_id"]]
        for i, donutId in enumerate(donutIds):
            source = int(donutId.split("_")[-1])
            member = next(k for k, g in enumerate(groupIds) if g.endswith(f"_{source}") and g)
            np.testing.assert_allclose(raw["zk_deviation_CCS"][i], catDev[member][np.array(NOLL_INDICES)])

    def testGeometryFromSingleRow(self) -> None:
        """Geometry is the donut's own, not an average of two."""
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        grouped = np.asarray(self.catalog["group_id"], dtype=str) != ""
        np.testing.assert_allclose(np.sort(raw["snr"]), np.sort(np.asarray(self.catalog["snr"])[grouped]))
        np.testing.assert_allclose(
            np.sort(raw["centroid_x"]),
            np.sort(np.asarray(self.catalog["x_det"])[grouped]),
        )
        self.assertTrue(np.all(np.isfinite(raw["coord_ra"])))

    def testMetadata(self) -> None:
        result = self.task.run(self.catalog, self.visitInfo, None)
        self.assertEqual(set(result.raw.meta), META_KEYS | {"estimatorInfo"})
        # The avg table carries no estimatorInfo, matching the unpaired CWFS
        # contract in test_donut_viz_pipeline_unpaired_cwfs_sensors.
        self.assertEqual(set(result.avg.meta), META_KEYS)
        self.assertEqual(len(result.raw.meta["estimatorInfo"]["fwhm"]), 2 * len(CORNERS))

    def testModelBkgIsTwoDimensional(self) -> None:
        """PlotDonutFitsTask indexes model_bkg.shape[1]."""
        raw = self.task.run(self.catalog, self.visitInfo, None).raw
        modelBkg = np.asarray(raw.meta["estimatorInfo"]["model_bkg"])
        self.assertEqual(modelBkg.ndim, 2)
        self.assertEqual(modelBkg.shape[0], 2 * len(CORNERS))

    def testAvgTable(self) -> None:
        result = self.task.run(self.catalog, self.visitInfo, None)
        avg = result.avg
        # One row per detector that contributed a donut.
        self.assertEqual(len(avg), 2 * len(CORNERS))
        self.assertEqual(avg["zk_CCS"].shape, (2 * len(CORNERS), len(NOLL_INDICES)))
        # One donut per detector here, so the average equals the raw row.
        for det in avg["detector"]:
            aRow = avg[avg["detector"] == det]
            rRow = result.raw[np.array([str(d) for d in result.raw["detector"]]) == det]
            np.testing.assert_allclose(aRow["zk_CCS"][0], rRow["zk_CCS"][0])

    def testSurplusDonutExcluded(self) -> None:
        withSurplus = self.task.run(self.catalog, self.visitInfo, None).raw
        without = self.task.run(_makeUnpairedCatalog(surplus=False), self.visitInfo, None).raw
        self.assertEqual(len(withSurplus), 2 * len(CORNERS))
        self.assertEqual(len(without), 2 * len(CORNERS))

    def testPairedCatalogYieldsEmptyShapedTable(self) -> None:
        """A paired catalog must fail loudly, not emit mislabelled rows.

        The inverse of the paired task's non-paired guard. The table is still
        correctly shaped so consumers that index columns unconditionally see
        an empty plot rather than a KeyError.
        """
        paired = _makeCatalog()
        with self.assertLogs(self.task.log.name, level="WARNING") as cm:
            result = self.task.run(paired, self.visitInfo, None)
        self.assertIn("only handles wfEstimationMode='unpaired'", "".join(cm.output))
        self.assertEqual(len(result.raw), 0)
        for col in ("zk_CCS", "zk_OCS", "used", "detector", "donut_id", "snr"):
            self.assertIn(col, result.raw.colnames)
        self.assertEqual(result.raw["zk_CCS"].shape, (0, len(NOLL_INDICES)))
        self.assertNotIn("thx_CCS_intra", result.raw.colnames)

    def testEmptyCatalog(self) -> None:
        empty = Table()
        empty.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(empty, self.visitInfo, None)
        self.assertEqual(len(result.raw), 0)
        self.assertEqual(len(result.avg), 0)

    def testFitFailureMarkedUnused(self) -> None:
        rows = [
            _makeDonutRow(f"{CORNERS[0][2]}_1", CORNERS[0][0], CORNERS[0][2], 1, seed=1, fitSuccess=False)
        ]
        catalog = _applyUnits(Table(rows))
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(catalog, self.visitInfo, None)
        self.assertEqual(len(result.raw), 1)
        self.assertFalse(bool(result.raw["used"][0]))
        self.assertFalse(result.raw.meta["estimatorInfo"]["fit_success"][0])

    def testRunQuantumUsesOnlyDeclaredConnections(self) -> None:
        """``runQuantum`` must not reach for a connection it does not declare.

        It is overridden rather than inherited precisely because the paired
        version reads ``inputRefs.camera``, which these connections drop. A
        mismatch only shows up under a real quantum, so assert the contract
        here: every attribute ``runQuantum`` touches exists on the connection
        class.
        """
        source = inspect.getsource(FormatBlitzUnpairedTask.runQuantum)
        used = set(re.findall(r"(?:inputRefs|outputRefs)\.([A-Za-z_]\w*)", source))
        declared = set(FormatBlitzUnpairedTaskConnections.allConnections)
        self.assertTrue(used, "found no connection references to check")
        self.assertEqual(used - declared, set())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: ModuleType) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
