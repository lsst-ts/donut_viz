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

import unittest
from types import ModuleType

import astropy.units as u
import galsim
import numpy as np
from astropy.table import Table

import lsst.utils.tests
from lsst.afw.cameraGeom import Camera
from lsst.afw.coord import Observatory
from lsst.afw.image import VisitInfo
from lsst.daf.base import DateTime
from lsst.donut.viz.format_blitz import FormatBlitzTask, FormatBlitzTaskConfig
from lsst.geom import SpherePoint, degrees
from lsst.obs.lsst import LsstCam

# DonutBlitzCorner default fitted Noll indices.
NOLL_INDICES = list(range(4, 20)) + list(range(22, 27))

# Widths of the two dense Zernike array columns. They differ: the deviation
# column spans Noll 0..max(fitted) while the intrinsic column is always the
# fixed ts_wep _ZK_JMAX + 1, so the fixture must reproduce both.
ZK_DEV_WIDTH = max(NOLL_INDICES) + 1
ZK_INTRINSIC_WIDTH = 67

# (extra det id, intra det id, extra sensor, intra sensor) for two corners.
CORNERS = [
    (191, 192, "R00_SW0", "R00_SW1"),
    (195, 196, "R04_SW0", "R04_SW1"),
]


def _makeVisitInfo() -> VisitInfo:
    """A VisitInfo with a finite, computed parallactic angle.

    ``boresightParAngle`` is derived (from era, boresightRaDec, observatory),
    so it cannot be set directly; supply the inputs it is computed from.
    """
    return VisitInfo(
        id=9999,
        era=1.5 * degrees,
        boresightRaDec=SpherePoint(30.0 * degrees, -20.0 * degrees),
        boresightAzAlt=SpherePoint(95.0 * degrees, 60.0 * degrees),
        boresightRotAngle=10.0 * degrees,
        observatory=Observatory(-70.7494 * degrees, -30.2446 * degrees, 2663.0),
        date=DateTime(60000.0, DateTime.MJD, DateTime.TAI),
    )


def _makeDonutRow(
    groupId: str,
    detId: int,
    detName: str,
    donutId: int,
    seed: int,
    fitSuccess: bool = True,
    nanIntrinsic: bool = False,
) -> dict:
    """One per-donut blitz catalog row as a dict.

    Mirrors the schema ``lsst.ts.wep.blitz.catalogBuilder`` emits: the side
    of focus is implied by ``det_name`` (SW0 extra, SW1 intra) rather than
    carried in a column, and the Zernikes are dense Noll-indexed arrays
    rather than per-term scalars.
    """
    rng = np.random.default_rng(seed)

    # The deviation is a property of the joint fit, so it is replicated onto
    # every member row; key it on the group so both sides agree.
    devRng = np.random.default_rng(abs(hash(groupId)) % (2**31))
    zkDev = np.zeros(ZK_DEV_WIDTH)
    for j in NOLL_INDICES:
        zkDev[j] = devRng.uniform(-0.5, 0.5)

    # The intrinsic differs per donut, and is NaN where the calibration was
    # missing -- which happens to one side of a pair on real data.
    zkIntrinsic = np.zeros(ZK_INTRINSIC_WIDTH)
    for j in NOLL_INDICES:
        zkIntrinsic[j] = np.nan if nanIntrinsic else rng.uniform(-0.2, 0.2)

    return {
        "visit_id": 9999,
        "det_id": detId,
        "det_name": detName,
        "donut_id": donutId,
        "band": "r",
        "candidate": True,
        "group_id": groupId,
        "group_size": 2,
        "group_fit_success": fitSuccess,
        "thx_ccs": float(rng.uniform(-0.03, 0.03)),
        "thy_ccs": float(rng.uniform(-0.03, 0.03)),
        "x_det": float(rng.uniform(0, 4000)),
        "y_det": float(rng.uniform(0, 4000)),
        "coord_ra": float(rng.uniform(29.0, 31.0)),
        "coord_dec": float(rng.uniform(-21.0, -19.0)),
        "snr": float(rng.uniform(100, 2000)),
        "group_fwhm": 1.1,
        "fit_dx": 0.2,
        "fit_dy": -0.1,
        "group_fit_cost": 12.3,
        "fit_flux": 1e6,
        "fit_bkg": np.array([5.0]),
        "zk_deviation_ccs": zkDev,
        "zk_intrinsic_ccs": zkIntrinsic,
    }


def _isExtra(catalog: Table) -> np.ndarray:
    """Mask of extra-focal rows, from det_name (SW0 extra, SW1 intra)."""
    return np.array([str(n).endswith("SW0") for n in catalog["det_name"]])


def _applyUnits(catalog: Table) -> Table:
    """Attach the units catalogBuilder sets, which the task converts from."""
    for name, unit in (
        ("thx_ccs", u.rad),
        ("thy_ccs", u.rad),
        ("x_det", u.pix),
        ("y_det", u.pix),
        ("coord_ra", u.deg),
        ("coord_dec", u.deg),
        ("group_fwhm", u.arcsec),
        ("fit_dx", u.arcsec),
        ("fit_dy", u.arcsec),
        ("zk_deviation_ccs", u.micron),
        ("zk_intrinsic_ccs", u.micron),
    ):
        if name in catalog.colnames:
            catalog[name].unit = unit
    return catalog


def _makeCatalog(surplus: bool = True) -> Table:
    """A paired catalog, by default with one surplus (ungrouped) donut."""
    rows = []
    donutId = 5601386396281737472
    for grp, (extraDet, intraDet, extraName, intraName) in enumerate(CORNERS):
        groupId = f"{extraName[:3]}_{donutId}_{donutId + 1}"
        rows.append(_makeDonutRow(groupId, extraDet, extraName, donutId, seed=grp * 10 + 1))
        rows.append(_makeDonutRow(groupId, intraDet, intraName, donutId + 1, seed=grp * 10 + 2))
        donutId += 2

    if surplus:
        # A candidate donut no fit consumed: group_id "" is how the catalog
        # marks both surplus and selection-rejected donuts, and it must not
        # reach the output.
        rows.append(_makeDonutRow("", CORNERS[0][0], CORNERS[0][2], donutId, seed=99))

    catalog = _applyUnits(Table(rows))
    catalog.meta["noll_indices"] = list(NOLL_INDICES)
    return catalog


class TestFormatBlitzTask(lsst.utils.tests.TestCase):
    # Built once in setUpClass; declared here so they are visible to mypy.
    camera: Camera
    visitInfo: VisitInfo
    task: FormatBlitzTask
    catalog: Table

    @classmethod
    def setUpClass(cls) -> None:
        cls.camera = LsstCam().getCamera()
        cls.visitInfo = _makeVisitInfo()

    def setUp(self) -> None:
        self.task = FormatBlitzTask(config=FormatBlitzTaskConfig())
        self.catalog = _makeCatalog()

    def testRowCountAndColumns(self) -> None:
        result = self.task.run(self.catalog, self.visitInfo, self.camera)
        raw = result.raw
        self.assertEqual(len(raw), len(CORNERS))

        expectedCols = [
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
            "extra_donut_id",
            "intra_donut_id",
            "donut_id_extra",
            "donut_id_intra",
        ]
        for col in expectedCols:
            self.assertIn(col, raw.colnames)
        for base in (
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
        ):
            for col in (base, base + "_intra", base + "_extra"):
                self.assertIn(col, raw.colnames)

        # Zernike arrays span the fitted Noll indices.
        self.assertEqual(raw["zk_CCS"].shape, (len(CORNERS), len(NOLL_INDICES)))

        # Detector is labelled by the extra-focal sensor.
        self.assertEqual(sorted(raw["detector"]), ["R00_SW0", "R04_SW0"])

    def testZernikeRelationship(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        # CCS total = intrinsic + deviation.
        np.testing.assert_allclose(raw["zk_CCS"], raw["zk_intrinsic_CCS"] + raw["zk_deviation_CCS"])
        # The deviation is replicated across the pair, so the output row
        # reproduces the fitted terms of the input array column.
        catDev = np.asarray(self.catalog["zk_deviation_ccs"])
        groupIds = np.asarray(self.catalog["group_id"], dtype=str)
        for i, det in enumerate(raw["detector"]):
            corner = str(det)[:3]
            member = next(k for k, g in enumerate(groupIds) if g.startswith(f"{corner}_"))
            np.testing.assert_allclose(raw["zk_deviation_CCS"][i], catDev[member][np.array(NOLL_INDICES)])

    def testFrameTransforms(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        noll = np.array(NOLL_INDICES)
        jmin, jmax = noll.min(), noll.max()
        rtp = raw.meta["rotTelPos"]

        # Independently recompute zk_OCS from zk_CCS.
        rotOCS = galsim.zernike.zernikeRotMatrix(int(jmax), -rtp)[4:, 4:]
        full = np.zeros((len(raw), jmax - jmin + 1))
        full[:, noll - 4] = raw["zk_CCS"]
        expectedOCS = (full @ rotOCS)[:, noll - 4]
        np.testing.assert_allclose(raw["zk_OCS"], expectedOCS)

        # Field-angle OCS transform for the averaged column.
        np.testing.assert_allclose(
            raw["thx_OCS"],
            np.cos(rtp) * raw["thx_CCS"] - np.sin(rtp) * raw["thy_CCS"],
        )

    def testGeometryAveraging(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        for base in ("centroid_x", "thx_CCS", "snr"):
            np.testing.assert_allclose(raw[base], 0.5 * (raw[base + "_intra"] + raw[base + "_extra"]))
        # The catalog does carry per-donut sky coords, so these come
        # through rather than being NaN-filled.
        self.assertTrue(np.all(np.isfinite(raw["coord_ra"])))
        self.assertTrue(np.all(np.isfinite(raw["coord_dec"])))

    def testUnitsConverted(self) -> None:
        """Field angles stay in radians and centroids in pixels."""
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        catalog = self.catalog
        grouped = np.asarray(catalog["group_id"], dtype=str) != ""
        self.assertLess(np.max(np.abs(raw["thx_CCS"])), 0.1)
        np.testing.assert_allclose(
            np.sort(raw["centroid_x_extra"]),
            np.sort(np.asarray(catalog["x_det"])[grouped & _isExtra(catalog)]),
        )
        # fwhm is arcsec in and arcsec out.
        self.assertAlmostEqual(raw.meta["estimatorInfo"]["fwhm"][0], 1.1)

    def testSurplusDonutExcluded(self) -> None:
        """A donut with group_id "" never reaches the output."""
        withSurplus = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        withoutSurplus = self.task.run(_makeCatalog(surplus=False), self.visitInfo, self.camera).raw
        self.assertEqual(len(withSurplus), len(CORNERS))
        self.assertEqual(len(withoutSurplus), len(CORNERS))

    def testNanIntrinsicOneSide(self) -> None:
        """One side missing its intrinsic calibration keeps the other.

        Real catalogs NaN the intrinsic Zernikes of a donut whose
        calibration was absent, and it lands on just one side of a pair
        often enough that a plain mean would discard the good side.
        """
        rows = []
        donutId = 10
        for grp, (extraDet, intraDet, extraName, intraName) in enumerate(CORNERS):
            groupId = f"{extraName[:3]}_{donutId}_{donutId + 1}"
            rows.append(_makeDonutRow(groupId, extraDet, extraName, donutId, seed=grp * 10 + 1))
            rows.append(
                _makeDonutRow(groupId, intraDet, intraName, donutId + 1, seed=grp * 10 + 2, nanIntrinsic=True)
            )
            donutId += 2
        catalog = _applyUnits(Table(rows))
        catalog.meta["noll_indices"] = list(NOLL_INDICES)

        raw = self.task.run(catalog, self.visitInfo, self.camera).raw
        self.assertEqual(len(raw), len(CORNERS))
        self.assertTrue(np.all(np.isfinite(raw["zk_intrinsic_CCS"])))
        # The surviving value is the extra side's, not a half of it.
        extraIntrinsic = np.asarray(catalog["zk_intrinsic_ccs"])[0][np.array(NOLL_INDICES)]
        np.testing.assert_allclose(raw["zk_intrinsic_CCS"][0], extraIntrinsic)

    def testFitFailureMarkedUnused(self) -> None:
        """group_fit_success False lands in `used`, and the row survives."""
        rows = []
        extraDet, intraDet, extraName, intraName = CORNERS[0]
        groupId = f"{extraName[:3]}_1_2"
        rows.append(_makeDonutRow(groupId, extraDet, extraName, 1, seed=1, fitSuccess=False))
        rows.append(_makeDonutRow(groupId, intraDet, intraName, 2, seed=2, fitSuccess=False))
        catalog = _applyUnits(Table(rows))
        catalog.meta["noll_indices"] = list(NOLL_INDICES)

        result = self.task.run(catalog, self.visitInfo, self.camera)
        self.assertEqual(len(result.raw), 1)
        self.assertFalse(bool(result.raw["used"][0]))
        self.assertFalse(result.raw.meta["estimatorInfo"]["fit_success"][0])
        # Averaging falls back to all rows when none are used, so the avg
        # table still describes the detector rather than being all-NaN.
        self.assertEqual(len(result.avg), 1)
        self.assertTrue(np.all(np.isfinite(result.avg["zk_deviation_CCS"][0])))

    def testMissingBkgColumn(self) -> None:
        """bkgOrder=-1 drops fit_bkg entirely; model_bkg stays 2-D."""
        catalog = self.catalog.copy()
        catalog.remove_column("fit_bkg")
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        raw = self.task.run(catalog, self.visitInfo, self.camera).raw
        modelBkg = np.asarray(raw.meta["estimatorInfo"]["model_bkg"])
        self.assertEqual(modelBkg.ndim, 2)
        self.assertTrue(np.all(np.isnan(modelBkg)))

    def testModelBkgIsTwoDimensional(self) -> None:
        """PlotDonutFitsTask indexes model_bkg.shape[1]."""
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        modelBkg = np.asarray(raw.meta["estimatorInfo"]["model_bkg"])
        self.assertEqual(modelBkg.ndim, 2)
        self.assertEqual(modelBkg.shape[0], len(CORNERS))

    def testNonPairedGroupSkipped(self) -> None:
        """A group that is not one extra + one intra donut is dropped."""
        rows = []
        extraDet, _, extraName, _ = CORNERS[0]
        groupId = f"{extraName[:3]}_1_2"
        # Two extra-focal donuts, no intra: a full_detector-style group.
        rows.append(_makeDonutRow(groupId, extraDet, extraName, 1, seed=1))
        rows.append(_makeDonutRow(groupId, extraDet, extraName, 2, seed=2))
        catalog = _applyUnits(Table(rows))
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(catalog, self.visitInfo, self.camera)
        self.assertEqual(len(result.raw), 0)
        # Still correctly shaped, so consumers indexing columns do not fail.
        self.assertIn("zk_OCS", result.raw.colnames)

    def testMetadata(self) -> None:
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        q = self.visitInfo.boresightParAngle.asRadians()
        rot = self.visitInfo.boresightRotAngle.asRadians()
        self.assertEqual(raw.meta["visit"], 9999)
        self.assertTrue(np.isfinite(q))
        self.assertAlmostEqual(raw.meta["parallacticAngle"], q)
        self.assertAlmostEqual(raw.meta["rotAngle"], rot)
        self.assertAlmostEqual(raw.meta["rotTelPos"], q - rot - np.pi / 2)
        self.assertEqual(raw.meta["band"], "r")
        self.assertAlmostEqual(raw.meta["mjd"], 60000.0)
        self.assertIn("estimatorInfo", raw.meta)
        self.assertEqual(len(raw.meta["estimatorInfo"]["fwhm"]), len(CORNERS))

    def testAvgTable(self) -> None:
        result = self.task.run(self.catalog, self.visitInfo, self.camera)
        avg = result.avg
        self.assertEqual(len(avg), len(CORNERS))
        self.assertEqual(avg["zk_CCS"].shape, (len(CORNERS), len(NOLL_INDICES)))
        # One pair per detector -> avg equals the single raw row.
        for det in avg["detector"]:
            aRow = avg[avg["detector"] == det]
            rRow = result.raw[np.array([str(d) for d in result.raw["detector"]]) == det]
            np.testing.assert_allclose(aRow["zk_CCS"][0], rRow["zk_CCS"][0])

    def testEmptyCatalog(self) -> None:
        empty = Table()
        empty.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(empty, self.visitInfo, self.camera)
        self.assertEqual(len(result.raw), 0)
        self.assertEqual(len(result.avg), 0)

    def testUnmatchedGroupSkipped(self) -> None:
        # Drop the intra donut of the first corner; that group should vanish,
        # since this task only has a row shape for a complete pair.
        catalog = self.catalog[[i for i in range(len(self.catalog)) if i != 1]]
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        raw = self.task.run(catalog, self.visitInfo, self.camera).raw
        self.assertEqual(len(raw), len(CORNERS) - 1)
        self.assertEqual(list(raw["detector"]), ["R04_SW0"])

    def testAllGroupsEmptyStillShaped(self) -> None:
        """Rows present but none grouped -- the realistic degenerate case.

        A column-less table here would make PlotAOSTask raise KeyError on
        zk_OCS rather than draw an empty plot.
        """
        rows = [_makeDonutRow("", CORNERS[0][0], CORNERS[0][2], i, seed=i) for i in range(3)]
        catalog = _applyUnits(Table(rows))
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(catalog, self.visitInfo, self.camera)
        self.assertEqual(len(result.raw), 0)
        for col in ("zk_CCS", "zk_OCS", "zk_NW", "used", "detector", "thx_CCS", "snr_extra"):
            self.assertIn(col, result.raw.colnames)
        self.assertEqual(result.raw["zk_CCS"].shape, (0, len(NOLL_INDICES)))
        self.assertIn("estimatorInfo", result.raw.meta)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: ModuleType) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
