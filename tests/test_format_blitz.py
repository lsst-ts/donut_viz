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

import galsim
import numpy as np
from astropy.table import Table

import lsst.utils.tests
from lsst.afw.coord import Observatory
from lsst.afw.image import VisitInfo
from lsst.daf.base import DateTime
from lsst.donut.viz.format_blitz import FormatBlitzTask, FormatBlitzTaskConfig
from lsst.geom import SpherePoint, degrees
from lsst.obs.lsst import LsstCam

# DonutBlitzMonolith default fitted Noll indices.
NOLL_INDICES = list(range(4, 20)) + list(range(22, 27))

# (extra det id, intra det id, extra sensor, intra sensor) for two corners.
CORNERS = [
    (191, 192, "R00_SW0", "R00_SW1"),
    (195, 196, "R04_SW0", "R04_SW1"),
]


def _makeVisitInfo():
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


def _makeDonutRow(group, defocal, detId, sensor, sourceId, seed):
    """One per-donut blitz catalog row as a dict."""
    rng = np.random.default_rng(seed)
    row = {
        "visit_id": 9999,
        "det_id": detId,
        "sensor": sensor,
        "source_id": sourceId,
        "defocal": defocal,
        "band": "r",
        "accepted": True,
        "group": group,
        "group_size": 2,
        "fit_success": True,
        "fa_x_ccs": float(rng.uniform(-0.03, 0.03)),
        "fa_y_ccs": float(rng.uniform(-0.03, 0.03)),
        "centroid_x_raw": float(rng.uniform(0, 4000)),
        "centroid_y_raw": float(rng.uniform(0, 4000)),
        "snr": float(rng.uniform(100, 2000)),
        "fit_fwhm": 1.1,
        "fit_dx": 0.2,
        "fit_dy": -0.1,
        "fit_cost": 12.3,
        "fit_flux": 1e6,
        "fit_bkg": 5.0,
    }
    # Deviation must match across the pair; key it on the group so extra/intra
    # share it. Intrinsic differs per donut.
    devRng = np.random.default_rng(1000 + group)
    for j in NOLL_INDICES:
        row[f"Z{j}_dev"] = float(devRng.uniform(-0.5, 0.5))
        row[f"Z{j}_intrinsic"] = float(rng.uniform(-0.2, 0.2))
    return row


def _makeCatalog():
    rows = []
    sid = 0
    for grp, (extraDet, intraDet, extraName, intraName) in enumerate(CORNERS):
        rows.append(_makeDonutRow(grp, "extra", extraDet, extraName, sid, seed=grp * 10 + 1))
        rows.append(_makeDonutRow(grp, "intra", intraDet, intraName, sid + 1, seed=grp * 10 + 2))
        sid += 2
    catalog = Table(rows)
    catalog.meta["noll_indices"] = list(NOLL_INDICES)
    return catalog


class TestFormatBlitzTask(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = LsstCam().getCamera()
        cls.visitInfo = _makeVisitInfo()

    def setUp(self):
        self.task = FormatBlitzTask(config=FormatBlitzTaskConfig())
        self.catalog = _makeCatalog()

    def testRowCountAndColumns(self):
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

    def testZernikeRelationship(self):
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        # CCS total = intrinsic + deviation.
        np.testing.assert_allclose(raw["zk_CCS"], raw["zk_intrinsic_CCS"] + raw["zk_deviation_CCS"])
        # Deviation shared across the pair -> equals the input Z{j}_dev.
        for grp in range(len(CORNERS)):
            devRng = np.random.default_rng(1000 + grp)
            expected = np.array([devRng.uniform(-0.5, 0.5) for _ in NOLL_INDICES])
            np.testing.assert_allclose(raw["zk_deviation_CCS"][grp], expected)

    def testFrameTransforms(self):
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

    def testGeometryAveraging(self):
        raw = self.task.run(self.catalog, self.visitInfo, self.camera).raw
        for base in ("centroid_x", "thx_CCS", "snr"):
            np.testing.assert_allclose(raw[base], 0.5 * (raw[base + "_intra"] + raw[base + "_extra"]))
        # No per-donut sky coords in the blitz catalog.
        self.assertTrue(np.all(np.isnan(raw["coord_ra"])))

    def testMetadata(self):
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

    def testAvgTable(self):
        result = self.task.run(self.catalog, self.visitInfo, self.camera)
        avg = result.avg
        self.assertEqual(len(avg), len(CORNERS))
        self.assertEqual(avg["zk_CCS"].shape, (len(CORNERS), len(NOLL_INDICES)))
        # One pair per detector -> avg equals the single raw row.
        for det in avg["detector"]:
            aRow = avg[avg["detector"] == det]
            rRow = result.raw[np.array([str(d) for d in result.raw["detector"]]) == det]
            np.testing.assert_allclose(aRow["zk_CCS"][0], rRow["zk_CCS"][0])

    def testEmptyCatalog(self):
        empty = Table()
        empty.meta["noll_indices"] = list(NOLL_INDICES)
        result = self.task.run(empty, self.visitInfo, self.camera)
        self.assertEqual(len(result.raw), 0)
        self.assertEqual(len(result.avg), 0)

    def testUnmatchedGroupSkipped(self):
        # Drop the intra donut of the first corner; that group should vanish.
        catalog = self.catalog[[i for i in range(len(self.catalog)) if i != 1]]
        catalog.meta["noll_indices"] = list(NOLL_INDICES)
        raw = self.task.run(catalog, self.visitInfo, self.camera).raw
        self.assertEqual(len(raw), len(CORNERS) - 1)
        self.assertEqual(list(raw["detector"]), ["R04_SW0"])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
