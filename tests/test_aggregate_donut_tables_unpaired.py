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

"""Tests for AggregateDonutTablesUnpairedTask's empty-input guard.

The unpaired task fits each half-chip independently, so unlike the paired
CWFS task it has no extra/intra pairing to skip on; a detector is skipped
only when it has no quality table, or an empty one because no donut
survived selection. When that happens for *every* detector the loop that
binds ``table`` never runs, which previously raised an UnboundLocalError
at the ``convertDictToVisitInfo(table.meta[...])`` below the loop.

This is the unpaired counterpart of the guard tested in
test_aggregate_donut_tables_cwfs.py.
"""

import numpy as np
from astropy import units as u
from astropy.table import QTable

from lsst.donut.viz.aggregate_donut_tables_unpaired_task import (
    AggregateDonutTablesUnpairedTask,
    AggregateDonutTablesUnpairedTaskConfig,
)
from lsst.obs.lsst import LsstCam
from lsst.utils.tests import TestCase

# Two of the eight WAVEFRONT half-chips. Unlike the paired task these are
# fit independently, so either one alone is a complete unit of work.
EXTRA_DETECTOR_ID = 191
INTRA_DETECTOR_ID = 192


def make_visit_info() -> dict:
    """Mimic the visit_info dict addVisitInfoToCatTable stamps onto a
    donut table's metadata; convertDictToVisitInfo requires every key."""
    return {
        "boresight_ra": 10.0 * u.deg,
        "boresight_dec": -20.0 * u.deg,
        "boresight_alt": 60.0 * u.deg,
        "boresight_az": 100.0 * u.deg,
        "boresight_rot_angle": 0.0 * u.deg,
        "rot_type_value": 1,
        "focus_z": 0.0 * u.mm,
        "mjd": 60000.0,
        "visit_id": 2025111500226,
        "instrument_label": "LSSTCam",
        "observatory_elevation": 2663.0 * u.m,
        "observatory_latitude": -30.24 * u.deg,
        "observatory_longitude": -70.75 * u.deg,
        "ERA": 0.0 * u.deg,
        "exposure_time": 30.0 * u.s,
    }


class TestAggregateDonutTablesUnpairedRun(TestCase):
    def setUp(self) -> None:
        self.task = AggregateDonutTablesUnpairedTask(config=AggregateDonutTablesUnpairedTaskConfig())
        self.camera = LsstCam().getCamera()

    def testNoDonutTablesReturnsEmptyTable(self) -> None:
        # No donut tables at all for this visit: the loop body never runs.
        struct = self.task.run(self.camera, {}, {})

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        self.assertEqual(struct.aggregateDonutTable.meta, {})

    def testEmptyQualityTablesReturnsEmptyTable(self) -> None:
        # Every detector's quality table is empty (no donut survived
        # selection anywhere on this visit), so each iteration `continue`s
        # and `table` is never bound. This is the case observed in
        # production on RSO-1072.
        donutTables = {
            EXTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []}),
            INTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []}),
        }
        qualityTables = {
            EXTRA_DETECTOR_ID: QTable({"FINAL_SELECT": [], "SN": []}),
            INTRA_DETECTOR_ID: QTable({"FINAL_SELECT": [], "SN": []}),
        }

        struct = self.task.run(self.camera, donutTables, qualityTables)

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        self.assertEqual(struct.aggregateDonutTable.meta, {})

    def testMissingQualityTablesReturnsEmptyTable(self) -> None:
        # Donut tables present but no quality tables at all, so every
        # detector is skipped by the `detector not in qualityTables` check.
        donutTables = {EXTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []})}

        struct = self.task.run(self.camera, donutTables, {})

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        self.assertEqual(struct.aggregateDonutTable.meta, {})

    def testAllFinalSelectFalseStillAggregates(self) -> None:
        # Contrast case: the quality table is non-empty but every donut is
        # rejected (FINAL_SELECT all False). The loop still runs and
        # appends a zero-row table, so the guard must NOT fire here -- the
        # task should reach its metadata-stamping code as before.
        n = 2
        donutTables = {
            EXTRA_DETECTOR_ID: QTable({"centroid_x": np.zeros(n), "centroid_y": np.zeros(n)}),
        }
        donutTables[EXTRA_DETECTOR_ID].meta["visit_info"] = make_visit_info()
        qualityTables = {
            EXTRA_DETECTOR_ID: QTable({"FINAL_SELECT": [False] * n, "SN": np.zeros(n)}),
        }

        struct = self.task.run(self.camera, donutTables, qualityTables)

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        # Unlike the empty-input guard, this path reaches the
        # metadata-stamping code below the guard, so meta is populated.
        self.assertIn("visitInfo", struct.aggregateDonutTable.meta)
