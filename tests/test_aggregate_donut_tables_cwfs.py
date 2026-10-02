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

"""Tests for AggregateDonutTablesCwfsTask's empty-input guards.

These guard against two distinct ways a visit can have nothing to
aggregate: no donut tables at all (a fast path in runQuantum that skips
calling run() entirely - functionally redundant with the second guard
below, since run() on an empty donutTables dict falls straight through to
it, but kept as it documents the no-refs-at-all case explicitly), and no
donuts surviving quality selection on any detector (e.g. every extra-focal
detector's quality table is empty), which previously hit an
UnboundLocalError since the loop that binds ``table`` never ran.
"""

import types
from typing import Any
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.table import QTable

from lsst.donut.viz.aggregate_donut_tables_cwfs_task import (
    AggregateDonutTablesCwfsTask,
    AggregateDonutTablesCwfsTaskConfig,
)
from lsst.obs.lsst import LsstCam
from lsst.utils.tests import TestCase

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


class FakeQuantumContext:
    """Minimal stand-in for `lsst.pipe.base.QuantumContext`.

    Maps opaque "refs" (any hashable placeholder) to pre-set values for
    `get`, and records `put` calls for later inspection - enough to
    exercise `runQuantum`'s empty-input short-circuit without a real
    butler or quantum graph.
    """

    def __init__(self) -> None:
        self._values: dict[Any, Any] = {}
        self.puts: dict[Any, Any] = {}

    def set_get(self, ref: Any, value: Any) -> None:
        self._values[ref] = value

    def get(self, ref: Any) -> Any:
        return self._values[ref]

    def put(self, obj: Any, ref: Any) -> None:
        self.puts[ref] = obj


class TestAggregateDonutTablesCwfsRunQuantum(TestCase):
    def setUp(self) -> None:
        self.task = AggregateDonutTablesCwfsTask(config=AggregateDonutTablesCwfsTaskConfig())

    def testNoDonutTablesWritesEmptyOutput(self) -> None:
        # No donutTables refs at all for this visit: the dict built in
        # runQuantum is empty, so the fast path must short-circuit before
        # ever calling run().
        butlerQC = FakeQuantumContext()
        butlerQC.set_get("camera_ref", object())
        inputRefs = types.SimpleNamespace(donutTables=[], qualityTables=[], camera="camera_ref")
        outputRefs = types.SimpleNamespace(aggregateDonutTable="out_ref")

        with patch.object(self.task, "run") as mock_run:
            self.task.runQuantum(butlerQC, inputRefs, outputRefs)  # type: ignore[arg-type]
        mock_run.assert_not_called()

        result = butlerQC.puts["out_ref"]
        self.assertEqual(len(result), 0)
        self.assertEqual(result.meta, {})


class TestAggregateDonutTablesCwfsRun(TestCase):
    def setUp(self) -> None:
        self.task = AggregateDonutTablesCwfsTask(config=AggregateDonutTablesCwfsTaskConfig())
        self.camera = LsstCam().getCamera()

    def testNoDonutsSurviveQualitySelectionReturnsEmptyTable(self) -> None:
        # A visit where the (single) extra-focal detector's quality table
        # is empty: the extra-detector loop `continue`s immediately, so
        # `table` is never bound and there is nothing to aggregate.
        donutTables = {
            EXTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []}),
            INTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []}),
        }
        qualityTables = {
            EXTRA_DETECTOR_ID: QTable({"DEFOCAL_TYPE": [], "FINAL_SELECT": [], "SN": []}),
        }

        struct = self.task.run(self.camera, donutTables, qualityTables)

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        self.assertEqual(struct.aggregateDonutTable.meta, {})

    def testMissingIntraDetectorReturnsEmptyTable(self) -> None:
        # Incomplete corner ingestion: only the extra-focal detector's
        # donut table is present, so the pair is skipped and, with no
        # other detectors, nothing is aggregated.
        donutTables = {EXTRA_DETECTOR_ID: QTable({"centroid_x": [], "centroid_y": []})}
        qualityTables = {
            EXTRA_DETECTOR_ID: QTable({"DEFOCAL_TYPE": ["extra"], "FINAL_SELECT": [True], "SN": [10.0]})
        }

        struct = self.task.run(self.camera, donutTables, qualityTables)

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        self.assertEqual(struct.aggregateDonutTable.meta, {})

    def testAllFinalSelectFalseStillAppendsEmptyRows(self) -> None:
        # Contrast case: quality tables are non-empty but every donut is
        # rejected (FINAL_SELECT all False). Unlike the empty-quality-table
        # case, the inner loop still runs and appends a zero-row table per
        # side, so the empty-tables guard does NOT fire here; this must not
        # regress into a crash either.
        n = 2
        donutTables = {
            EXTRA_DETECTOR_ID: QTable({"centroid_x": np.zeros(n), "centroid_y": np.zeros(n)}),
            INTRA_DETECTOR_ID: QTable({"centroid_x": np.zeros(n), "centroid_y": np.zeros(n)}),
        }
        visit_info = make_visit_info()
        for table in donutTables.values():
            table.meta["visit_info"] = visit_info
        qualityTables = {
            EXTRA_DETECTOR_ID: QTable(
                {
                    "DEFOCAL_TYPE": ["extra"] * n + ["intra"] * n,
                    "FINAL_SELECT": [False] * (2 * n),
                    "SN": np.zeros(2 * n),
                }
            )
        }

        struct = self.task.run(self.camera, donutTables, qualityTables)

        self.assertEqual(len(struct.aggregateDonutTable), 0)
        # Unlike the fully-empty-input guard, this path does reach the
        # metadata-stamping code below the guard, so meta is populated.
        self.assertIn("visitInfo", struct.aggregateDonutTable.meta)
