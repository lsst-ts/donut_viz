"""Tests for AggregateAOSVisitTableCwfsTask.run with synthetic tables.

These tests guard against the raw-table fill crashing when the donut rows
cannot be aligned with the zernike rows for a detector pair - most notably
when a visit produced no used donut pairs at all, so aggregateZernikesRaw
contains only the per-detector NaN placeholder rows written by
AggregateZernikeTablesTask (previously a ValueError: "cannot assign 0 input
values to the 1 output values where the mask is true"). In that case the
donut values must be left as the NaN allocations, matching how the task
already skips detectors with no donuts at all.
"""

import io

import numpy as np
from astropy.table import Table

from lsst.donut.viz.aggregate_visit import AggregateAOSVisitTableCwfsTask
from lsst.utils.tests import TestCase

EXTRA_DETECTORS = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
INTRA_DETECTORS = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

AVG_KEYS = [
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
]

N_ZK = 21


def make_zernike_tables(rows_per_detector: int, used: bool) -> tuple[Table, Table]:
    """Make synthetic aggregateZernikesRaw/Avg tables.

    rows_per_detector=1 with used=False and NaN coefficients mimics the
    per-detector placeholder rows AggregateZernikeTablesTask writes when no
    donut pairs were used anywhere in the visit.
    """
    n_rows = rows_per_detector * len(EXTRA_DETECTORS)
    raw = Table()
    raw["zk_CCS"] = np.full((n_rows, N_ZK), np.nan if not used else 0.1, dtype=np.float32)
    raw["detector"] = np.repeat(EXTRA_DETECTORS, rows_per_detector)
    raw["used"] = used

    avg = Table()
    avg["zk_CCS"] = np.full((len(EXTRA_DETECTORS), N_ZK), np.nan if not used else 0.1, dtype=np.float32)
    avg["detector"] = EXTRA_DETECTORS
    return raw, avg


def make_donut_table(detectors: list[str], donuts_per_detector: int) -> Table:
    """Make a synthetic aggregateDonutTable with donuts on given detectors."""
    n_rows = donuts_per_detector * len(detectors)
    table = Table()
    table["detector"] = np.repeat(detectors, donuts_per_detector)
    for i, k in enumerate(AVG_KEYS):
        table[k] = np.arange(n_rows, dtype=float) + i
    table["donut_id"] = [f"donut{i}" for i in range(n_rows)]
    return table


def roundtrip_ecsv(table: Table) -> Table:
    """Write a table to ECSV and read it back, as the butler does."""
    buffer = io.StringIO()
    table.write(buffer, format="ascii.ecsv")
    return Table.read(buffer.getvalue(), format="ascii.ecsv")


class TestAggregateAOSVisitTableCwfs(TestCase):
    def setUp(self) -> None:
        self.task = AggregateAOSVisitTableCwfsTask(config=AggregateAOSVisitTableCwfsTask.ConfigClass())

    def testPlaceholderZernikesWithOneSidedDonuts(self) -> None:
        # The regression case: no donut pairs were used in the visit, so the
        # zernike tables contain one NaN placeholder row per detector, while
        # donuts exist on only one side of each pair (which is why no pairs
        # could be formed). The donut values must be left as NaN rather than
        # crashing on the row-count mismatch, and the output must still
        # survive the ECSV roundtrip the butler performs.
        azr, aza = make_zernike_tables(rows_per_detector=1, used=False)
        adt = make_donut_table(INTRA_DETECTORS, donuts_per_detector=3)

        struct = self.task.run(adt, azr, aza)

        self.assertEqual(len(struct.raw), len(EXTRA_DETECTORS))
        for k in AVG_KEYS:
            self.assertTrue(np.all(np.isnan(struct.raw[k])))
        # The output schema must not depend on the data: the per-side columns
        # must exist (as NaN / empty strings) even though every detector was
        # skipped, since downstream consumers (e.g. PlotCwfsPairingTask) index
        # them unconditionally.
        for k in AVG_KEYS:
            self.assertTrue(np.all(np.isnan(struct.raw[k + "_intra"])))
            self.assertTrue(np.all(np.isnan(struct.raw[k + "_extra"])))
        self.assertTrue(all(struct.raw["donut_id_intra"] == ""))
        self.assertTrue(all(struct.raw["donut_id_extra"] == ""))
        # The avg table is filled from the donut table alone, so it does get
        # real values where donuts exist.
        self.assertFalse(np.any(np.isnan(struct.avg["coord_ra"])))

        self.assertEqual(len(roundtrip_ecsv(struct.raw)), len(EXTRA_DETECTORS))
        self.assertEqual(len(roundtrip_ecsv(struct.avg)), len(EXTRA_DETECTORS))

    def testAlignedDonutPairsAreAveraged(self) -> None:
        # The normal case must be unaffected by the mismatch guard: donuts on
        # both sides of every pair, with one zernike row per donut pair.
        azr, aza = make_zernike_tables(rows_per_detector=2, used=True)
        adt = make_donut_table(EXTRA_DETECTORS + INTRA_DETECTORS, donuts_per_detector=2)

        struct = self.task.run(adt, azr, aza)

        self.assertEqual(len(struct.raw), 2 * len(EXTRA_DETECTORS))
        for k in AVG_KEYS:
            self.assertFalse(np.any(np.isnan(struct.raw[k])))
        # Filled values are the intra/extra average of the donut values, with
        # the per-side values preserved in the _intra/_extra columns.
        w = struct.raw["detector"] == "R00_SW0"
        wextra = adt["detector"] == "R00_SW0"
        wintra = adt["detector"] == "R00_SW1"
        expected = 0.5 * (adt["coord_ra"][wextra] + adt["coord_ra"][wintra])
        np.testing.assert_array_equal(struct.raw["coord_ra"][w], expected)
        np.testing.assert_array_equal(struct.raw["coord_ra_extra"][w], adt["coord_ra"][wextra])
        np.testing.assert_array_equal(struct.raw["coord_ra_intra"][w], adt["coord_ra"][wintra])
