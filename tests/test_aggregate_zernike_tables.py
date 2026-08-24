"""Tests for AggregateZernikeTablesTask.run with synthetic zernike tables.

These tests guard against the aggregate task persisting a zero-row
aggregateZernikesRaw table when no donut pairs were used on any detector
in the visit (every input table contains only its "average" row, as
produced by CalcZernikesTask.empty()). A zero-row table with fixed-shape
vector columns writes to ECSV successfully but cannot be read back
(astropy fails with "shape mismatch between value and column specifier"),
which poisons the butler repo for every downstream consumer. Following
the convention of CalcZernikesTask.empty(), the task must instead write
NaN placeholder rows marked unused, which round-trip through ECSV and
which downstream consumers already filter out via the "used" column.
"""

import io

import numpy as np
from astropy import units as u
from astropy.table import QTable, Table

from lsst.donut.viz.aggregate_visit import AggregateZernikeTablesTask
from lsst.pipe.base import NoWorkFound
from lsst.utils.tests import TestCase

NOLL_INDICES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]


def make_zk_table(det_name: str, n_pairs: int) -> QTable:
    """Mimic the zernikes table from CalcZernikesTask.

    The table has one "average" row plus n_pairs "pairN" rows; n_pairs=0
    mimics the output of CalcZernikesTask.empty() for a detector with no
    usable donuts.
    """
    table = QTable()
    n_rows = 1 + n_pairs
    table["label"] = ["average"] + [f"pair{i + 1}" for i in range(n_pairs)]
    table["used"] = [False] + [True] * n_pairs
    table["intra_donut_id"] = [""] * n_rows
    table["extra_donut_id"] = [""] * n_rows
    rng = np.random.default_rng(12345)
    for j in NOLL_INDICES:
        for suffix in ("", "_intrinsic", "_deviation"):
            if n_pairs == 0:
                vals = np.array([np.nan], dtype=np.float32)
            else:
                vals = rng.normal(size=n_rows).astype(np.float32)
            table[f"Z{j}{suffix}"] = vals * u.nm

    det_meta = {
        "det_name": det_name,
        "visit": 2025111500226,
        "dfc_dist": 1.5,
        "band": "i",
        "boresight_rot_angle_rad": 0.1,
        "boresight_par_angle_rad": 0.2,
        "boresight_alt_rad": 1.0,
        "boresight_az_rad": 2.0,
        "boresight_ra_rad": 1.1,
        "boresight_dec_rad": -0.5,
        "mjd": 60000.0,
    }
    table.meta = {
        "intra": dict(det_meta),
        "extra": dict(det_meta),
        "cam_name": "LSSTCam",
        "noll_indices": list(NOLL_INDICES),
        "opd_columns": [f"Z{j}" for j in NOLL_INDICES],
        "intrinsic_columns": [f"Z{j}_intrinsic" for j in NOLL_INDICES],
        "deviation_columns": [f"Z{j}_deviation" for j in NOLL_INDICES],
        "estimatorInfo": {"fwhm": [0.7] * n_pairs},
    }
    return table


def roundtrip_ecsv(table: Table) -> Table:
    """Write a table to ECSV and read it back, as the butler does."""
    buffer = io.StringIO()
    table.write(buffer, format="ascii.ecsv")
    return Table.read(buffer.getvalue(), format="ascii.ecsv")


class TestAggregateZernikeTables(TestCase):
    def setUp(self) -> None:
        self.task = AggregateZernikeTablesTask(config=AggregateZernikeTablesTask.ConfigClass())

    def testAllTablesEmptyWritesUnusedPlaceholders(self) -> None:
        # The regression case: every detector produced an empty()-style
        # table (average row only), so the stacked raw table would have
        # zero rows and be unreadable once written to ECSV. The task must
        # instead write one NaN placeholder row per detector, marked
        # unused, and both outputs must survive an ECSV roundtrip.
        tables = [make_zk_table(f"R{i}{i}_SW0", n_pairs=0) for i in range(4)]
        struct = self.task.run(tables)

        self.assertEqual(len(struct.raw), 4)
        self.assertFalse(any(struct.raw["used"]))
        self.assertTrue(np.all(np.isnan(struct.raw["zk_CCS"])))

        self.assertEqual(len(roundtrip_ecsv(struct.raw)), 4)
        self.assertEqual(len(roundtrip_ecsv(struct.avg)), 4)

    def testNoTablesRaisesNoWorkFound(self) -> None:
        # Zero-length tables are skipped entirely, leaving nothing to
        # build even a placeholder from (the output metadata comes from
        # the input tables); previously this crashed in vstack rather
        # than declaring NoWorkFound.
        with self.assertRaises(NoWorkFound):
            self.task.run([])

    def testPartiallyEmptyTablesRoundtrip(self) -> None:
        # One detector with no pairs must not break the aggregate or
        # trigger the placeholder path, and the raw output must survive
        # an ECSV roundtrip (the on-disk format used by the butler for
        # these tables).
        tables = [make_zk_table(f"R{i}{i}_SW0", n_pairs=3) for i in range(3)]
        tables.append(make_zk_table("R44_SW1", n_pairs=0))
        struct = self.task.run(tables)
        self.assertEqual(len(struct.raw), 9)
        self.assertTrue(all(struct.raw["used"]))
        self.assertEqual(len(struct.avg), 4)

        self.assertEqual(len(roundtrip_ecsv(struct.raw)), 9)
        self.assertEqual(len(roundtrip_ecsv(struct.avg)), 4)
