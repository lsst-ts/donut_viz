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

import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod

from .aggregate_aos_visit_table import (
    AggregateAOSVisitTableTask,
    AggregateAOSVisitTableTaskConfig,
)

__all__ = [
    "AggregateAOSVisitTableUnpairedTask",
]


class AggregateAOSVisitTableUnpairedTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableUnpaired"

    @timeMethod
    def run(self, adt: Table, azr: Table, aza: Table) -> pipeBase.Struct:
        """
        Create overall summary table for the visit.

        Parameters
        ----------
        adt : `astropy.table.Table`
            Aggregated donut table.
        azr : `astropy.table.Table`
            Aggregated raw Zernike table.
        aza : `astropy.table.Table`
            Aggregated average Zernike table.

        Returns
        -------
        struct : `pipeBase.Struct`
            Struct with `avg` and `raw` tables:
            avg_table : `astropy.table.Table`
                Table with average donut and Zernike values by detector.
            raw_table : `astropy.table.Table`
                Table with donut and Zernike values from every
                source that went into the averages.
        """
        dets = np.unique(adt["detector"])
        # Only take extra focal detector names
        avg_table = aza.copy()
        avg_keys = [
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
        for k in avg_keys:
            # Add keys into table. That way even if no donuts
            # are present, the column will exist.
            avg_table[k] = np.nan

        # Process average table
        for det in dets:
            w = avg_table["detector"] == det
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][adt["detector"] == det])

        # Process raw table. Allocate all columns up front so that the output
        # schema does not depend on which detectors had donuts matching their
        # zernike rows.
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan
        if "donut_id" in adt.colnames:  # safeguard against older data
            raw_table["donut_id"] = np.full(len(raw_table), "", dtype=adt["donut_id"].dtype)
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            # Check if there are any matching rows
            if not np.any(wadt):
                continue
            if wadt.sum() != w.sum():
                # The donut rows don't align with the zernike rows for this
                # detector (e.g. the zernike table contains only NaN
                # placeholder rows because no donuts were used), so leave the
                # NaN values allocated above.
                self._warnOnMismatch(int(wadt.sum()), int(w.sum()), str(det))
                continue

            for k in avg_keys:
                # ought to be the same length now
                raw_table[k][w] = adt[k][wadt]
            if "donut_id" in adt.colnames:  # safeguard against older data
                raw_table["donut_id"][w] = adt["donut_id"][wadt]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)
