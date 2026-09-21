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
    "AggregateAOSVisitTableCwfsTask",
]


class AggregateAOSVisitTableCwfsTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableCwfs"

    @timeMethod
    def run(self, adt: Table, azr: Table, aza: Table) -> pipeBase.Struct:
        """
        Create overall summary tables for the visit.

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
        extraDetectorNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        intraDetectorNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]
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
            avg_table[k] = np.nan  # Allocate

        # Process average table
        for det_extra, det_intra in zip(extraDetectorNames, intraDetectorNames):
            w = avg_table["detector"] == det_extra
            wextra = adt["detector"] == det_extra
            wintra = adt["detector"] == det_intra
            # Combine extra and intra detector masks
            wadt = np.logical_or(wextra, wintra)
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][wadt])

        # Process raw table. Allocate all columns up front so that the output
        # schema does not depend on which detectors had donuts matching their
        # zernike rows.
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
            raw_table[k + "_intra"] = np.nan
            raw_table[k + "_extra"] = np.nan
        if "donut_id" in adt.colnames:  # safeguard against older data
            raw_table["donut_id_intra"] = np.full(len(raw_table), "", dtype=adt["donut_id"].dtype)
            raw_table["donut_id_extra"] = np.full(len(raw_table), "", dtype=adt["donut_id"].dtype)
        for det_extra, det_intra in zip(extraDetectorNames, intraDetectorNames):
            w = raw_table["detector"] == det_extra
            wextra = adt["detector"] == det_extra
            wintra = adt["detector"] == det_intra
            # Combine extra and intra detector masks
            wadt = np.logical_or(wextra, wintra)
            # Check if there are any matching rows
            if not np.any(wadt):
                continue
            if min(wintra.sum(), wextra.sum()) != w.sum():
                # No aligned donut pairs for this detector pair (e.g. donuts
                # on only one side, or the zernike table contains only NaN
                # placeholder rows because no pairs were used), so leave the
                # NaN values allocated above.
                self._warnOnMismatch(int(min(wintra.sum(), wextra.sum())), int(w.sum()), det_extra)
                continue

            for k in avg_keys:
                # If one table has more rows than the other,
                # trim the longer one
                if wintra.sum() > wextra.sum():
                    wintra[wintra] = [True] * wextra.sum() + [False] * (wintra.sum() - wextra.sum())
                elif wextra.sum() > wintra.sum():
                    wextra[wextra] = [True] * wintra.sum() + [False] * (wextra.sum() - wintra.sum())
                # ought to be the same length now
                raw_table[k][w] = 0.5 * (adt[k][wintra] + adt[k][wextra])
                raw_table[k + "_intra"][w] = adt[k][wintra]
                raw_table[k + "_extra"][w] = adt[k][wextra]
            # donut id can't be averaged like coordinates or centroids,
            # so we process it separately
            k = "donut_id"
            if k in adt.colnames:  # safeguard against older data
                raw_table[k + "_intra"][w] = adt[k][wintra]
                raw_table[k + "_extra"][w] = adt[k][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)
