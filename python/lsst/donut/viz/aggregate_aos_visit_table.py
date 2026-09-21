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
from lsst.pipe.base import connectionTypes as ct
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
]


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=(
        "visit",
        "instrument",
    ),  # type: ignore
):
    aggregateDonutTable = ct.Input(
        doc="Visit-level table of donuts",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
        deferGraphConstraint=True,
    )
    aggregateZernikesRaw = ct.Input(
        doc="Visit-level table of raw Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
        deferGraphConstraint=True,
    )
    aggregateZernikesAvg = ct.Input(
        doc="Visit-level table of average Zernikes by detector",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
        deferGraphConstraint=True,
    )
    aggregateAOSRaw = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    aggregateAOSAvg = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
    )


class AggregateAOSVisitTableTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateAOSVisitTableTaskConnections,  # type: ignore
):
    pass


class AggregateAOSVisitTableTask(pipeBase.PipelineTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTable"

    def _warnOnMismatch(self, n_donuts: int, n_zernikes: int, det: str) -> None:
        """Warn when donut rows cannot be matched to the zernike rows.

        A mismatch with donuts present means the donut values in the raw
        table are being left as NaN for this detector; no warning is issued
        when there are no donuts to match at all.
        """
        if n_donuts > 0:
            self.log.warning(
                "Number of donut rows (%d) does not match the number of zernike rows (%d) "
                "for detector %s; leaving NaN values in the raw table.",
                n_donuts,
                n_zernikes,
                det,
            )

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        adt = butlerQC.get(inputRefs.aggregateDonutTable)
        azr = butlerQC.get(inputRefs.aggregateZernikesRaw)
        aza = butlerQC.get(inputRefs.aggregateZernikesAvg)

        tables = self.run(adt, azr, aza)

        butlerQC.put(tables.avg, outputRefs.aggregateAOSAvg)
        butlerQC.put(tables.raw, outputRefs.aggregateAOSRaw)

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
        dets = np.unique(adt["detector"])
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

        for det in dets:
            w = avg_table["detector"] == det
            for k in avg_keys:
                avg_table[k][w] = np.mean(adt[k][adt["detector"] == det])

        raw_table = azr.copy()
        # Use focusZ to learn about contents of the table
        single_sided = False
        adt["focusZ"] = np.round(adt["focusZ"], decimals=4)
        visit_fzmin = adt["focusZ"].min()
        visit_fzmax = adt["focusZ"].max()
        # If entire donut table has a single focusZ value, then
        # we can assume single-sided Zernike estimates
        if visit_fzmin == visit_fzmax:
            single_sided = True
        # Create the final table. Allocate all columns up front so that the
        # output schema does not depend on which detectors had donuts
        # matching their zernike rows.
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        if not single_sided:
            for k in avg_keys:
                raw_table[k + "_intra"] = np.nan
                raw_table[k + "_extra"] = np.nan
            if "donut_id" in adt.colnames:  # safeguard against older data
                raw_table["donut_id_intra"] = np.full(len(raw_table), "", dtype=adt["donut_id"].dtype)
                raw_table["donut_id_extra"] = np.full(len(raw_table), "", dtype=adt["donut_id"].dtype)
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            if single_sided:  # single-sided Zernike estimates
                if wadt.sum() != w.sum():
                    # The donut rows don't align with the zernike rows for
                    # this detector (e.g. the zernike table contains only NaN
                    # placeholder rows because no donuts were used), so leave
                    # the NaN values allocated above.
                    self._warnOnMismatch(int(wadt.sum()), int(w.sum()), str(det))
                    continue
                for k in avg_keys:
                    raw_table[k][w] = adt[k][wadt]
                raw_table["donut_id"][w] = adt["donut_id"][wadt]
            else:  # double-sided Zernike estimates
                wintra = adt[wadt]["focusZ"] == visit_fzmin
                wextra = adt[wadt]["focusZ"] == visit_fzmax
                if min(wintra.sum(), wextra.sum()) != w.sum():
                    # No aligned donut pairs for this detector (e.g. donuts on
                    # only one side, or the zernike table contains only NaN
                    # placeholder rows because no pairs were used), so leave
                    # the NaN values allocated above.
                    self._warnOnMismatch(int(min(wintra.sum(), wextra.sum())), int(w.sum()), str(det))
                    continue
                for k in avg_keys:
                    # If one table has more rows than the other,
                    # trim the longer one
                    if wintra.sum() > wextra.sum():
                        wintra[wintra] = [True] * wextra.sum() + [False] * (wintra.sum() - wextra.sum())
                    elif wextra.sum() > wintra.sum():
                        wextra[wextra] = [True] * wintra.sum() + [False] * (wextra.sum() - wintra.sum())
                    # ought to be the same length now
                    raw_table[k][w] = 0.5 * (adt[k][wadt][wintra] + adt[k][wadt][wextra])
                    raw_table[k + "_intra"][w] = adt[k][wadt][wintra]
                    raw_table[k + "_extra"][w] = adt[k][wadt][wextra]
                # donut id can't be averaged like coordinates or centroids,
                # so we process it separately
                k = "donut_id"
                if k in adt.colnames:  # safeguard against older data
                    raw_table[k + "_intra"][w] = adt[k][wadt][wintra]
                    raw_table[k + "_extra"][w] = adt[k][wadt][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)
