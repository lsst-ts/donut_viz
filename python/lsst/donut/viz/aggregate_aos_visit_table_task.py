import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
    "AggregateAOSVisitTableCwfsTask",
    "AggregateAOSVisitTableUnpairedCwfsTask",
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
        # Create the final table
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            if single_sided:  # single-sided Zernike estimates
                for k in avg_keys:
                    raw_table[k][w] = adt[k][wadt]
            else:  # double-sided Zernike estimates
                wintra = adt[wadt]["focusZ"] == visit_fzmin
                wextra = adt[wadt]["focusZ"] == visit_fzmax
                for k in avg_keys:
                    # If one table has more rows than the other,
                    # trim the longer one
                    if wintra.sum() > wextra.sum():
                        wintra[wintra] = [True] * wextra.sum() + [False] * (wintra.sum() - wextra.sum())
                    elif wextra.sum() > wintra.sum():
                        wextra[wextra] = [True] * wintra.sum() + [False] * (wextra.sum() - wintra.sum())
                    # ought to be the same length now
                    raw_table[k][w] = 0.5 * (adt[k][wadt][wintra] + adt[k][wadt][wextra])
                    if k + "_intra" not in raw_table.colnames:
                        raw_table[k + "_intra"] = np.nan
                        raw_table[k + "_extra"] = np.nan
                    raw_table[k + "_intra"][w] = adt[k][wadt][wintra]
                    raw_table[k + "_extra"][w] = adt[k][wadt][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)


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

        # Process raw table
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det_extra, det_intra in zip(extraDetectorNames, intraDetectorNames):
            w = raw_table["detector"] == det_extra
            wextra = adt["detector"] == det_extra
            wintra = adt["detector"] == det_intra
            # Combine extra and intra detector masks
            wadt = np.logical_or(wextra, wintra)
            # Check if there are any matching rows
            if not np.any(wadt):
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
                if k + "_intra" not in raw_table.colnames:
                    raw_table[k + "_intra"] = np.nan
                    raw_table[k + "_extra"] = np.nan
                raw_table[k + "_intra"][w] = adt[k][wintra]
                raw_table[k + "_extra"][w] = adt[k][wextra]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)


class AggregateAOSVisitTableUnpairedCwfsTask(AggregateAOSVisitTableTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTableUnpairedCwfs"

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

        # Process raw table
        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan
        for det in dets:
            w = raw_table["detector"] == det
            wadt = adt["detector"] == det
            # Check if there are any matching rows
            if not np.any(wadt):
                continue

            for k in avg_keys:
                # ought to be the same length now
                raw_table[k][w] = adt[k][wadt]

        return pipeBase.Struct(raw=raw_table, avg=avg_table)
