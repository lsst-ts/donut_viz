import galsim
import numpy as np
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from astropy.table import Table, vstack
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.geom import Point2D, radians
from lsst.ts.wep.task.donutStamps import DonutStamps


__all__ = [
    "AggregateZernikesTaskConnections",
    "AggregateZernikesTaskConfig",
    "AggregateZernikesTask",
    "AggregateDonutCatalogsTaskConnections",
    "AggregateDonutCatalogsTaskConfig",
    "AggregateDonutCatalogsTask",
    "AggregateAOSVisitTableTaskConnections",
    "AggregateAOSVisitTableTaskConfig",
    "AggregateAOSVisitTableTask",
    "AggregateDonutStampsTaskConnections",
    "AggregateDonutStampsTaskConfig",
    "AggregateDonutStampsTask",
]


class AggregateZernikesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    visitInfos = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="VisitInfo",
        name="raw.visitInfo",
        multiple=True,
    )
    zernikesRaw = ct.Input(
        doc="Zernike Coefficients from all donuts",
        dimensions=(
            "visit",
            "detector",
            "instrument"
        ),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
        multiple=True,
    )
    zernikesAvg = ct.Input(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=(
            "visit",
            "detector",
            "instrument"
        ),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
        multiple=True,
    )
    aggregateZernikesRaw = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
        multiple=True
    )
    aggregateZernikesAvg = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
        multiple=True
    )


class AggregateZernikesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateZernikesTaskConnections,
):
    pass


class AggregateZernikesTask(pipeBase.PipelineTask):
    ConfigClass = AggregateZernikesTaskConfig
    _DefaultName = "AggregateZernikes"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ):
        raw_tables = []
        avg_tables = []
        visit = None
        for zernikeRawRef, zernikeAvgRef in zip(inputRefs.zernikesRaw, inputRefs.zernikesAvg):
            if visit is None:
                visit = zernikeRawRef.dataId['visit']
            if visit != zernikeRawRef.dataId['visit']:
                raise ValueError(f"Expected all zernikeRaw dataIds to have the same visit, got {visit} and {zernikeRawRef.dataId['visit']}")
            zernikeRaw = butlerQC.get(zernikeRawRef)
            zernikeAvg = butlerQC.get(zernikeAvgRef)
            raw_table = Table()
            raw_table['zk_CCS'] = zernikeRaw
            raw_table['detector'] = zernikeRawRef.dataId['detector']
            raw_tables.append(raw_table)
            avg_table = Table()
            avg_table['zk_CCS'] = zernikeAvg.reshape(1, -1)
            avg_table['detector'] = zernikeAvgRef.dataId['detector']
            avg_tables.append(avg_table)

        out_raw = vstack(raw_tables)
        out_avg = vstack(avg_tables)

        for visitInfoRef in inputRefs.visitInfos:
            if visitInfoRef.dataId['exposure'] == visit:
                visitInfo = butlerQC.get(visitInfoRef)
                break
        else:
            raise ValueError(f"Expected to find a visitInfo with exposure {visit}")

        meta = {}
        meta['visit'] = visit
        meta['parallacticAngle'] = visitInfo.boresightParAngle.asRadians()
        meta['rotAngle'] = visitInfo.boresightRotAngle.asRadians()
        meta['rotTelPos'] = visitInfo.boresightParAngle.asRadians() - visitInfo.boresightRotAngle.asRadians() - np.pi / 2
        meta['ra'] = visitInfo.boresightRaDec.getRa().asRadians()
        meta['dec'] = visitInfo.boresightRaDec.getDec().asRadians()
        meta['az'] = visitInfo.boresightAzAlt.getLongitude().asRadians()
        meta['alt'] = visitInfo.boresightAzAlt.getLatitude().asRadians()
        meta['mjd'] = visitInfo.date.toAstropy().mjd

        q = meta['parallacticAngle']
        rtp = meta['rotTelPos']

        jmax = out_raw['zk_CCS'].shape[1] + 3
        rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, rtp)[4:,4:]
        rot_NE = galsim.zernike.zernikeRotMatrix(jmax, q)[4:,4:]
        for cat in (out_raw, out_avg):
            cat.meta = meta
            cat['zk_OCS'] = cat['zk_CCS'] @ rot_OCS
            cat['zk_NE'] = cat['zk_CCS'] @ rot_NE

        butlerQC.put(out_raw, outputRefs.aggregateZernikesRaw[0])
        butlerQC.put(out_avg, outputRefs.aggregateZernikesAvg[0])


class AggregateDonutCatalogsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    visitInfos = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="VisitInfo",
        name="raw.visitInfo",
        multiple=True,
    )
    donutCatalogs = ct.Input(
        doc="Donut catalogs",
        dimensions=(
            "visit",
            "detector",
            "instrument"
        ),
        storageClass="DataFrame",
        name="donutCatalog",
        multiple=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    aggregateDonutCatalog = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateDonutCatalog",
        multiple=True
    )


class AggregateDonutCatalogsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutCatalogsTaskConnections,
):
    pass


class AggregateDonutCatalogsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutCatalogsTaskConfig
    _DefaultName = "AggregateDonutCatalogs"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ):
        camera = butlerQC.get(inputRefs.camera)

        # Only need one visitInfo per exposure.
        visitInfos = {}
        focusZs = {}
        for visitInfoRef in inputRefs.visitInfos:
            dataId = visitInfoRef.dataId
            if dataId['exposure'] not in visitInfos:
                visitInfo = butlerQC.get(visitInfoRef)
                visitInfos[dataId['exposure']] = visitInfo
                focusZs[dataId['exposure']] = visitInfo.focusZ

        if len(focusZs) == 1:
            extraIdx = focusZs.keys()[0]
            intraIdx = None
        elif len(focusZs) == 2:
            largestFocus = max(focusZs.values())
            extraIdx = [idx for idx, value in focusZs.items() if value == largestFocus][0]
            intraIdx = [idx for idx, value in focusZs.items() if value != largestFocus][0]
        else:
            raise ValueError(f"Expected 1 or 2 exposures, got {len(focusZs)}")

        tables = []
        for donutCatalogRef in inputRefs.donutCatalogs:
            dataId = donutCatalogRef.dataId
            det = camera[dataId['detector']]
            tform = det.getTransform(PIXELS, FIELD_ANGLE)

            visitInfo = visitInfos[dataId['visit']]
            donutCatalog = butlerQC.get(donutCatalogRef)
            table = Table.from_pandas(donutCatalog)
            table['detector'] = dataId['detector']
            table['focusZ'] = focusZs[dataId['visit']]
            pts = tform.applyForward(
                [Point2D(x, y) for x, y in zip(table['centroid_x'], table['centroid_y'])]
            )
            table['thx_CCS'] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
            table['thy_CCS'] = [pt.x for pt in pts]

            tables.append(table)
        out = vstack(tables)

        out.meta['extra'] = {
            'visit': extraIdx,
            'focusZ': focusZs[extraIdx],
            'parallacticAngle': visitInfos[extraIdx].boresightParAngle.asRadians(),
            'rotAngle': visitInfos[extraIdx].boresightRotAngle.asRadians(),
            'rotTelPos': visitInfos[extraIdx].boresightParAngle.asRadians() - visitInfos[extraIdx].boresightRotAngle.asRadians() - np.pi / 2,
            'ra': visitInfos[extraIdx].boresightRaDec.getRa().asRadians(),
            'dec': visitInfos[extraIdx].boresightRaDec.getDec().asRadians(),
            'az': visitInfos[extraIdx].boresightAzAlt.getLongitude().asRadians(),
            'alt': visitInfos[extraIdx].boresightAzAlt.getLatitude().asRadians(),
            'mjd': visitInfos[extraIdx].date.toAstropy().mjd
        }
        if intraIdx is not None:
            out.meta['intra'] = {
                'visit': intraIdx,
                'focusZ': focusZs[intraIdx],
                'parallacticAngle': visitInfos[intraIdx].boresightParAngle.asRadians(),
                'rotAngle': visitInfos[intraIdx].boresightRotAngle.asRadians(),
                'rotTelPos': visitInfos[intraIdx].boresightParAngle.asRadians() - visitInfos[intraIdx].boresightRotAngle.asRadians() - np.pi / 2,
                'ra': visitInfos[intraIdx].boresightRaDec.getRa().asRadians(),
                'dec': visitInfos[intraIdx].boresightRaDec.getDec().asRadians(),
                'az': visitInfos[intraIdx].boresightAzAlt.getLongitude().asRadians(),
                'alt': visitInfos[intraIdx].boresightAzAlt.getLatitude().asRadians(),
                'mjd': visitInfos[intraIdx].date.toAstropy().mjd
            }

        # Carefully average angles in meta
        out.meta['average'] = {}
        for k in ('parallacticAngle', 'rotAngle', 'rotTelPos', 'ra', 'dec', 'az', 'alt'):
            a1 = out.meta['extra'][k] * radians
            a2 = out.meta['intra'][k] * radians
            a2 = a2.wrapNear(a1)
            out.meta['average'][k] = ((a1 + a2) / 2).wrapCtr().asRadians()

        # Easier to average the MJDs
        if intraIdx is not None:
            out.meta['average']['mjd'] = np.mean([out.meta['extra']['mjd'], out.meta['intra']['mjd']])
        else:
            out.meta['average']['mjd'] = out.meta['extra']['mjd']

        q = out.meta['average']['parallacticAngle']
        rtp = out.meta['average']['rotTelPos']
        out['thx_OCS'] = np.cos(rtp) * out['thx_CCS'] + np.sin(rtp) * out['thy_CCS']
        out['thy_OCS'] = -np.sin(rtp) * out['thx_CCS'] + np.cos(rtp) * out['thy_CCS']
        out['th_N'] = np.cos(q) * out['thx_CCS'] + np.sin(q) * out['thy_CCS']
        out['th_E'] = -np.sin(q) * out['thx_CCS'] + np.cos(q) * out['thy_CCS']

        for outRef in outputRefs.aggregateDonutCatalog:
            if outRef.dataId['visit'] == extraIdx:
                butlerQC.put(out, outRef)
                return
        raise ValueError(f"Expected to find an output reference with visit {extraIdx}")


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    aggregateDonutCatalog = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateDonutCatalog",
        multiple=True
    )
    aggregateZernikesRaw = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
        multiple=True
    )
    aggregateZernikesAvg = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
        multiple=True
    )
    aggregateAOSRaw = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
        multiple=True
    )
    aggregateAOSAvg = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
        multiple=True
    )

class AggregateAOSVisitTableTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateAOSVisitTableTaskConnections,
):
    pass


class AggregateAOSVisitTableTask(pipeBase.PipelineTask):
    ConfigClass = AggregateAOSVisitTableTaskConfig
    _DefaultName = "AggregateAOSVisitTable"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ) -> None:
        adc = butlerQC.get(inputRefs.aggregateDonutCatalog[0])
        azr = butlerQC.get(inputRefs.aggregateZernikesRaw[0])
        aza = butlerQC.get(inputRefs.aggregateZernikesAvg[0])

        dets = np.unique(adc['detector'])
        avg_table = aza.copy()
        avg_keys = [
            'coord_ra', 'coord_dec',
            'centroid_x', 'centroid_y',
            'thx_CCS', 'thy_CCS',
            'thx_OCS', 'thy_OCS',
            'th_N', 'th_E'
        ]
        for k in avg_keys:
            avg_table[k] = np.nan  # Allocate

        for det in dets:
            w = avg_table['detector'] == det
            for k in avg_keys:
                avg_table[k][w] = np.mean(adc[k][adc['detector'] == det])

        raw_table = azr.copy()
        for k in avg_keys:
            raw_table[k] = np.nan  # Allocate
        for det in dets:
            w = raw_table['detector'] == det
            wadc = adc['detector'] == det
            fzmin = adc[wadc]['focusZ'].min()
            fzmax = adc[wadc]['focusZ'].max()
            if fzmin == fzmax:  # single-sided Zernike estimates
                for k in avg_keys:
                    raw_table[k][w] = adc[k][wadc]
            else:  # double-sided Zernike estimates
                wintra = adc[wadc]['focusZ'] == fzmin
                wextra = adc[wadc]['focusZ'] == fzmax
                for k in avg_keys:
                    # If one catalog has more rows than the other, trim the longer one
                    if wintra.sum() > wextra.sum():
                        wintra[wintra] = [True]*wextra.sum() + [False]*(wintra.sum() - wextra.sum())
                    elif wextra.sum() > wintra.sum():
                        wextra[wextra] = [True]*wintra.sum() + [False]*(wextra.sum() - wintra.sum())
                    # ought to be the same length now
                    raw_table[k][w] = 0.5 * (adc[k][wadc][wintra] + adc[k][wadc][wextra])
                    if k+'_intra' not in raw_table.colnames:
                        raw_table[k+'_intra'] = np.nan
                        raw_table[k+'_extra'] = np.nan
                    raw_table[k+'_intra'][w] = adc[k][wadc][wintra]
                    raw_table[k+'_extra'][w] = adc[k][wadc][wextra]

        butlerQC.put(avg_table, outputRefs.aggregateAOSAvg[0])
        butlerQC.put(raw_table, outputRefs.aggregateAOSRaw[0])


class AggregateDonutStampsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    donutStampsIntra = ct.Input(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True
    )
    donutStampsExtra = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True
    )
    donutStampsIntraVisit = ct.Output(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraVisit",
        multiple=True
    )
    donutStampsExtraVisit = ct.Output(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtraVisit",
        multiple=True
    )


class AggregateDonutStampsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutStampsTaskConnections,
):
    maxDonutsPerDetector = pexConfig.Field[int](
        doc="Maximum number of donuts to use per detector",
        default=1,
    )

    def validate(self):
        if self.maxDonutsPerDetector < 1:
            raise pexConfig.FieldValidationError("maxDonutsPerDetector must be at least 1")


class AggregateDonutStampsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutStampsTaskConfig
    _DefaultName = "AggregateDonutStamps"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ) -> None:
        intraStamps = DonutStamps([])
        extraStamps = DonutStamps([])
        for intraRef, extraRef in zip(inputRefs.donutStampsIntra, inputRefs.donutStampsExtra):
            intra = butlerQC.get(intraRef)
            extra = butlerQC.get(extraRef)
            intraStamps.extend(intra[:self.config.maxDonutsPerDetector])
            extraStamps.extend(extra[:self.config.maxDonutsPerDetector])

        butlerQC.put(intraStamps, outputRefs.donutStampsIntraVisit[0])
        butlerQC.put(extraStamps, outputRefs.donutStampsExtraVisit[0])
