import galsim
import numpy as np
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as ct
from astropy.table import Table, vstack
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.geom import Point2D, radians
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.pairTask import ExposurePairer


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
    dimensions=("instrument", "visit"),
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
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
        multiple=True,
    )
    zernikesAvg = ct.Input(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
        multiple=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    aggregateZernikesRaw = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
    )
    aggregateZernikesAvg = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
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
        visit = outputRefs.aggregateZernikesRaw.dataId['visit']
        camera = butlerQC.get(inputRefs.camera)

        raw_tables = []
        avg_tables = []
        for zernikeRawRef, zernikeAvgRef in zip(inputRefs.zernikesRaw, inputRefs.zernikesAvg):
            zernikeRaw = butlerQC.get(zernikeRawRef)
            zernikeAvg = butlerQC.get(zernikeAvgRef)
            raw_table = Table()
            raw_table['zk_CCS'] = zernikeRaw
            raw_table['detector'] = camera[zernikeRawRef.dataId['detector']].getName()
            raw_tables.append(raw_table)
            avg_table = Table()
            avg_table['zk_CCS'] = zernikeAvg.reshape(1, -1)
            avg_table['detector'] = camera[zernikeAvgRef.dataId['detector']].getName()
            avg_tables.append(avg_table)
        out_raw = vstack(raw_tables)
        out_avg = vstack(avg_tables)

        # just get the first one, they're all the same
        visitInfo = butlerQC.get(inputRefs.visitInfos[0])

        meta = {}
        meta['visit'] = visit
        meta['parallacticAngle'] = visitInfo.boresightParAngle.asRadians()
        meta['rotAngle'] = visitInfo.boresightRotAngle.asRadians()
        rtp = (
            visitInfo.boresightParAngle
            - visitInfo.boresightRotAngle
            - (np.pi / 2 * radians)
        ).asRadians()
        meta['rotTelPos'] = rtp
        meta['ra'] = visitInfo.boresightRaDec.getRa().asRadians()
        meta['dec'] = visitInfo.boresightRaDec.getDec().asRadians()
        meta['az'] = visitInfo.boresightAzAlt.getLongitude().asRadians()
        meta['alt'] = visitInfo.boresightAzAlt.getLatitude().asRadians()
        meta['mjd'] = visitInfo.date.toAstropy().mjd

        q = meta['parallacticAngle']
        rtp = meta['rotTelPos']

        jmax = out_raw['zk_CCS'].shape[1] + 3
        rot_OCS = galsim.zernike.zernikeRotMatrix(jmax, -rtp)[4:,4:]
        rot_NW = galsim.zernike.zernikeRotMatrix(jmax, -q)[4:,4:]
        for cat in (out_raw, out_avg):
            cat.meta = meta
            cat['zk_OCS'] = cat['zk_CCS'] @ rot_OCS
            cat['zk_NW'] = cat['zk_CCS'] @ rot_NW

        # Find the right output references
        butlerQC.put(out_raw, outputRefs.aggregateZernikesRaw)
        butlerQC.put(out_avg, outputRefs.aggregateZernikesAvg)


# Note: cannot make visit a dimension because we have not yet paired visits.
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
    donut_visit_pair_table = ct.Input(
        doc="Visit pair table",
        dimensions=("instrument",),
        storageClass="AstropyTable",
        name="donut_visit_pair_table",
    )
    donutCatalogs = ct.Input(
        doc="Donut catalogs",
        dimensions=("visit", "detector","instrument"),
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

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.pairer.target._needsPairTable:
            del self.donut_visit_pair_table


class AggregateDonutCatalogsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutCatalogsTaskConnections,
):
    pairer = pexConfig.ConfigurableField(
        target=ExposurePairer,
        doc="Task to pair up intra- and extra-focal exposures",
    )


class AggregateDonutCatalogsTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutCatalogsTaskConfig
    _DefaultName = "AggregateDonutCatalogs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("pairer")

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ):
        camera = butlerQC.get(inputRefs.camera)

        # Only need one visitInfo per exposure.
        # And the pairer only works with uniquified visitInfos.
        visitInfoDict = {}
        for visitInfoRef in inputRefs.visitInfos:
            exposure = visitInfoRef.dataId['exposure']
            if exposure in visitInfoDict:
                continue
            visitInfoDict[exposure] = butlerQC.get(visitInfoRef)

        if hasattr(inputRefs, "donut_visit_pair_table"):
            pairs = self.pairer.run(visitInfoDict, butlerQC.get(inputRefs.donut_visit_pair_table))
        else:
            pairs = self.pairer.run(visitInfoDict)

        for pair in pairs:
            intraVisitInfo = visitInfoDict[pair.intra]
            extraVisitInfo = visitInfoDict[pair.extra]

            tables = []
            # Find all detectors...
            for donutCatalogRef in inputRefs.donutCatalogs:
                dataId = donutCatalogRef.dataId
                if dataId['visit'] not in (pair.intra, pair.extra):
                    continue

                det = camera[dataId['detector']]
                tform = det.getTransform(PIXELS, FIELD_ANGLE)

                donutCatalog = butlerQC.get(donutCatalogRef)
                table = Table.from_pandas(donutCatalog)
                table['focusZ'] = intraVisitInfo.focusZ if dataId['visit'] == pair.intra else extraVisitInfo.focusZ
                pts = tform.applyForward(
                    [Point2D(x, y) for x, y in zip(table['centroid_x'], table['centroid_y'])]
                )
                table['thx_CCS'] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
                table['thy_CCS'] = [pt.x for pt in pts]
                table['detector'] = det.getName()

                tables.append(table)
            out = vstack(tables)

            out.meta['extra'] = {
                'visit': pair.extra,
                'focusZ': extraVisitInfo.focusZ,
                'parallacticAngle': extraVisitInfo.boresightParAngle.asRadians(),
                'rotAngle': extraVisitInfo.boresightRotAngle.asRadians(),
                'rotTelPos': extraVisitInfo.boresightParAngle.asRadians() - extraVisitInfo.boresightRotAngle.asRadians() - np.pi / 2,
                'ra': extraVisitInfo.boresightRaDec.getRa().asRadians(),
                'dec': extraVisitInfo.boresightRaDec.getDec().asRadians(),
                'az': extraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
                'alt': extraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
                'mjd': extraVisitInfo.date.toAstropy().mjd
            }
            out.meta['intra'] = {
                'visit': pair.intra,
                'focusZ': intraVisitInfo.focusZ,
                'parallacticAngle': intraVisitInfo.boresightParAngle.asRadians(),
                'rotAngle': intraVisitInfo.boresightRotAngle.asRadians(),
                'rotTelPos': intraVisitInfo.boresightParAngle.asRadians() - intraVisitInfo.boresightRotAngle.asRadians() - np.pi / 2,
                'ra': intraVisitInfo.boresightRaDec.getRa().asRadians(),
                'dec': intraVisitInfo.boresightRaDec.getDec().asRadians(),
                'az': intraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
                'alt': intraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
                'mjd': intraVisitInfo.date.toAstropy().mjd
            }

            # Carefully average angles in meta
            out.meta['average'] = {}
            for k in ('parallacticAngle', 'rotAngle', 'rotTelPos', 'ra', 'dec', 'az', 'alt'):
                a1 = out.meta['extra'][k] * radians
                a2 = out.meta['intra'][k] * radians
                a2 = a2.wrapNear(a1)
                out.meta['average'][k] = ((a1 + a2) / 2).wrapCtr().asRadians()

            # Easier to average the MJDs
            out.meta['average']['mjd'] = 0.5*(out.meta['extra']['mjd'] + out.meta['intra']['mjd'])

            q = out.meta['average']['parallacticAngle']
            rtp = out.meta['average']['rotTelPos']
            out['thx_OCS'] = np.cos(rtp) * out['thx_CCS'] - np.sin(rtp) * out['thy_CCS']
            out['thy_OCS'] = np.sin(rtp) * out['thx_CCS'] + np.cos(rtp) * out['thy_CCS']
            out['th_N'] = np.cos(q) * out['thx_CCS'] - np.sin(q) * out['thy_CCS']
            out['th_W'] = np.sin(q) * out['thx_CCS'] + np.cos(q) * out['thy_CCS']

        # Find the right output references
            for outRef in outputRefs.aggregateDonutCatalog:
                if outRef.dataId['visit'] == pair.extra:
                    butlerQC.put(out, outRef)
                    break
            else:
                raise ValueError(f"Expected to find an output reference with visit {pair.extra}")


class AggregateAOSVisitTableTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument",),
):
    aggregateDonutCatalog = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateDonutCatalog",
    )
    aggregateZernikesRaw = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesRaw",
    )
    aggregateZernikesAvg = ct.Input(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateZernikesAvg",
    )
    aggregateAOSRaw = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    aggregateAOSAvg = ct.Output(
        doc="Visit-level catalog of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
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
        adc = butlerQC.get(inputRefs.aggregateDonutCatalog)
        azr = butlerQC.get(inputRefs.aggregateZernikesRaw)
        aza = butlerQC.get(inputRefs.aggregateZernikesAvg)

        dets = np.unique(adc['detector'])
        avg_table = aza.copy()
        avg_keys = [
            'coord_ra', 'coord_dec',
            'centroid_x', 'centroid_y',
            'thx_CCS', 'thy_CCS',
            'thx_OCS', 'thy_OCS',
            'th_N', 'th_W'
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

        butlerQC.put(avg_table, outputRefs.aggregateAOSAvg)
        butlerQC.put(raw_table, outputRefs.aggregateAOSRaw)


class AggregateDonutStampsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),
):
    donutStampsIntra = ct.Input(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )
    donutStampsExtra = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
    )
    donutStampsIntraVisit = ct.Output(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraVisit",
    )
    donutStampsExtraVisit = ct.Output(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtraVisit",
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

        butlerQC.put(intraStamps, outputRefs.donutStampsIntraVisit)
        butlerQC.put(extraStamps, outputRefs.donutStampsExtraVisit)
