import lsst.pipe.base.connectionTypes as ct
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import galsim
import tempfile
import yaml

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .zernikePyramid import zernikePyramid
from .utilities import rose, add_rotated_axis, get_instrument_channel_name
from pathlib import Path

try:
    from lsst.rubintv.production.uploaders import MultiUploader
except ImportError:
    MultiUploader = None

__all__ = [
    "PlotAOSTaskConnections",
    "PlotAOSTaskConfig",
    "PlotAOSTask",
    "PlotDonutTaskConnections",
    "PlotDonutTaskConfig",
    "PlotDonutTask",
]


class PlotAOSTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
    aggregateAOSRaw = ct.Input(
        doc="AOS raw catalog",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    aggregateAOSAvg = ct.Input(
        doc="AOS average catalog",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
    )
    measuredZernikePyramid = ct.Output(
        doc="Measurement AOS Zernike pyramid",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="measuredZernikePyramid",
    )
    intrinsicZernikePyramid = ct.Output(
        doc="Intrinsic AOS Zernike pyramid",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="intrinsicZernikePyramid",
    )
    residualZernikePyramid = ct.Output(
        doc="Residual AOS Zernike pyramid",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="residualZernikePyramid",
    )


class PlotAOSTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotAOSTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotAOSTask(pipeBase.PipelineTask):
    ConfigClass = PlotAOSTaskConfig
    _DefaultName = "plotAOSTask"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.doRubinTVUpload:
            if not MultiUploader:
                raise RuntimeError("MultiUploader is not available")
            self.uploader = MultiUploader()

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ) -> None:
        aos_raw = butlerQC.get(inputRefs.aggregateAOSRaw)
        # aos_avg = butlerQC.get(inputRefs.aggregateAOSAvg)

        zkPyramid, residPyramid, intrinsicPyramid = self.plotZernikePyramids(aos_raw)

        butlerQC.put(zkPyramid, outputRefs.measuredZernikePyramid)
        butlerQC.put(residPyramid, outputRefs.residualZernikePyramid)
        butlerQC.put(intrinsicPyramid, outputRefs.intrinsicZernikePyramid)

        if self.config.doRubinTVUpload:
            instrument = inputRefs.aggregateAOSRaw.dataId['instrument']
            visit = inputRefs.aggregateAOSRaw.dataId['visit']
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
            with tempfile.TemporaryDirectory() as tmpdir:
                zk_meas_fn = Path(tmpdir) / 'zk_measurement_pyramid.png'
                zkPyramid.savefig(zk_meas_fn)
                zk_resid_fn = Path(tmpdir) / 'zk_residual_pyramid.png'
                residPyramid.savefig(zk_resid_fn)

                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(instrument),
                    plotName='zk_measurement_pyramid',
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=zk_meas_fn
                )
                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(instrument),
                    plotName='zk_residual_pyramid',
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=zk_resid_fn
                )

    def doPyramid(
        self,
        x, y, zk,
        rtp, q,
    ):
        fig = zernikePyramid(x, y, zk, cmap='seismic', s=2)
        vecs_xy = {
            '$x_\mathrm{Opt}$':(1,0),
            '$y_\mathrm{Opt}$':(0,-1),
            '$x_\mathrm{Cam}$':(np.cos(rtp), -np.sin(rtp)),
            '$y_\mathrm{Cam}$':(-np.sin(rtp), -np.cos(rtp)),
        }
        rose(fig, vecs_xy, p0=(0.15, 0.8))

        vecs_NE = {
            'az':(1,0),
            'alt':(0,+1),
            'N':(np.sin(q), np.cos(q)),
            'E':(np.sin(q-np.pi/2), np.cos(q-np.pi/2))
        }
        rose(fig, vecs_NE, p0=(0.85, 0.8))

        return fig

    def plotZernikePyramids(
        self,
        aos_raw,
    ) -> plt.Figure:
        # Cut out R30 for coordinate system check
        # wbad = np.isin(aos_raw['detector'], range(117, 126))
        # Cut out ComCam 'S21' and 'S22'
        # wbad = np.isin(aos_raw['detector'], [7, 8])
        # aos_raw = aos_raw[~wbad]

        x = aos_raw['thx_OCS']
        y = -aos_raw['thy_OCS']  # +y is down on plot
        zk = aos_raw['zk_OCS'].T
        rtp = aos_raw.meta['rotTelPos']
        q = aos_raw.meta['parallacticAngle']

        zkPyramid = self.doPyramid(x, y, zk, rtp, q)

        # We want residuals from the intrinsic design too.
        path = Path(__file__).parent.parent.parent.parent.parent / 'data'
        band = 'r'  # for a minute
        path /= f'intrinsic_dz_{band}.yaml'
        coefs = np.array(yaml.safe_load(open(path, 'r')))
        dzs = galsim.zernike.DoubleZernike(
            coefs,
            uv_outer=np.deg2rad(1.82),
            xy_outer=4.18,
            xy_inner=4.18*0.612,
        )
        intrinsic = np.array([z.coef for z in dzs(aos_raw['thx_OCS'], aos_raw['thy_OCS'])]).T[4:23]
        intrinsicPyramid = self.doPyramid(x, y, intrinsic, rtp, q)

        resid = zk - intrinsic
        residPyramid = self.doPyramid(x, y, resid, rtp, q)

        return zkPyramid, residPyramid, intrinsicPyramid


class PlotDonutTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
    visitInfos = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="VisitInfo",
        name="raw.visitInfo",
        multiple=True,
    )
    donutStampsIntraVisit = ct.Input(
        doc="Intrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraVisit",
    )
    donutStampsExtraVisit = ct.Input(
        doc="Extrafocal Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtraVisit",
    )
    donutPlot = ct.Output(
        doc="Donut Plot",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlot",
    )

class PlotDonutTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotDonutTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )

class PlotDonutTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutTaskConfig
    _DefaultName = "plotDonutTask"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.doRubinTVUpload:
            if not MultiUploader:
                raise RuntimeError("MultiUploader is not available")
            self.uploader = MultiUploader()

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection
    ) -> None:
        visit = inputRefs.donutStampsIntraVisit.dataId['visit']
        for visitInfoRef in inputRefs.visitInfos:
            if visitInfoRef.dataId['exposure'] == visit:
                visitInfo = butlerQC.get(visitInfoRef)
                break
        else:
            raise ValueError(f"Expected to find a visitInfo with exposure {visit}")

        donutStampsIntra = butlerQC.get(inputRefs.donutStampsIntraVisit)
        donutStampsExtra = butlerQC.get(inputRefs.donutStampsExtraVisit)

        fig = plt.figure(figsize=(11, 8.5))
        aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]

        # LSST detector layout
        q = visitInfo.boresightParAngle.asRadians()
        rotAngle = visitInfo.boresightRotAngle.asRadians()
        rtp = q - rotAngle - np.pi/2
        match inst:=visitInfo.getInstrumentLabel():
            case 'LSSTCam' | 'LSSTCamSim':
                nacross = 15
                fp_size = 0.55  # 55% of horizontal space
            case 'LSSTComCam' | 'LSSTComCamSim':
                nacross = 3
                fp_size = 0.50  # 50% of horizontal space
            case _:
                raise ValueError(f"Unknown instrument {inst}")
        det_size = fp_size/nacross
        fp_center = 0.5, 0.475

        for donut in donutStampsIntra:
            det_name = donut.detector_name
            # if 'R30' in det_name:
            #     continue
            # if 'S21' in det_name:
            #     continue
            # if 'S22' in det_name:
            #     continue
            i = 3*int(det_name[1]) + int(det_name[5])
            j = 3*int(det_name[2]) + int(det_name[6])
            x = i-7
            y = 7-j
            xp = np.cos(rtp)*x + np.sin(rtp)*y
            yp = -np.sin(rtp)*x + np.cos(rtp)*y
            ax, aux_ax = add_rotated_axis(
                fig,
                (xp*det_size + fp_center[0], yp*det_size*aspect + fp_center[1]),
                (det_size*1.25, det_size*1.25),
                -np.rad2deg(rtp)
            )
            arr = donut.stamp_im.image.array
            vmin, vmax = np.quantile(arr, (0.01, 0.99))
            aux_ax.imshow(
                donut.stamp_im.image.array.T,
                vmin=vmin, vmax=vmax,
                extent=[0, det_size*1.25, 0, det_size*1.25],
                origin='upper'  # +y is down
            )

        vecs_xy = {
            '$x_\mathrm{Opt}$':(1,0),
            '$y_\mathrm{Opt}$':(0,-1),
            '$x_\mathrm{Cam}$':(np.cos(rtp), -np.sin(rtp)),
            '$y_\mathrm{Cam}$':(-np.sin(rtp), -np.cos(rtp)),
        }
        rose(fig, vecs_xy, p0=(0.15, 0.8))

        vecs_NE = {
            'az':(1,0),
            'alt':(0,+1),
            'N':(np.sin(q), np.cos(q)),
            'E':(np.sin(q-np.pi/2), np.cos(q-np.pi/2))
        }
        rose(fig, vecs_NE, p0=(0.85, 0.8))

        butlerQC.put(fig, outputRefs.donutPlot)

        if self.config.doRubinTVUpload:
            instrument = inputRefs.donutStampsIntraVisit.dataId['instrument']
            visit = inputRefs.donutStampsIntraVisit.dataId['visit']
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
            with tempfile.TemporaryDirectory() as tmpdir:
                donut_gallery_fn = Path(tmpdir) / 'fp_donut_gallery.png'
                fig.savefig(donut_gallery_fn)

                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(instrument),
                    plotName='fp_donut_gallery',
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=donut_gallery_fn
                )
