import tempfile
from pathlib import Path

import galsim
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
import matplotlib.pyplot as plt
import numpy as np
import yaml
from lsst.utils.timer import timeMethod

from .utilities import (
    add_rotated_axis,
    get_day_obs_seq_num_from_visitid,
    get_instrument_channel_name,
    rose,
)
from .zernikePyramid import zernikePyramid

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

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        aos_raw = butlerQC.get(inputRefs.aggregateAOSRaw)
        # aos_avg = butlerQC.get(inputRefs.aggregateAOSAvg)

        zkPyramid, residPyramid, intrinsicPyramid = self.plotZernikePyramids(aos_raw)

        butlerQC.put(zkPyramid, outputRefs.measuredZernikePyramid)
        butlerQC.put(residPyramid, outputRefs.residualZernikePyramid)
        butlerQC.put(intrinsicPyramid, outputRefs.intrinsicZernikePyramid)

        if self.config.doRubinTVUpload:
            instrument = inputRefs.aggregateAOSRaw.dataId["instrument"]
            visit = inputRefs.aggregateAOSRaw.dataId["visit"]
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
            with tempfile.TemporaryDirectory() as tmpdir:
                zk_meas_fn = Path(tmpdir) / "zk_measurement_pyramid.png"
                zkPyramid.savefig(zk_meas_fn)
                zk_resid_fn = Path(tmpdir) / "zk_residual_pyramid.png"
                residPyramid.savefig(zk_resid_fn)

                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(instrument),
                    plotName="zk_measurement_pyramid",
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=zk_meas_fn,
                )
                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(instrument),
                    plotName="zk_residual_pyramid",
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=zk_resid_fn,
                )

    def doPyramid(
        self,
        x,
        y,
        zk,
        rtp,
        q,
    ):
        fig = zernikePyramid(x, y, zk, cmap="seismic", s=10)
        vecs_xy = {
            r"$x_\mathrm{Opt}$": (1, 0),
            r"$y_\mathrm{Opt}$": (0, -1),
            r"$x_\mathrm{Cam}$": (np.cos(rtp), -np.sin(rtp)),
            r"$y_\mathrm{Cam}$": (-np.sin(rtp), -np.cos(rtp)),
        }
        rose(fig, vecs_xy, p0=(0.15, 0.8))

        vecs_NE = {
            "az": (1, 0),
            "alt": (0, +1),
            "N": (np.sin(q), np.cos(q)),
            "E": (np.sin(q - np.pi / 2), np.cos(q - np.pi / 2)),
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

        x = aos_raw["thx_OCS"]
        y = -aos_raw["thy_OCS"]  # +y is down on plot
        zk = aos_raw["zk_OCS"].T
        rtp = aos_raw.meta["rotTelPos"]
        q = aos_raw.meta["parallacticAngle"]

        zkPyramid = self.doPyramid(x, y, zk, rtp, q)

        # We want residuals from the intrinsic design too.
        path = Path(__file__).parent.parent.parent.parent.parent / "data"
        band = "r"  # for a minute
        path /= f"intrinsic_dz_{band}.yaml"
        coefs = np.array(yaml.safe_load(open(path, "r")))
        dzs = galsim.zernike.DoubleZernike(
            coefs,
            uv_outer=np.deg2rad(1.82),
            xy_outer=4.18,
            xy_inner=4.18 * 0.612,
        )
        intrinsic = np.array(
            [z.coef for z in dzs(aos_raw["thx_OCS"], aos_raw["thy_OCS"])]
        ).T[4:29]
        intrinsicPyramid = self.doPyramid(x, y, intrinsic, rtp, q)

        resid = zk - intrinsic
        residPyramid = self.doPyramid(x, y, resid, rtp, q)

        return zkPyramid, residPyramid, intrinsicPyramid


class PlotDonutTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
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
    donutPlotIntra = ct.Output(
        doc="Donut Plot Intra",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlotIntra",
    )
    donutPlotExtra = ct.Output(
        doc="Donut Plot Extra",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlotExtra",
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

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        visit = inputRefs.donutStampsIntraVisit.dataId["visit"]
        inst = inputRefs.donutStampsIntraVisit.dataId["instrument"]

        donutStampsIntra = butlerQC.get(inputRefs.donutStampsIntraVisit)
        donutStampsExtra = butlerQC.get(inputRefs.donutStampsExtraVisit)
        # We take visitId corresponding to each donut sets from
        # donutStamps metadata as the
        # visitId above under which donutStamps were saved
        # is only the extra-focal visitId
        visitIntra = donutStampsIntra.metadata.getArray("VISIT")[0]
        visitExtra = donutStampsExtra.metadata.getArray("VISIT")[0]

        # LSST detector layout
        q = donutStampsExtra.metadata["BORESIGHT_PAR_ANGLE_RAD"]
        rotAngle = donutStampsExtra.metadata["BORESIGHT_ROT_ANGLE_RAD"]
        rtp = q - rotAngle - np.pi / 2
        match inst:
            case "LSSTCam" | "LSSTCamSim":
                nacross = 15
                fp_size = 0.55  # 55% of horizontal space
            case "LSSTComCam" | "LSSTComCamSim":
                nacross = 3
                fp_size = 0.50  # 50% of horizontal space
            case _:
                raise ValueError(f"Unknown instrument {inst}")
        det_size = fp_size / nacross
        fp_center = 0.5, 0.475

        for donutStampSet, visit in zip(
            [donutStampsIntra, donutStampsExtra], [visitIntra, visitExtra]
        ):

            fig = plt.figure(figsize=(11, 8.5))
            aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]
            for donut in donutStampSet:
                det_name = donut.detector_name
                # if 'R30' in det_name:
                #     continue
                # if 'S00' in det_name:
                #     continue
                # if 'S01' in det_name:
                #     continue
                i = 3 * int(det_name[1]) + int(det_name[5])
                j = 3 * int(det_name[2]) + int(det_name[6])
                x = i - 7
                y = 7 - j
                xp = np.cos(rtp) * x + np.sin(rtp) * y
                yp = -np.sin(rtp) * x + np.cos(rtp) * y
                ax, aux_ax = add_rotated_axis(
                    fig,
                    (
                        xp * det_size + fp_center[0],
                        yp * det_size * aspect + fp_center[1],
                    ),
                    (det_size * 1.25, det_size * 1.25),
                    -np.rad2deg(rtp),
                )
                arr = donut.stamp_im.image.array
                vmin, vmax = np.quantile(arr, (0.01, 0.99))
                aux_ax.imshow(
                    donut.stamp_im.image.array.T,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[0, det_size * 1.25, 0, det_size * 1.25],
                    origin="upper",  # +y is down
                )
                xlim = aux_ax.get_xlim()
                ylim = aux_ax.get_ylim()
                aux_ax.text(
                    xlim[0] + 0.03 * (xlim[1] - xlim[0]),
                    ylim[1] - 0.03 * (ylim[1] - ylim[0]),
                    det_name,
                    color="w",
                    rotation=-np.rad2deg(rtp),
                    rotation_mode="anchor",
                    ha="left",
                    va="top",
                )

            vecs_xy = {
                r"$x_\mathrm{Opt}$": (1, 0),
                r"$y_\mathrm{Opt}$": (0, -1),
                r"$x_\mathrm{Cam}$": (np.cos(rtp), -np.sin(rtp)),
                r"$y_\mathrm{Cam}$": (-np.sin(rtp), -np.cos(rtp)),
            }
            rose(fig, vecs_xy, p0=(0.15, 0.8))

            vecs_NE = {
                "az": (1, 0),
                "alt": (0, +1),
                "N": (np.sin(q), np.cos(q)),
                "E": (np.sin(q - np.pi / 2), np.cos(q - np.pi / 2)),
            }
            rose(fig, vecs_NE, p0=(0.85, 0.8))
            fig.text(0.47, 0.93, f"{donut.defocal_type}: {visit}")

            if donut.defocal_type == "extra":
                butlerQC.put(fig, outputRefs.donutPlotExtra)
            elif donut.defocal_type == "intra":
                butlerQC.put(fig, outputRefs.donutPlotIntra)

            if self.config.doRubinTVUpload:
                # that's the same for intra and extra-focal
                instrument = inputRefs.donutStampsIntraVisit.dataId["instrument"]

                # seq_num is sometimes different for
                # intra vs extra-focal if pistoning
                day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
                with tempfile.TemporaryDirectory() as tmpdir:
                    donut_gallery_fn = Path(tmpdir) / f"fp_donut_gallery_{visit}.png"
                    fig.savefig(donut_gallery_fn)

                    self.uploader.uploadPerSeqNumPlot(
                        instrument=get_instrument_channel_name(instrument),
                        plotName="fp_donut_gallery",
                        dayObs=day_obs,
                        seqNum=seq_num,
                        filename=donut_gallery_fn,
                    )
