from copy import copy
from pathlib import Path

import galsim
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u
from lsst.ts.wep.task import DonutStamps
from lsst.ts.wep.utils import convertZernikesToPsfWidth
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod

from .psf_from_zern import psfPanel
from .utilities import (
    add_coordinate_roses,
    add_rotated_axis,
    get_day_obs_seq_num_from_visitid,
    get_instrument_channel_name,
    rose,
)
from .zernike_pyramid import zernikePyramid

try:
    from lsst.rubintv.production.uploaders import MultiUploader
    from lsst.rubintv.production.utils import getAutomaticLocationConfig, makePlotFile
except ImportError:
    MultiUploader = None

__all__ = [
    "PlotAOSTaskConnections",
    "PlotAOSTaskConfig",
    "PlotAOSTask",
    "PlotDonutTaskConnections",
    "PlotDonutTaskConfig",
    "PlotDonutTask",
    "PlotDonutCwfsTaskConnections",
    "PlotDonutCwfsTaskConfig",
    "PlotDonutCwfsTask",
    "PlotPsfZernTaskConnections",
    "PlotPsfZernTaskConfig",
    "PlotPsfZernTask",
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
    shiftFactor = pexConfig.Field(
        dtype=float,
        doc="A shift to be applied to the x,y position of \
        the Zernike data plotted for the Zernike pyramid,\
        expressed as a fraction of the distance from the \
        corner to the center (i.e. between 0 and 1.0).",
        default=0.90,
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
            locationConfig = getAutomaticLocationConfig()
            instrument = inputRefs.aggregateAOSRaw.dataId["instrument"]
            visit = inputRefs.aggregateAOSRaw.dataId["visit"]
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "zk_measurement_pyramid"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
            )
            zkPyramid.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

            plotName = "zk_residual_pyramid"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
            )
            residPyramid.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def doPyramid(self, x, y, zk, rtp, q, nollIndices):
        fig = zernikePyramid(x, y, zk, nollIndices, cmap="seismic", s=10)
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

        zk = aos_raw["zk_OCS"].T
        rtp = aos_raw.meta["rotTelPos"]
        q = aos_raw.meta["parallacticAngle"]
        nollIndices = aos_raw.meta["nollIndices"]

        # check if there is data for any corner sensor
        if (
            np.sum(
                ["SW" in detName for detName in np.unique(aos_raw["detector"].value)]
            )
            > 0
        ):
            # in that case, shift x,y positions
            # towards the center, along the diagonal
            # rotate the original CCS into OCS,
            # and then invert y
            x_ccs = aos_raw["thx_CCS"].value
            y_ccs = aos_raw["thy_CCS"].value
            detector = aos_raw["detector"].value
            x_ocs_shift, y_ocs_shift = self.shiftAlongDiagonalCwfs(
                x_ccs, y_ccs, detector, rtp, self.config.shiftFactor
            )
            x = x_ocs_shift
            y = -y_ocs_shift  # +y is down on plot
        # otherwise it's FAM data, which requires no shifting
        else:
            x = aos_raw["thx_OCS"]
            y = -aos_raw["thy_OCS"]  # +y is down on plot
        zkPyramid = self.doPyramid(x, y, zk, rtp, q, nollIndices)

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
        intrinsic = intrinsic[: len(zk)]
        intrinsicPyramid = self.doPyramid(x, y, intrinsic, rtp, q, nollIndices)

        resid = zk - intrinsic
        residPyramid = self.doPyramid(x, y, resid, rtp, q, nollIndices)

        return zkPyramid, residPyramid, intrinsicPyramid

    def shiftAlongDiagonalCwfs(
        self,
        x_ccs: np.ndarray,
        y_ccs: np.ndarray,
        detector: np.ndarray,
        rtp: float,
        shift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A function to take x,y coordinates in CCS,
        and depending on which detector they belong to,
        shift them by a fraction of their mean distance
        from the coordinate origin (0,0).

        Parameters:
        -----------
        x_CCS, y_CCS: np.ndarray
            Original x,y coordinates in CCS
        detector: np.ndarray
            List of detector names
        rotTelPos: float
           The rotation angle of the telescope in radians.
           Rotating CCS counterclockwise by rotTelPos
           aligns it with OCS.
        shift: float
            Amount of shift expressed as a fraction of
            the mean detector distance from the origin
            in CCS (i.e. between 0 and 1).

        Returns:
        --------
        x_shifted, y_shifted : np.ndarray
            Shifted x,y coordinates, rotated to OCS
        """

        x_ccs_shift = copy(x_ccs)
        y_ccs_shift = copy(y_ccs)

        # Shift x,y coordinates by the
        # fraction of their mean distance
        # from the center of the coordinate
        # system
        for det in np.unique(detector):
            rows = detector == det
            mean_x = np.mean(x_ccs[rows])
            mean_y = np.mean(y_ccs[rows])

            shift_x = -shift * mean_x
            shift_y = -shift * mean_y

            x_ccs_shift[rows] = x_ccs[rows] + shift_x
            y_ccs_shift[rows] = y_ccs[rows] + shift_y

        # Rotate these by rotTelPos
        R = np.array(
            [
                [np.cos(rtp), -np.sin(rtp)],
                [np.sin(rtp), np.cos(rtp)],
            ]
        )

        points_ccs_shift = np.vstack((x_ccs_shift, y_ccs_shift))
        points_ccs_shift_rotated = R @ points_ccs_shift
        x_ccs_shift_rot, y_ccs_shift_rot = (
            points_ccs_shift_rotated[0, :],
            points_ccs_shift_rotated[1, :],
        )
        return x_ccs_shift_rot, y_ccs_shift_rot


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
    doS11only = pexConfig.Field(
        dtype=bool, doc="Use only S11 in FAM mode", default=False
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
        inst = inputRefs.donutStampsIntraVisit.dataId["instrument"]

        donutStampsIntra = butlerQC.get(inputRefs.donutStampsIntraVisit)
        donutStampsExtra = butlerQC.get(inputRefs.donutStampsExtraVisit)
        # We take visitId corresponding to each donut sets from
        # donutStamps metadata as the
        # visitId above under which donutStamps were saved
        # is only the extra-focal visitId
        fig_dict = self.run(donutStampsIntra, donutStampsExtra, inst)

        butlerQC.put(fig_dict["extra"], outputRefs.donutPlotExtra)
        butlerQC.put(fig_dict["intra"], outputRefs.donutPlotIntra)

        visitIntra = donutStampsIntra.metadata.getArray("VISIT")[0]
        visitExtra = donutStampsExtra.metadata.getArray("VISIT")[0]

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            # seq_num is sometimes different for
            # intra vs extra-focal if pistoning
            for defocal_type, visit_id in zip(
                ["extra", "intra"], [visitExtra, visitIntra]
            ):
                day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit_id)

                plotName = "fp_donut_gallery"
                plotFile = makePlotFile(
                    locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
                )
                fig_dict[defocal_type].savefig(plotFile)
                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(inst),
                    plotName=plotName,
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=plotFile,
                )

    @timeMethod
    def run(
        self, donutStampsIntra: DonutStamps, donutStampsExtra: DonutStamps, inst: str
    ):

        visitIntra = donutStampsIntra.metadata.getArray("VISIT")[0]
        visitExtra = donutStampsExtra.metadata.getArray("VISIT")[0]

        # LSST detector layout
        q = donutStampsExtra.metadata["BORESIGHT_PAR_ANGLE_RAD"]
        rotAngle = donutStampsExtra.metadata["BORESIGHT_ROT_ANGLE_RAD"]
        rtp = q - rotAngle - np.pi / 2

        # Default multiplication factor and offset
        factor = 3
        offset = 7

        match inst:
            case "LSSTCam" | "LSSTCamSim":
                nacross = 15
                fp_size = 0.55  # 55% of horizontal space
                if self.config.doS11only:
                    factor = 1
                    offset = 3
                    nacross = 5
            case "LSSTComCam" | "LSSTComCamSim":
                nacross = 3
                fp_size = 0.50  # 50% of horizontal space
            case _:
                raise ValueError(f"Unknown instrument {inst}")
        det_size = fp_size / nacross
        fp_center = 0.5, 0.475

        fig_dict = dict()

        for donutStampSet, visit in zip(
            [donutStampsIntra, donutStampsExtra], [visitIntra, visitExtra]
        ):

            fig = make_figure(figsize=(11, 8.5))
            aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]
            for donut in donutStampSet:
                det_name = donut.detector_name
                # For FAM mode, if plotting only S11 corner, do not
                # plot anything else
                if self.config.doS11only:
                    if det_name[-2:] != "11":
                        continue
                i = factor * int(det_name[1]) + int(det_name[5])
                j = factor * int(det_name[2]) + int(det_name[6])
                x = i - offset
                y = offset - j
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

            add_coordinate_roses(fig, rtp, q)

            fig.text(0.47, 0.93, f"{donut.defocal_type}: {visit}")
            fig_dict[donut.defocal_type] = fig

        return fig_dict


class PlotDonutCwfsTaskConnections(
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
    donutPlot = ct.Output(
        doc="Donut Plot",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlot",
    )


class PlotDonutCwfsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotDonutCwfsTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotDonutCwfsTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutCwfsTaskConfig
    _DefaultName = "plotDonutCwfsTask"

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
        inst = inputRefs.donutStampsIntraVisit.dataId["instrument"]

        donutStampsIntra = butlerQC.get(inputRefs.donutStampsIntraVisit)
        donutStampsExtra = butlerQC.get(inputRefs.donutStampsExtraVisit)

        fig = self.run(donutStampsIntra, donutStampsExtra, inst)

        butlerQC.put(fig, outputRefs.donutPlot)

        # Same visit for both extra and intra-focal corner sensors
        visit = donutStampsIntra.metadata.getArray("VISIT")[0]

        if self.config.doRubinTVUpload:
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "fp_donut_gallery"
            plotFile = makePlotFile(
                self.locationConfig, self.instrument, day_obs, seq_num, plotName, "png"
            )
            fig.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(inst),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    @timeMethod
    def run(
        self, donutStampsIntra: DonutStamps, donutStampsExtra: DonutStamps, inst: str
    ):

        visit = donutStampsIntra.metadata.getArray("VISIT")[0]
        # LSST detector layout
        q = donutStampsExtra.metadata["BORESIGHT_PAR_ANGLE_RAD"]
        rotAngle = donutStampsExtra.metadata["BORESIGHT_ROT_ANGLE_RAD"]
        rtp = q - rotAngle - np.pi / 2

        # Combine all donuts into one list
        donutStampsList = []
        for stamp in donutStampsExtra:
            donutStampsList.append(stamp)
        for stamp in donutStampsIntra:
            donutStampsList.append(stamp)

        fp_center = 0.5, 0.475
        fp_size = 0.7
        nacross = 4
        det_size = fp_size / nacross
        fig = make_figure(figsize=(11, 8.5))

        aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]

        for donut in donutStampsList:
            det_name = donut.detector_name
            if det_name == "R00_SW0":
                i = 0
                j = -1
                nrot90 = 2
            elif det_name == "R00_SW1":
                i = -1
                j = -1
                nrot90 = 0
            elif det_name == "R44_SW0":
                i = 0 + 0.5
                j = 1 - 0.5
                nrot90 = 0
            elif det_name == "R44_SW1":
                i = 1 + 0.5
                j = 1 - 0.5
                nrot90 = 2
            elif det_name == "R04_SW0":
                i = -1 + 0.5
                j = 0
                nrot90 = 3
            elif det_name == "R04_SW1":
                i = -1 + 0.5
                j = 1
                nrot90 = 1
            elif det_name == "R40_SW0":
                i = 1
                j = 0 - 0.5
                nrot90 = 1
            elif det_name == "R40_SW1":
                i = 1
                j = -1 - 0.5
                nrot90 = 3
            x = i - 0.25
            y = -j
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
                np.rot90(donut.stamp_im.image.array.T, nrot90),
                vmin=vmin,
                vmax=vmax,
                extent=[0, det_size * 1.25, 0, det_size * 1.25],
                origin="upper",  # +y is down
            )
            xlim = aux_ax.get_xlim()
            ylim = aux_ax.get_ylim()
            defocal = "extra" if det_name[-3:] == "SW0" else "intra"
            label = f"{det_name} {defocal}"
            aux_ax.text(
                xlim[0] + 0.03 * (xlim[1] - xlim[0]),
                ylim[1] - 0.03 * (ylim[1] - ylim[0]),
                label,
                color="w",
                rotation=-np.rad2deg(rtp),
                rotation_mode="anchor",
                ha="left",
                va="top",
            )
        add_coordinate_roses(fig, rtp, q)
        fig.text(0.47, 0.97, f"{visit}")
        return fig


class PlotPsfZernTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
    zernikes = ct.Input(
        doc="Zernikes catalog",
        dimensions=("visit", "instrument", "detector"),
        storageClass="AstropyTable",
        multiple=True,
        name="zernikes",
    )
    psfFromZernPanel = ct.Output(
        doc="PSF value retrieved from zernikes",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="psfFromZernPanel",
    )


class PlotPsfZernTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotPsfZernTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotPsfZernTask(pipeBase.PipelineTask):
    ConfigClass = PlotPsfZernTaskConfig
    _DefaultName = "plotPsfZernTask"

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

        zernikes = butlerQC.get(inputRefs.zernikes)

        zkPanel = self.run(zernikes, figsize=(11, 14))

        butlerQC.put(zkPanel, outputRefs.psfFromZernPanel)

        if self.config.doRubinTVUpload:
            instrument = inputRefs.zernikes[0].dataId["instrument"]
            visit = inputRefs.zernikes[0].dataId["visit"]
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "psf_zk_panel"
            plotFile = makePlotFile(
                self.locationConfig, self.instrument, day_obs, seq_num, plotName, "png"
            )
            zkPanel.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName="psf_zk_panel",
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def run(self, zernikes, **kwargs) -> plt.figure:
        """Run the PlotPsfZern AOS task.

        This task create a 3x3 grid of subplots,
        each subplot shows the psf value calculated from the Zernike
        coefficients for each pair of intra-extra donuts found
        for each detector.

        Parameters
        ----------
        zernikes: list of tables.
            List of tables containing the zernike sets
            for each donut pair in each detector.
        **kwargs:
            Additional keyword arguments passed to
            matplotlib.pyplot.figure constructor.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure.
        """

        xs = []
        ys = []
        zs = []
        dname = []
        angles_set = False
        for i, qt in enumerate(zernikes):
            if len(qt) == 0:
                zs.append(np.array([]))
                dname.append([])
                xs.append([])
                ys.append([])
                continue
            dname.append(qt.meta["extra"]["det_name"])
            xs.append(qt["extra_centroid"]["x"][1:].value)
            ys.append(qt["extra_centroid"]["y"][1:].value)
            z = []
            for row in qt[[col for col in qt.colnames if "Z" in col]][1:].iterrows():
                z.append([el.to(u.micron).value for el in row])
            zs.append(np.array(z))

            if not angles_set:
                q = qt.meta["extra"]["boresight_par_angle_rad"]
                rot = qt.meta["extra"]["boresight_rot_angle_rad"]
                rtp = q - rot - np.pi / 2
                angles_set = True

        psf = [
            [
                np.sqrt(np.sum(convertZernikesToPsfWidth(pair_zset) ** 2))
                for pair_zset in det
            ]
            for det in zs
        ]

        fig = make_figure(**kwargs)
        fig.suptitle(
            f"PSF from Zernikes\nvisit: {zernikes[-1].meta['extra']['visit']}",
            fontsize="xx-large",
            fontweight="book",
        )
        fig = psfPanel(xs, ys, psf, dname, fig=fig)

        # draw rose
        add_coordinate_roses(fig, rtp, q, [(0.15, 0.94), (0.85, 0.94)])

        return fig
