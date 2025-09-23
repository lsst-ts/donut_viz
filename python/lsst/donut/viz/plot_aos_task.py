from copy import copy
from pathlib import Path

import batoid
import danish
import galsim
import lsst.afw
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
import numpy as np
import yaml
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from lsst.summit.utils.efdUtils import (
    getEfdData,
    getMostRecentRowWithDataBefore,
    makeEfdClient,
)
from lsst.summit.utils.plotting import stretchDataMidTone
from lsst.ts.wep.task import DonutStamps
from lsst.ts.wep.utils import convertZernikesToPsfWidth, getTaskInstrument
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch
from scipy.optimize import least_squares

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
    "PlotCwfsPairingTaskConnections",
    "PlotCwfsPairingTaskConfig",
    "PlotCwfsPairingTask",
    "PlotPsfZernTaskConnections",
    "PlotPsfZernTaskConfig",
    "PlotPsfZernTask",
    "PlotDonutFitsTaskConnections",
    "PlotDonutFitsTaskConfig",
    "PlotDonutFitsTask",
    "PlotDonutUnpairedCwfsTaskConnections",
    "PlotDonutUnpairedCwfsTaskConfig",
    "PlotDonutUnpairedCwfsTask",
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
    ) -> Figure:
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
            locationConfig = getAutomaticLocationConfig()
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "fp_donut_gallery"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
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
        # We make sure to pick the first, i.e.
        # the brightest, donut for each detector
        detectorsRead = []
        donutStampsList = []
        for stamp in donutStampsExtra:
            if stamp.detector_name not in detectorsRead:
                donutStampsList.append(stamp)
                detectorsRead.append(stamp.detector_name)
        for stamp in donutStampsIntra:
            if stamp.detector_name not in detectorsRead:
                donutStampsList.append(stamp)
                detectorsRead.append(stamp.detector_name)

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


class PlotDonutUnpairedCwfsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
    donutStampsUnpairedVisit = ct.Input(
        doc="Unpaired Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsUnpairedVisit",
    )
    donutPlot = ct.Output(
        doc="Donut Plot",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlot",
    )


class PlotDonutUnpairedCwfsTaskConfig(
    PlotDonutCwfsTaskConfig,
    pipelineConnections=PlotDonutUnpairedCwfsTaskConnections,
):
    pass


class PlotDonutUnpairedCwfsTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutUnpairedCwfsTaskConfig
    _DefaultName = "plotDonutUnpairedCwfsTask"

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
        inst = inputRefs.donutStampsUnpairedVisit.dataId["instrument"]

        donutStampsUnpaired = butlerQC.get(inputRefs.donutStampsUnpairedVisit)

        fig = self.run(donutStampsUnpaired, inst)

        butlerQC.put(fig, outputRefs.donutPlot)

        # Same visit for both extra and intra-focal corner sensors
        visit = donutStampsUnpaired.metadata.getArray("VISIT")[0]

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "fp_donut_gallery"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
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
    def run(self, donutStampsUnpaired: DonutStamps, inst: str):

        visit = donutStampsUnpaired.metadata.getArray("VISIT")[0]
        # LSST detector layout
        q = donutStampsUnpaired.metadata["BORESIGHT_PAR_ANGLE_RAD"]
        rotAngle = donutStampsUnpaired.metadata["BORESIGHT_ROT_ANGLE_RAD"]
        rtp = q - rotAngle - np.pi / 2

        # Combine all donuts into one list
        # We make sure to pick the first, i.e.
        # the brightest, donut for each detector
        detectorsRead = []
        donutStampsList = []
        for stamp in donutStampsUnpaired:
            if stamp.detector_name not in detectorsRead:
                donutStampsList.append(stamp)
                detectorsRead.append(stamp.detector_name)

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


class PlotCwfsPairingTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("exposure", "visit", "instrument"),
):
    exposures = ct.Input(
        doc="Input exposures to plot",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
        multiple=True,
    )
    aggregateAOSRaw = ct.Input(
        doc="AOS raw catalog",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    pairingPlot = ct.Output(
        doc="Wavefront Sensor Donut Pairing Plot",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="pairingPlot",
    )


class PlotCwfsPairingTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotCwfsPairingTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotCwfsPairingTask(pipeBase.PipelineTask):
    ConfigClass = PlotCwfsPairingTaskConfig
    _DefaultName = "plotCwfsPairingTask"

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
        visit = inputRefs.aggregateAOSRaw.dataId["visit"]
        images = dict()
        for exposureRef in inputRefs.exposures:
            exposure = butlerQC.get(exposureRef)
            image = exposure.image.array
            detId = exposure.getDetector().getId()
            images[detId] = image

        camera = butlerQC.get(inputRefs.camera)
        fig = self.run(images, aos_raw, camera, visit)

        # put the plot in butler
        butlerQC.put(fig, outputRefs.pairingPlot)

        # put the plot in RubinTV
        if self.config.doRubinTVUpload:
            inst = inputRefs.aggregateAOSRaw.dataId["instrument"]
            locationConfig = getAutomaticLocationConfig()
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "fp_pairing_plot"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
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
        self,
        images: dict[np.ndarray],
        aggregateAOSRaw: Table,
        camera: lsst.afw.cameraGeom.Camera,
        visit: int,
    ):
        table = aggregateAOSRaw

        # Store image components as a dict
        layout = [
            [192, 191, 200, 200],
            [192, 191, 199, 199],
            [195, 195, 203, 204],
            [196, 196, 203, 204],
        ]
        fig = make_figure(figsize=(10, 10))
        axs = fig.subplot_mosaic(layout)
        for k, ax in axs.items():
            if k in images.keys():
                is_extra = k % 2 == 1
                extra_det = k if is_extra else (k - 1)
                nquarter = camera[k].getOrientation().getNQuarter() % 4
                # handle negative regions in the image for plotting
                arr = np.copy(images[k])
                arr[np.where(arr < 0)] = 0
                ax.imshow(
                    np.rot90(stretchDataMidTone(arr), -nquarter).T, cmap="Greys_r"
                )
                if is_extra:
                    dettable = table[table["detector"] == camera[extra_det].getName()]
                    dx1 = dettable["centroid_x_extra"]
                    dy1 = dettable["centroid_y_extra"]
                    dx2 = dettable["centroid_x_intra"]
                    dy2 = dettable["centroid_y_intra"]
                    if nquarter == 0:
                        dy2, dx2 = 2000 - dy2, 4072 - dx2
                    if nquarter == 1:
                        dy1, dx1 = dx1, 2000 - dy1
                        dy2, dx2 = 4072 - dx2, dy2
                    if nquarter == 2:
                        dy1, dx1 = 2000 - dy1, 4072 - dx1
                    if nquarter == 3:
                        dy1, dx1 = 4072 - dx1, dy1
                        dy2, dx2 = dx2, 2000 - dy2
                    for x1, y1, x2, y2 in zip(dx1, dy1, dx2, dy2):
                        conn = ConnectionPatch(
                            xyA=(y1, x1),
                            xyB=(y2, x2),
                            coordsA=ax.transData,
                            coordsB=axs[k + 1].transData,
                            arrowstyle="-",
                            color="blue",
                            linestyle=":",
                        )
                        fig.add_artist(conn)
                    ax.scatter(
                        dy1,
                        dx1,
                        facecolors="none",
                        edgecolors="blue",
                        s=500,
                        linewidth=1,
                    )
                    axs[k + 1].scatter(
                        dy2,
                        dx2,
                        facecolors="none",
                        edgecolors="blue",
                        s=500,
                        linewidth=1,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                x, y = (0.02, 0.96) if (nquarter % 2) == 0 else (0.01, 0.92)
                ax.text(x, y, str(k), transform=ax.transAxes, color="red", fontsize=13)
            else:
                ax.axis("off")
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1
        )
        fig.suptitle(f"{visit}", fontsize=15)
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
            locationConfig = getAutomaticLocationConfig()
            instrument = inputRefs.zernikes[0].dataId["instrument"]
            visit = inputRefs.zernikes[0].dataId["visit"]
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "psf_zk_panel"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
            )
            zkPanel.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName="psf_zk_panel",
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def run(self, zernikes, **kwargs) -> Figure:
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


class PlotDonutFitsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),
):
    aggregateAOSRaw = ct.Input(
        doc="AOS raw catalog",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
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
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    donutFits = ct.Output(
        doc="Donut Fits",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutFits",
    )


class PlotDonutFitsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotDonutFitsTaskConnections,
):
    doRubinTVUpload = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotDonutFitsTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutFitsTaskConfig
    _DefaultName = "plotDonutFitsTask"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.doRubinTVUpload:
            if not MultiUploader:
                raise RuntimeError("MultiUploader is not available")
            self.uploader = MultiUploader()

        mask_params_fn = Path(danish.datadir) / "RubinObsc.yaml"
        with open(mask_params_fn) as f:
            self.mask_params = yaml.safe_load(f)
        instConfigFile = None
        self.instrument = getTaskInstrument(
            "LSSTCam",
            "R00_SW0",
            instConfigFile,
        )
        self.obsc = self.instrument.obscuration
        self.fL = self.instrument.focalLength
        self.R_outer = self.instrument.radius
        self.wavelengths = self.instrument.wavelength
        self.pixel_scale = self.instrument.pixelSize
        self.factory = danish.DonutFactory(
            R_outer=self.R_outer,
            R_inner=self.R_outer * self.obsc,
            mask_params=self.mask_params,
            focal_length=self.fL,
            pixel_scale=self.pixel_scale,
        )

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Get the inputs
        aos_raw = butlerQC.get(inputRefs.aggregateAOSRaw)
        donutStampsIntra = butlerQC.get(inputRefs.donutStampsIntraVisit)
        donutStampsExtra = butlerQC.get(inputRefs.donutStampsExtraVisit)
        camera = butlerQC.get(inputRefs.camera)
        visit = inputRefs.aggregateAOSRaw.dataId["visit"]
        inputRefs.donutStampsIntraVisit.dataId.records["visit"]
        record = inputRefs.aggregateAOSRaw.dataId.records["visit"]

        day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
        fig = self.run(
            aos_raw,
            donutStampsIntra,
            donutStampsExtra,
            camera,
            day_obs,
            seq_num,
            record,
        )

        butlerQC.put(fig, outputRefs.donutFits)

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            instrument = inputRefs.aggregateAOSRaw.dataId["instrument"]
            plotName = "donut_fits"
            plotFile = makePlotFile(
                locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png"
            )
            fig.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def getModel(self, telescope, defocused_telescope, row, img, wavelength, inex):
        thx = row[f"thx_CCS_{inex}"]
        thy = row[f"thy_CCS_{inex}"]
        zk_CCS = row["zk_CCS"]
        nollIndices = row.meta["nollIndices"]

        z_intrinsic = (
            batoid.zernikeGQ(
                telescope,
                thx,
                thy,
                wavelength,
                jmax=28,
                eps=self.obsc,
            )
            * wavelength
        )
        z_ref = (
            batoid.zernikeTA(
                defocused_telescope,
                thx,
                thy,
                wavelength,
                jmax=78,
                eps=self.obsc,
                focal_length=self.fL,
            )
            * wavelength
        )
        zk = z_ref.copy()
        for ij, j in enumerate(nollIndices):
            zk[j] += zk_CCS[ij] * 1e-6 - z_intrinsic[j]

        fitter = danish.SingleDonutModel(
            self.factory,
            z_ref=zk,
            z_terms=tuple(),  # only fitting x/y/fwhm
            thx=thx,
            thy=thy,
            npix=img.shape[0],
        )

        guess = [0.0, 0.0, 0.7]
        sky_level = 10000  # What!?
        result = least_squares(
            fitter.chi,
            guess,
            jac=fitter.jac,
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
            args=(img, sky_level),
        )
        dx, dy, fwhm = result.x

        model = fitter.model(*result.x, [])
        return model, fwhm

    def plotResults(self, axs, imgs, models, row, blur):
        colors = [
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 1.0),  # White
            (1.0, 0.0, 0.0),  # Red
        ]
        positions = [0.0, 1 / 11, 1.0]
        cmap = LinearSegmentedColormap.from_list(
            "cyan_white_magenta", list(zip(positions, colors))
        )

        vmax = np.nanquantile(imgs[0], 0.99)
        axs[0].imshow(imgs[0], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[0].text(5, 150, f"blur: {blur:5.3f}")
        axs[1].imshow(models[0], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[2].imshow(imgs[0] - models[0], cmap="bwr", vmin=-vmax / 3, vmax=vmax / 3)
        axs[3].imshow(imgs[1], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[4].imshow(models[1], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[5].imshow(imgs[1] - models[1], cmap="bwr", vmin=-vmax / 3, vmax=vmax / 3)
        axs[6].bar(row.meta["nollIndices"], row["zk_CCS"], color="k")
        axs[6].axhline(0, color="k", lw=0.5)
        axs[6].set_ylim(-2.5, 2.5)
        axs[6].set_xlim(3.5, 28.5)
        axs[6].scatter(
            [4, 11, 22], [2.2] * 3, marker="o", ec="k", c="none", s=10, lw=0.5
        )
        axs[6].scatter([7, 17], [2.2] * 2, marker="$\u2191$", c="k", s=10, lw=0.5)
        axs[6].scatter([8, 16], [2.2] * 2, marker="$\u2192$", c="k", s=10, lw=0.5)
        axs[6].scatter([5, 13, 23], [2.2] * 3, marker=(2, 2, 45), c="k", s=10, lw=0.5)
        axs[6].scatter([6, 12, 24], [2.2] * 3, marker=(2, 2, 90), c="k", s=10, lw=0.5)

        axs[6].scatter([9, 19], [2.2] * 2, marker=(3, 2, 60), c="k", s=10, lw=0.5)
        axs[6].scatter([10, 18], [2.2] * 2, marker=(3, 2, 30), c="k", s=10, lw=0.5)

        axs[6].scatter([14, 26], [2.2] * 2, marker=(4, 2), c="k", s=10, lw=0.5)
        axs[6].scatter([15, 25], [2.2] * 2, marker=(4, 2, 22.5), c="k", s=10, lw=0.5)

        axs[6].scatter([20], [2.2], marker=(5, 2, -18), c="k", s=10, lw=0.5)
        axs[6].scatter([21], [2.2], marker=(5, 2), c="k", s=10, lw=0.5)

        axs[6].scatter([27], [2.2], marker=(6, 2, 15), c="k", s=10, lw=0.5)
        axs[6].scatter([28], [2.2], marker=(6, 2), c="k", s=10, lw=0.5)

        for j in [4, 11, 22]:
            axs[6].axvspan(j - 0.5, j + 0.5, color="red", alpha=0.2, ec="none")
        for j in [5, 12, 23]:
            axs[6].axvspan(j - 0.5, j + 1.5, color="orange", alpha=0.2, ec="none")
        for j in [7, 16]:
            axs[6].axvspan(j - 0.5, j + 1.5, color="yellow", alpha=0.2, ec="none")
        for j in [9, 18]:
            axs[6].axvspan(j - 0.5, j + 1.5, color="green", alpha=0.2, ec="none")
        for j in [14, 25]:
            axs[6].axvspan(j - 0.5, j + 1.5, color="blue", alpha=0.2, ec="none")
        axs[6].axvspan(19.5, 21.5, color="indigo", alpha=0.2, ec="none")
        axs[6].axvspan(26.5, 28.5, color="violet", alpha=0.2, ec="none")
        color = "green" if row["used"] else "red"
        axs[6].spines["right"].set_edgecolor(color)
        axs[6].spines["right"].set_linewidth(3)

    def run(
        self,
        aos_raw,
        donutStampsIntra,
        donutStampsExtra,
        camera,
        day_obs,
        seq_num,
        record,
    ) -> Figure:
        """Run the PlotDonutFits AOS task.

        Creates a figure of donuts / models / and residuals.

        Parameters
        ----------
        aos_raw: Astropy Table
            The AOS raw catalog.
        donutStampsIntra: DonutStamps
            The intra-focal donut stamps.
        donutStampsExtra: DonutStamps
            The extra-focal donut stamps.
        camera: Camera
            The camera object to get detector information.
        day_obs: int
            The day of observation.
        seq_num: int
            The sequence number of the observation.
        record: lsst.daf.butler.dimensions._records.visit.RecordClass
            The butler exposure level record

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure.
        """
        bandpass = donutStampsIntra.getBandpasses()[0]
        assert all([bandpass == bp for bp in donutStampsIntra.getBandpasses()])
        assert all([bandpass == bp for bp in donutStampsExtra.getBandpasses()])

        wavelength = self.wavelengths[bandpass]
        telescope = batoid.Optic.fromYaml(f"LSST_{bandpass}.yaml")
        intra_telescope = telescope.withGloballyShiftedOptic(
            "Detector", [0, 0, -1.5e-3]
        )
        extra_telescope = telescope.withGloballyShiftedOptic(
            "Detector", [0, 0, +1.5e-3]
        )

        # Get the trim from EFD: applied corrections
        startTime = record.timespan.begin
        endTime = record.timespan.end
        efd_client = makeEfdClient()
        efd_topic = "lsst.sal.MTAOS.logevent_degreeOfFreedom"
        states_val = np.empty(
            50,
        )
        visit_logevent: int | str = "unknown"
        # catch test data that may have historic day_obs
        if day_obs > 20250101:
            event = getMostRecentRowWithDataBefore(
                efd_client,
                efd_topic,
                timeToLookBefore=Time(startTime, scale="utc"),
            )

            for i in range(50):
                states_val[i] = event[f"aggregatedDoF{i}"]
            if "visitId" in event.keys():
                visit_logevent = event["visitId"]

        # Get the rotator angle
        rotData = getEfdData(
            client=efd_client,
            topic="lsst.sal.MTRotator.rotation",
            begin=startTime,
            end=endTime,
        )

        # Prepare figure
        fig = make_figure(figsize=(16, 11))
        axdict = {}
        gs0 = GridSpec(
            nrows=3,
            ncols=2,
            left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.95,
            wspace=0.04,
            hspace=0.12,
            height_ratios=[2, 2, 1],
        )
        for i, j, raft in [(0, 0, "R00"), (0, 1, "R40"), (1, 0, "R04"), (1, 1, "R44")]:
            gs1 = GridSpecFromSubplotSpec(
                nrows=4,
                ncols=1,
                subplot_spec=gs0[i, j],
                wspace=0.0,
                hspace=0.0,
            )
            axdict[raft] = []
            for k in range(4):
                gs2 = GridSpecFromSubplotSpec(
                    nrows=1,
                    ncols=7,
                    subplot_spec=gs1[k],
                    wspace=0.0,
                    hspace=0.0,
                    width_ratios=[1, 1, 1, 1, 1, 1, 2],
                )
                axs = []
                for ls in range(7):
                    ax = fig.add_subplot(gs2[0, ls])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    axs.append(ax)
                axdict[raft].append(axs)

        bottom_ax = fig.add_subplot(gs0[2, :])
        bottom_ax.set_xticks([])
        bottom_ax.set_yticks([])
        # a single value per donut
        donut_blur = np.zeros(len(aos_raw))
        if "fwhm" in aos_raw.meta["estimatorInfo"].keys():
            donut_blur = np.array(aos_raw.meta["estimatorInfo"].get("fwhm"))

        # Proceed raft by raft
        for iraft, raft in enumerate(["R00", "R04", "R40", "R44"]):
            detname = raft + "_SW0"
            selected_rows = aos_raw["detector"] == detname
            rows = aos_raw[selected_rows]
            blur = donut_blur[selected_rows]
            if len(rows) == 0:
                continue
            det = camera[detname]
            nquarter = det.getOrientation().getNQuarter() % 4

            # add title to each corner
            for defocal, sw, col in zip(["intra", "extra"], ["SW1", "SW0"], [0, 3]):
                raftName = f"{raft}_{sw}"
                detId = camera.get(raftName).getId()
                axdict[raft][0][col].set_title(
                    f"{defocal} {raftName} ({detId})", x=0.95
                )

            # get donuts corresponding to a given corner from
            # aggregatedDonutStamps
            idxToAggIntra = (
                np.array(donutStampsIntra.metadata.getArray("DET_NAME"))
                == f"{raft}_SW1"
            )
            donutStampsIntraSel = np.array(donutStampsIntra)[idxToAggIntra]
            intra_x = [stamp.centroid_position.x for stamp in donutStampsIntraSel]
            intra_y = [stamp.centroid_position.y for stamp in donutStampsIntraSel]

            idxToAggExtra = (
                np.array(donutStampsExtra.metadata.getArray("DET_NAME"))
                == f"{raft}_SW0"
            )
            donutStampsExtraSel = np.array(donutStampsExtra)[idxToAggExtra]
            extra_x = [stamp.centroid_position.x for stamp in donutStampsExtraSel]
            extra_y = [stamp.centroid_position.y for stamp in donutStampsExtraSel]

            for irow, row in enumerate(rows[:4]):
                # intra
                dists = np.hypot(
                    intra_x - row["centroid_x_intra"], intra_y - row["centroid_y_intra"]
                )
                idx = np.argmin(dists)
                # select stamps from the subset of aggregated donuts
                # corresponding to current corner
                intra_stamp = donutStampsIntraSel[idx]
                intra_img = np.rot90(
                    intra_stamp.stamp_im.image.array[1:, 1:], -nquarter + 2
                ).T
                intra_model, intra_fwhm = self.getModel(
                    telescope, intra_telescope, row, intra_img, wavelength, "intra"
                )

                intra_img /= np.sum(intra_img)
                intra_model /= np.sum(intra_model)

                # extra
                dists = np.hypot(
                    extra_x - row["centroid_x_extra"], extra_y - row["centroid_y_extra"]
                )
                idx = np.argmin(dists)
                extra_stamp = donutStampsExtraSel[idx]
                extra_img = np.rot90(
                    extra_stamp.stamp_im.image.array[1:, 1:], -nquarter
                ).T
                extra_model, extra_fwhm = self.getModel(
                    telescope, extra_telescope, row, extra_img, wavelength, "extra"
                )

                extra_img /= np.sum(extra_img)
                extra_model /= np.sum(extra_model)

                self.plotResults(
                    axs=axdict[raft][irow],
                    imgs=[intra_img, extra_img],
                    models=[intra_model, extra_model],
                    row=row,
                    blur=blur[irow],
                )

        def format_group(
            vals, label, wrap_width=2, rigid=False, label_width=20, prec=3, max_int=None
        ):
            """
            Format DOF group for rigid-body or bending modes.
            """
            rows = []

            if rigid:
                # Single-value rigid-body row (decimal-aligned)
                formatted = f"{vals[0]:.{prec}f}"
                ip, fp = formatted.split(".")
                if max_int is None:
                    max_int = len(ip)
                row = f"{label:<{label_width}} {ip:>{max_int}}.{fp}"
                rows.append(row)

            else:
                # Bending-mode formatting (column-major)
                n = len(vals)
                labels = [f"b{i+1}" for i in range(n)]
                formatted = [f"{v:.{prec}f}" for v in vals]
                int_parts = [f.split(".")[0] for f in formatted]
                frac_parts = [f.split(".")[1] for f in formatted]
                max_int_local = max(len(ip) for ip in int_parts)
                max_frac_local = max(len(fp) for fp in frac_parts)

                tokens = [
                    f"{lbl:<3} {ip:>{max_int_local}}.{fp:<{max_frac_local}}"
                    for lbl, ip, fp in zip(labels, int_parts, frac_parts)
                ]

                nrows = int(np.ceil(n / wrap_width))
                padded = tokens + [""] * (nrows * wrap_width - n)
                arr = np.array(padded).reshape(nrows, wrap_width, order="F")

                rows.append(label + ":")
                for row_arr in arr:
                    rows.append("  ".join(f"{cell:<15}" for cell in row_arr).rstrip())

            return rows

        # --- Define groups ---
        rigid_groups = [
            ("M2 dz (microns)", [0]),
            ("M2 dx (microns)", [1]),
            ("M2 dy (microns)", [2]),
            ("M2 rx (deg)", [3]),
            ("M2 ry (deg)", [4]),
            ("Camera dz (microns)", [5]),
            ("Camera dx (microns)", [6]),
            ("Camera dy (microns)", [7]),
            ("Camera rx (deg)", [8]),
            ("Camera ry (deg)", [9]),
        ]

        bending_groups = [
            ("M1M3 bending modes (microns)", list(range(10, 30))),
            ("M2 bending modes (microns)", list(range(30, 50))),
        ]

        bottom_ax.set_frame_on(False)
        bottom_ax.set_title(
            f"{day_obs} seq{seq_num}: current offset from lookup table (based on seq={visit_logevent})"
        )

        # Layout for 4 columns
        col_xpos = [0.05, 0.28, 0.51, 0.72]  # relative x positions in axes coords
        y_start = 0.8
        y_step = 0.07  # tighter spacing so we can fit more

        # --- Precompute max integer width for rigid-body numbers ---
        rigid_vals = [states_val[i] for _, idxs in rigid_groups for i in idxs]
        formatted_all = [f"{v:.3f}" for v in rigid_vals]
        int_parts = [f.split(".")[0] for f in formatted_all]
        max_int_rigid = max(len(ip) for ip in int_parts)

        # Track y position per column separately
        ypos = {0: y_start, 1: y_start, 2: y_start, 3: y_start}

        # --- Render rigid-body groups in column 0 ---
        bottom_ax.text(
            col_xpos[0],
            ypos[0],
            "Rigid body motions",
            transform=bottom_ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            family="monospace",
            weight="bold",
        )

        ypos[0] -= y_step

        # --- Render rigid-body group in first column
        for label, idxs in rigid_groups:
            val = states_val[idxs[0]]
            lines = format_group([val], label, rigid=True, max_int=max_int_rigid)
            for line in lines:
                bottom_ax.text(
                    col_xpos[0],
                    ypos[0],
                    line,
                    transform=bottom_ax.transAxes,
                    fontsize=9,
                    va="top",
                    ha="left",
                    family="monospace",
                )
                ypos[0] -= y_step

        # --- Render bending-mode groups in remaining columns ---
        for i, (label, idxs) in enumerate(bending_groups):
            col = i + 1  # start from column 1
            vals = [states_val[j] for j in idxs]
            lines = format_group(vals, label, wrap_width=2)
            for j, line in enumerate(lines):
                # Bold only the first line (the label)
                is_title = j == 0
                bottom_ax.text(
                    col_xpos[col],
                    ypos[col],
                    line,
                    transform=bottom_ax.transAxes,
                    fontsize=9,
                    va="top",
                    ha="left",
                    family="monospace",
                    weight="bold" if is_title else "normal",
                )
                ypos[col] -= y_step

        # Plot the exposure record data
        records = {
            "filter": record.physical_filter,
            "observation reason": record.observation_reason,
            "science program": record.science_program,
            "elevation": (
                90 if record.zenith_angle is None else 90 - record.zenith_angle
            ),
            "azimuth": 0 if record.azimuth is None else record.azimuth,
            "rotator": (
                0 if len(rotData) == 0 else rotData["actualPosition"].values.mean()
            ),
        }
        col = 3

        # Decide which keys are floats that should be decimal-aligned
        float_keys = {"elevation", "azimuth", "rotator"}

        # Format values so floats are aligned
        formatted_records = {}
        for k, v in records.items():
            if k in float_keys:
                # Format floats with consistent width + alignment to decimal
                formatted_records[k] = f"{v:7.3f}"
            else:
                formatted_records[k] = str(v)

        # Figure out widest key string for alignment
        key_width = max(len(k) for k in formatted_records.keys())

        for key, val in formatted_records.items():
            # Pad key names to same width, keep monospace look
            txt = f"{key.ljust(key_width)} : {val}"
            bottom_ax.text(
                col_xpos[col],
                ypos[col],
                txt,
                transform=bottom_ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                family="monospace",
            )
            ypos[col] -= y_step

        return fig
