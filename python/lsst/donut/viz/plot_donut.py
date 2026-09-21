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

from typing import Any, cast

import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.ts.wep.task import DonutStamps
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod

from .utilities import (
    add_coordinate_roses,
    add_rotated_axis,
    get_day_obs_seq_num_from_visitid,
    get_instrument_channel_name,
)

try:
    from lsst.rubintv.production.formatters import makePlotFile
    from lsst.rubintv.production.locationConfig import getAutomaticLocationConfig
    from lsst.rubintv.production.uploaders import MultiUploader
except ImportError:
    MultiUploader = None  # type: ignore[assignment,misc]

__all__ = [
    "PlotDonutTaskConnections",
    "PlotDonutTaskConfig",
    "PlotDonutTask",
]


class PlotDonutTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
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
    pipelineConnections=PlotDonutTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )
    doS11only: pexConfig.Field = pexConfig.Field(dtype=bool, doc="Use only S11 in FAM mode", default=False)


class PlotDonutTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutTaskConfig
    _DefaultName = "plotDonutTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotDonutTaskConfig = cast(PlotDonutTaskConfig, self.config)

        if self.config.doRubinTVUpload:
            if MultiUploader is None:
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
        fig_struct = self.run(donutStampsIntra, donutStampsExtra, inst)

        butlerQC.put(fig_struct.extra, outputRefs.donutPlotExtra)
        butlerQC.put(fig_struct.intra, outputRefs.donutPlotIntra)

        visitIntra = donutStampsIntra.metadata.getArray("VISIT")[0]
        visitExtra = donutStampsExtra.metadata.getArray("VISIT")[0]

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            # seq_num is sometimes different for
            # intra vs extra-focal if pistoning
            for defocal_type, visit_id in zip(["extra", "intra"], [visitExtra, visitIntra]):
                day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit_id)

                plotName = "fp_donut_gallery"
                plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
                if defocal_type == "extra":
                    fig_struct.extra.savefig(plotFile)
                else:
                    fig_struct.intra.savefig(plotFile)
                self.uploader.uploadPerSeqNumPlot(
                    instrument=get_instrument_channel_name(inst),
                    plotName=plotName,
                    dayObs=day_obs,
                    seqNum=seq_num,
                    filename=plotFile,
                )

    @timeMethod
    def run(self, donutStampsIntra: DonutStamps, donutStampsExtra: DonutStamps, inst: str) -> pipeBase.Struct:
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

        for donutStampSet, visit in zip([donutStampsIntra, donutStampsExtra], [visitIntra, visitExtra]):
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

        return pipeBase.Struct(extra=fig_dict["extra"], intra=fig_dict["intra"])
