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
from matplotlib.figure import Figure

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
    "PlotDonutCwfsTaskConnections",
    "PlotDonutCwfsTaskConfig",
    "PlotDonutCwfsTask",
]


class PlotDonutCwfsTaskConnections(
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
    donutPlot = ct.Output(
        doc="Donut Plot",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutPlot",
    )


class PlotDonutCwfsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotDonutCwfsTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotDonutCwfsTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutCwfsTaskConfig
    _DefaultName = "plotDonutCwfsTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotDonutCwfsTaskConfig = cast(PlotDonutCwfsTaskConfig, self.config)

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

        fig = self.run(donutStampsIntra, donutStampsExtra, inst)

        butlerQC.put(fig, outputRefs.donutPlot)

        # Same visit for both extra and intra-focal corner sensors
        visit = donutStampsIntra.metadata.getArray("VISIT")[0]

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "fp_donut_gallery"
            plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
            fig.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(inst),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    @timeMethod
    def run(self, donutStampsIntra: DonutStamps, donutStampsExtra: DonutStamps, inst: str) -> Figure:
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

        i: float
        j: float
        nrot90: int
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
