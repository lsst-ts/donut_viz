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
from astropy.table import Table
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.afw.cameraGeom import Camera
from lsst.summit.utils.plotting import stretchDataMidTone
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod

from .utilities import (
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
    "PlotCwfsPairingTaskConnections",
    "PlotCwfsPairingTaskConfig",
    "PlotCwfsPairingTask",
]


class PlotCwfsPairingTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("exposure", "visit", "instrument"),  # type: ignore
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
    pipelineConnections=PlotCwfsPairingTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotCwfsPairingTask(pipeBase.PipelineTask):
    ConfigClass = PlotCwfsPairingTaskConfig
    _DefaultName = "plotCwfsPairingTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotCwfsPairingTaskConfig = cast(PlotCwfsPairingTaskConfig, self.config)

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
    def run(
        self,
        images: dict,
        aggregateAOSRaw: Table,
        camera: Camera,
        visit: int,
    ) -> Figure:
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
                ax.imshow(np.rot90(stretchDataMidTone(arr), -nquarter).T, cmap="Greys_r")
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
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
        fig.suptitle(f"{visit}", fontsize=15)
        return fig
