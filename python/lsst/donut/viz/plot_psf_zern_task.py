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
from astropy import units as u
from astropy.table import Table
from matplotlib.figure import Figure

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.ts.wep.utils import convertZernikesToPsfWidth
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod

from .psf_from_zern import psfPanel
from .utilities import (
    add_coordinate_roses,
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
    "PlotPsfZernTaskConnections",
    "PlotPsfZernTaskConfig",
    "PlotPsfZernTask",
]


class PlotPsfZernTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
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
    pipelineConnections=PlotPsfZernTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )


class PlotPsfZernTask(pipeBase.PipelineTask):
    ConfigClass = PlotPsfZernTaskConfig
    _DefaultName = "plotPsfZernTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotPsfZernTaskConfig = cast(PlotPsfZernTaskConfig, self.config)

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
        zernikes = butlerQC.get(inputRefs.zernikes)

        zkPanel = self.run(zernikes, figsize=(11, 14))

        butlerQC.put(zkPanel, outputRefs.psfFromZernPanel)

        if self.config.doRubinTVUpload:
            locationConfig = getAutomaticLocationConfig()
            instrument = inputRefs.zernikes[0].dataId["instrument"]
            visit = inputRefs.zernikes[0].dataId["visit"]
            day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)

            plotName = "psf_zk_panel"
            plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
            zkPanel.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName="psf_zk_panel",
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def run(self, zernikes: list[Table], **kwargs: Any) -> Figure:
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

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        dname: list[str] = []
        angles_set = False
        for i, qt in enumerate(zernikes):
            if len(qt) == 0:
                zs.append(np.array([]))
                dname.append("")
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
            [np.sqrt(np.sum(convertZernikesToPsfWidth(pair_zset) ** 2)) for pair_zset in det] for det in zs
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
