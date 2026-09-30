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

from copy import copy
from pathlib import Path
from typing import Any, cast

import galsim
import numpy as np
import yaml
from astropy.table import Table
from matplotlib.figure import Figure

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.utils.timer import timeMethod

from .utilities import (
    get_day_obs_seq_num_from_visitid,
    get_instrument_channel_name,
    rose,
)
from .zernike_pyramid import zernikePyramid

try:
    from lsst.rubintv.production.formatters import makePlotFile
    from lsst.rubintv.production.locationConfig import getAutomaticLocationConfig
    from lsst.rubintv.production.uploaders import MultiUploader
except ImportError:
    MultiUploader = None  # type: ignore[assignment,misc]

__all__ = [
    "PlotAOSTaskConnections",
    "PlotAOSTaskConfig",
    "PlotAOSTask",
]


class PlotAOSTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
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
    pipelineConnections=PlotAOSTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )
    shiftFactor: pexConfig.Field = pexConfig.Field(
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotAOSTaskConfig = cast(PlotAOSTaskConfig, self.config)

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
            plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
            zkPyramid.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

            plotName = "zk_residual_pyramid"
            plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
            residPyramid.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def doPyramid(
        self,
        x: float,
        y: float,
        zk: np.ndarray,
        rtp: float,
        q: float,
        nollIndices: np.ndarray,
    ) -> Figure:
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
        aos_raw: Table,
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
        if np.sum(["SW" in detName for detName in np.unique(aos_raw["detector"].value)]) > 0:
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
        intrinsic = np.array([z.coef for z in dzs(aos_raw["thx_OCS"], aos_raw["thy_OCS"])]).T[4:29]
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
