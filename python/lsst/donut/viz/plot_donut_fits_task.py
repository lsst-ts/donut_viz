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

from pathlib import Path
from typing import Any, cast

import danish
import galsim
import numpy as np
import yaml
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.time import Time
from galsim import GalSimFFTSizeError, GalSimRangeError
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from packaging import version as pkg_version

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.afw.cameraGeom import Camera
from lsst.daf.butler.dimensions import DimensionRecord
from lsst.summit.utils.efdUtils import (
    getEfdData,
    getMostRecentRowWithDataBefore,
    makeEfdClient,
)
from lsst.ts.wep.estimation import DanishAlgorithm
from lsst.ts.wep.task import DonutStamp, DonutStamps
from lsst.ts.wep.utils import getTaskInstrument
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
    "PlotDonutFitsTaskConnections",
    "PlotDonutFitsTaskConfig",
    "PlotDonutFitsTask",
]


class PlotDonutFitsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
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
    pipelineConnections=PlotDonutFitsTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )
    nDonutsPerCorner: pexConfig.Field = pexConfig.Field(
        dtype=int,
        doc="Number of donuts per corner (integer, default: 8).\
This sets the number of rows per corner in the figure layout.",
        default=8,
    )
    zkYmin: pexConfig.Field = pexConfig.Field(
        dtype=float, doc="Lower limit on Zernike plot (default: -1 micron).", default=-1
    )
    zkYmax: pexConfig.Field = pexConfig.Field(
        dtype=float, doc="Upper limit on Zernike plot (default: +1 micron).", default=1
    )


class PlotDonutFitsTask(pipeBase.PipelineTask):
    ConfigClass = PlotDonutFitsTaskConfig
    _DefaultName = "plotDonutFitsTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotDonutFitsTaskConfig = cast(PlotDonutFitsTaskConfig, self.config)

        galsim.errors.raise_fft_size_error = True

        if self.config.doRubinTVUpload:
            if MultiUploader is None:
                raise RuntimeError("MultiUploader is not available")
            self.uploader = MultiUploader()

        self.efd_client = makeEfdClient()

        mask_params_fn = Path(danish.datadir) / "RubinObsc.yaml"
        with open(mask_params_fn) as f:
            self.mask_params = yaml.safe_load(f)
        instConfigFile = None
        self.instrument = getTaskInstrument(
            "LSSTCam",
            "R00_SW0",
            instConfigFile,
        )

        # Setup danish algo
        self.danish_algo = DanishAlgorithm()
        self.danish_model_keys = ["fit_success", "fwhm", "model_bkg", "model_dx", "model_dy", "model_flux"]

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
            plotFile = makePlotFile(locationConfig, "LSSTCam", day_obs, seq_num, plotName, "png")
            fig.savefig(plotFile)
            self.uploader.uploadPerSeqNumPlot(
                instrument=get_instrument_channel_name(instrument),
                plotName=plotName,
                dayObs=day_obs,
                seqNum=seq_num,
                filename=plotFile,
            )

    def getModel(
        self,
        zk_deviation_ccs: np.ndarray,
        zk_intrinsic_ccs: np.ndarray,
        noll_indices: list[int],
        danish_meta: dict,
        stamp_extra: DonutStamp,
        stamp_intra: DonutStamp,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate the danish donut model images
        for a pair of extra- and intra-focal donuts.

        Parameters
        ----------
        zk_deviation_ccs : `numpy.ndarray`
            Zernike deviation coefficients in microns in CCS coordinates.
        zk_intrinsic_ccs : `numpy.ndarray`
            Intrinsic Zernike coefficients in microns in CCS coordinates.
        noll_indices : `list` of `int`
            List of Noll indices corresponding to the Zernike coefficients.
        danish_meta : `dict`
            Metadata required by the danish model. All of this metadata is
            available as included metadata with aggregateAOSVisitTableRaw.
            Includes:
            - model_dx: list(float, float)
            - model_dy: list(float, float)
            - fwhm: float
            - model_sky_level: list(float, float)
        stamp_extra : `DonutStamp`
            Donut stamp for the extra-focal donut.
        stamp_intra : `DonutStamp`
            Donut stamp for the intra-focal donut.

        Returns
        -------
        input_images : `list` of `numpy.ndarray`
            List of images input to danish in proper CCS
            orientation and returned as [extra-focal image, intra-focal image].
        model_images : `list` of `numpy.ndarray`
            List of model images returned as
            [extra-focal model image, intra-focal model image].
        """
        necessary_keys = set(self.danish_model_keys)
        if set(danish_meta.keys()) & necessary_keys != necessary_keys:
            raise ValueError(
                f"danish_meta must contain the following keys: {sorted(necessary_keys)}, "
                f"but only contains: {set(danish_meta.keys())}"
            )

        zk_deviation_CCS = zk_deviation_ccs * 1e-6  # convert to meters
        zk_intrinsic_CCS = zk_intrinsic_ccs * 1e-6  # convert to meters
        dz_terms = [(1, j) for j in noll_indices]
        wep_im_extra = stamp_extra.wep_im
        wep_im_intra = stamp_intra.wep_im

        img_extra, angle_extra, zkRef_extra, backgroundStd_extra = self.danish_algo._prepDanish(
            image=wep_im_extra,
            zkStart=zk_intrinsic_CCS,
            nollIndices=noll_indices,
            instrument=self.instrument,
        )
        img_intra, angle_intra, zkRef_intra, backgroundStd_intra = self.danish_algo._prepDanish(
            image=wep_im_intra,
            zkStart=zk_intrinsic_CCS,
            nollIndices=noll_indices,
            instrument=self.instrument,
        )
        input_images = [img_extra, img_intra]

        if danish_meta["fit_success"] <= 0:
            self.log.warning("Original Danish fit was not successful, returning empty model images")
            model_images = [np.zeros_like(img_extra), np.zeros_like(img_intra)]
            return input_images, model_images

        zk_fit = zk_deviation_CCS - zk_intrinsic_CCS

        nbkg = danish_meta["model_bkg"].shape[1]
        bkg_order = int(np.sqrt(9 + 8 * (nbkg - 1)) - 3) // 2
        model = danish.DZMultiDonutModel(
            self.factory,
            z_refs=[zkRef_extra, zkRef_intra],
            dz_terms=dz_terms,
            field_radius=np.deg2rad(1.81),
            thxs=[angle_extra[0], angle_intra[0]],
            thys=[angle_extra[1], angle_intra[1]],
            npix=img_extra.shape[0],
            bkg_order=bkg_order,
        )

        # need inner dimension to be type other than ndarray for the cache
        bkgs = [tuple(bkg) for bkg in danish_meta["model_bkg"]]
        try:
            model_images = model.model(
                danish_meta["model_flux"],
                danish_meta["model_dx"],
                danish_meta["model_dy"],
                danish_meta["fwhm"],
                zk_fit,
                bkgs=bkgs,
            )
        except (GalSimFFTSizeError, GalSimRangeError, ValueError) as e:
            if isinstance(
                e, (GalSimFFTSizeError, GalSimRangeError)
            ) or "cannot convert float NaN to integer" in str(e):
                self.log.warning(f"Returning empty model images due to following galsim error: {str(e)}")
            else:
                raise e

            # If the model fails to generate due to known error,
            # we return empty model images
            model_images = [np.zeros_like(img_extra), np.zeros_like(img_intra)]

        return input_images, model_images

    def computeResidualStats(self, img: np.ndarray, model: np.ndarray) -> tuple[float, float, float]:
        """
        Compute summary statistics of the residuals between an
        image and the model.

        Parameters
        ----------
        img : ndarray
            The observed image array.
        model : ndarray
            The model image array of the same shape as `img`.

        Returns
        -------
        pos_res : float
            Sum of the absolute values of positive residuals
            (where `img - model > 0`).
        neg_res : float
            Sum of the absolute values of negative residuals
            (where `img - model < 0`).
        tot_res : float
            Sum of the absolute values of all residuals.

        Notes
        -----
        Residuals are computed as `res = img - model`. This function
        provides a simple way to quantify asymmetry between positive
        and negative deviations as well as the overall magnitude of
        the residuals.
        """
        # Compute residual stats
        res = img - model
        pos_res = np.sum(np.abs(res[res > 0]))
        neg_res = np.sum(np.abs(res[res < 0]))
        tot_res = np.sum(np.abs(res))
        return pos_res, neg_res, tot_res

    def plotResults(
        self,
        axs: list,
        imgs: list[np.ndarray],
        models: list[np.ndarray],
        row: Table,
        blur: float,
    ) -> None:
        colors = [
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 1.0),  # White
            (1.0, 0.0, 0.0),  # Red
        ]
        positions = [0.0, 1 / 11, 1.0]
        cmap = LinearSegmentedColormap.from_list("cyan_white_magenta", list(zip(positions, colors)))

        vmax = np.nanquantile(imgs[0], 0.99)
        axs[0].imshow(imgs[0], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[0].text(
            0.05, 0.05, f"blur: {blur:5.3f}", transform=axs[0].transAxes, fontsize="small", va="bottom"
        )
        axs[1].imshow(models[0], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[1].text(
            0.05,
            0.05,
            f"id:  {row['intra_donut_id'][-3:]}",
            transform=axs[1].transAxes,
            fontsize="small",
            va="bottom",
        )
        axs[2].imshow(imgs[0] - models[0], cmap="bwr", vmin=-vmax / 3, vmax=vmax / 3)
        _, _, ttl_res = self.computeResidualStats(imgs[0], models[0])
        axs[2].text(
            0.05, 0.05, f"res:  {ttl_res:5.3f}", transform=axs[2].transAxes, fontsize="small", va="bottom"
        )
        axs[3].imshow(imgs[1], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[4].imshow(models[1], cmap=cmap, vmin=-vmax / 10, vmax=vmax)
        axs[4].text(
            0.05,
            0.05,
            f"id:  {row['extra_donut_id'][-3:]}",
            transform=axs[4].transAxes,
            fontsize="small",
            va="bottom",
        )
        axs[5].imshow(imgs[1] - models[1], cmap="bwr", vmin=-vmax / 3, vmax=vmax / 3)
        _, _, ttl_res = self.computeResidualStats(imgs[1], models[1])
        axs[5].text(
            0.05, 0.05, f"res:  {ttl_res:5.3f}", transform=axs[5].transAxes, fontsize="small", va="bottom"
        )
        axs[6].bar(row.meta["nollIndices"], row["zk_CCS"], color="k")
        axs[6].axhline(0, color="k", lw=0.5)
        axs[6].set_ylim(self.config.zkYmin, self.config.zkYmax)
        axs[6].set_xlim(3.5, 28.5)
        axs[6].scatter([4, 11, 22], [2.2] * 3, marker="o", ec="k", c="none", s=10, lw=0.5)
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
        color = "gray"
        if "used" in row.columns:
            color = "green" if row["used"] else "red"
        axs[6].spines["right"].set_edgecolor(color)
        axs[6].spines["right"].set_linewidth(3)

    @staticmethod
    def _get_rtp(donutStamps: DonutStamps | None) -> Angle:
        if not donutStamps:
            return Angle(np.nan, "rad")
        metadata = donutStamps.metadata
        try:
            rsp = metadata["BORESIGHT_ROT_ANGLE_RAD"]
            q = metadata["BORESIGHT_PAR_ANGLE_RAD"]
        except KeyError:
            return Angle(np.nan, "rad")
        return Angle(q - rsp - np.pi / 2, "rad")

    def run(
        self,
        aos_raw: Table,
        donutStampsIntra: DonutStamps,
        donutStampsExtra: DonutStamps,
        camera: Camera,
        day_obs: int,
        seq_num: int,
        record: DimensionRecord | None,
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

        Raises
        ------
        RuntimeError
            If `record` is None.
        """
        ndonuts = self.config.nDonutsPerCorner

        # All stamps in the visit share a single bandpass; either stamp set
        # may be empty when no donuts were used on that side of the pairs.
        bandpasses = set(donutStampsIntra.getBandpasses()) | set(donutStampsExtra.getBandpasses())
        assert len(bandpasses) <= 1
        noll_indices = aos_raw.meta["nollIndices"]

        obsc = self.instrument.obscuration
        focal_length = self.instrument.focalLength
        r_outer = self.instrument.radius
        pixel_scale = self.instrument.pixelSize
        rtp = self._get_rtp(donutStampsExtra)
        danish_version = pkg_version.parse(danish.__version__)
        min_triangle_mode_version = pkg_version.parse("1.2")
        if danish_version >= min_triangle_mode_version:
            self.log.info(f"Using danish version {danish.__version__} with triangle mode support")
            factory_class = danish.DonutTriangleFactory
        else:
            self.log.info(f"Using danish version {danish.__version__} without triangle mode support")
            factory_class = danish.DonutFactory
        self.factory = factory_class(
            R_outer=r_outer,
            R_inner=r_outer * obsc,
            mask_params=self.mask_params,
            focal_length=focal_length,
            pixel_scale=pixel_scale,
            spider_angle=rtp.deg,
        )

        # Get the trim from EFD: applied corrections
        if record is None:
            raise RuntimeError("record is required to get timespan for EFD query")
        startTime = record.timespan.begin
        endTime = record.timespan.end
        efd_topic = "lsst.sal.MTAOS.logevent_degreeOfFreedom"
        # Default to zeros (no applied correction) so that a missing topic or
        # missing data degrades gracefully rather than plotting garbage.
        states_val = np.zeros(50)
        visit_logevent: int | str = "unknown"
        # catch test data that may have historic day_obs
        if day_obs > 20250101:
            try:
                event = getMostRecentRowWithDataBefore(
                    self.efd_client,
                    efd_topic,
                    timeToLookBefore=Time(startTime, scale="utc"),
                )
                for i in range(50):
                    states_val[i] = event[f"aggregatedDoF{i}"]
                if "visitId" in event.keys():
                    visit_logevent = event["visitId"]
            except ValueError as e:
                self.log.warning(f"Could not get {efd_topic} from EFD, using zeroed corrections: {e}")

        # Get the rotator angle. Allow a missing topic to return an empty
        # DataFrame, which is handled downstream when reading the position.
        rotData = getEfdData(
            client=self.efd_client,
            topic="lsst.sal.MTRotator.rotation",
            begin=startTime,
            end=endTime,
            raiseIfTopicNotInSchema=False,
        )
        # Prepare figure
        # number of rafts per column and rows
        ncols_donut = 7  # 7 columns per donut
        # 3 for intra-focal  image / model / residuals,
        # 3 for extra-focal  image / model / residuals,
        # 1 for Zernike fit result
        # desired square size per cell (in inches)
        cell_size = 1.0  # tweak as needed

        # total width per raft (7 columns)
        raft_width = ncols_donut * cell_size
        raft_height = ndonuts * cell_size
        # since we have 2 rafts per row in the top 2 rows
        fig_height = 2 * raft_height + 3.8  # + space for bottom panel
        fig_width = 2 * raft_width

        fig = make_figure(figsize=(fig_width, fig_height))
        axdict: dict = {}
        gs0 = GridSpec(
            nrows=4,
            ncols=2,
            left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.95,
            wspace=0.04,
            hspace=0.12,
            height_ratios=[4, 4, 0.5, 2],
        )
        for i, j, raft in [(0, 0, "R00"), (0, 1, "R40"), (1, 0, "R04"), (1, 1, "R44")]:
            gs1 = GridSpecFromSubplotSpec(
                nrows=ndonuts,
                ncols=1,
                subplot_spec=gs0[i, j],
                wspace=0.0,
                hspace=0.0,
            )
            axdict[raft] = []
            for k in range(ndonuts):
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

        middle_ax = fig.add_subplot(gs0[2, :])
        middle_ax.set_xticks([])
        middle_ax.set_yticks([])

        bottom_ax = fig.add_subplot(gs0[3, :])
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
            binning = None
            if len(rows) == 0:
                continue
            if len(donutStampsIntra) == 0 or len(donutStampsExtra) == 0:
                # No aggregated donut stamps on one side (e.g. no donut pairs
                # were used in this visit, leaving only NaN placeholder rows
                # in the aggregate), so there is nothing to plot for this
                # corner; leave the panels blank. The metadata arrays used
                # below don't exist on empty stamps.
                continue

            # add title to each corner
            for defocal, sw, col in zip(["intra", "extra"], ["SW1", "SW0"], [0, 3]):
                raftName = f"{raft}_{sw}"
                detId = camera.get(raftName).getId()
                axdict[raft][0][col].set_title(f"{defocal} {raftName} ({detId})", x=0.95)

            # get donuts corresponding to a given corner from
            # aggregatedDonutStamps
            idxToAggIntra = np.array(donutStampsIntra.metadata.getArray("DET_NAME")) == f"{raft}_SW1"
            donutStampsIntraSel = np.array(donutStampsIntra)[idxToAggIntra]
            intra_id = np.array([stamp.donut_id for stamp in donutStampsIntraSel])

            idxToAggExtra = np.array(donutStampsExtra.metadata.getArray("DET_NAME")) == f"{raft}_SW0"
            donutStampsExtraSel = np.array(donutStampsExtra)[idxToAggExtra]
            extra_id = np.array([stamp.donut_id for stamp in donutStampsExtraSel])

            # Grab the metadata for the selected rows
            raft_meta = {
                key: np.array(value)[selected_rows]
                for key, value in aos_raw.meta["estimatorInfo"].items()
                if isinstance(value, (list, tuple, np.ndarray))
            }

            # catching the case when we may wish to plot 8 donuts,
            # but the aggregated table has less than that
            nrows_plot = min(ndonuts, len(rows), len(donutStampsIntraSel), len(donutStampsExtraSel))

            for irow, row in enumerate(rows[:nrows_plot]):
                # intra
                intra_match = np.where(intra_id == row["intra_donut_id"])[0]
                # extra
                extra_match = np.where(extra_id == row["extra_donut_id"])[0]
                if len(intra_match) == 0 or len(extra_match) == 0:
                    self.log.warning(
                        f"No model plot produced for {raft}, donut index: {irow}. "
                        + "Could not find aggregated donut stamps matching donut ids "
                        + f"intra: {row['intra_donut_id']}, extra: {row['extra_donut_id']}."
                    )
                    continue
                # select stamps from the subset of aggregated donuts
                # corresponding to current corner
                intra_stamp = donutStampsIntraSel[intra_match[0]]
                extra_stamp = donutStampsExtraSel[extra_match[0]]

                necessary_keys = set(self.danish_model_keys)
                available_keys = set(row.meta["estimatorInfo"].keys())
                if available_keys & necessary_keys != necessary_keys:
                    missing_keys = necessary_keys - available_keys
                    self.log.warning(
                        f"No model plot produced for {raft}, donut index: {irow}. "
                        + "Required metadata for danish model not found in aggregateAOSVisitTableRaw. "
                        + f"Missing keys: {sorted(missing_keys)}"
                    )
                    continue
                elif (np.isnan(row["zk_deviation_CCS"]).any()) or (np.isnan(row["zk_intrinsic_CCS"]).any()):
                    self.log.warning(
                        f"NaN values found in zk_deviation_CCS or zk_intrinsic_CCS for {raft}, "
                        + f"donut index: {irow}. "
                        + "Skipping model plot production."
                    )
                    continue

                danish_meta = {key: value[irow] for key, value in raft_meta.items()}
                if "model_img" in danish_meta.keys():
                    self.log.info(f"Using precomputed model images for {raft}, donut index: {irow}")
                    if binning is None:
                        if "binning" in aos_raw.meta["estimatorInfo"].keys():
                            binning = int(danish_meta["binning"])
                        else:
                            binning = int(
                                extra_stamp.wep_im.image.shape[0] / danish_meta["model_img"][0].shape[0]
                            )
                    self.danish_algo.binning = binning

                    # Accept the whole stamp for plotting, don't try to mask
                    extra_stamp.wep_im.maskBackground = np.ones(np.shape(extra_stamp.wep_im.image))
                    intra_stamp.wep_im.maskBackground = np.ones(np.shape(intra_stamp.wep_im.image))
                    img_extra, backgroundStd_extra = self.danish_algo.prepImage(
                        image=extra_stamp.wep_im,
                        zkStart=row["zk_intrinsic_CCS"],
                        instrument=self.instrument,
                    )
                    img_intra, backgroundStd_intra = self.danish_algo.prepImage(
                        image=intra_stamp.wep_im,
                        zkStart=row["zk_intrinsic_CCS"],
                        instrument=self.instrument,
                    )
                    imgs = [img_extra, img_intra]
                    model_imgs = [danish_meta["model_img"][0], danish_meta["model_img"][1]]
                else:
                    self.log.info(f"Computing model images for {raft}, donut index: {irow}")
                    imgs, model_imgs = self.getModel(
                        row["zk_deviation_CCS"],
                        row["zk_intrinsic_CCS"],
                        noll_indices,
                        danish_meta,
                        extra_stamp,
                        intra_stamp,
                    )
                for img in imgs:
                    img[np.where(img < 0)] = 0
                extra_img = imgs[0]
                intra_img = imgs[1]
                extra_model = model_imgs[0]
                intra_model = model_imgs[1]

                if np.sum(intra_img) > 0:
                    intra_img /= np.sum(intra_img)
                if np.sum(intra_model) > 0:
                    intra_model /= np.sum(intra_model)

                if np.sum(extra_img) > 0:
                    extra_img /= np.sum(extra_img)
                if np.sum(extra_model) > 0:
                    extra_model /= np.sum(extra_model)

                self.plotResults(
                    axs=axdict[raft][irow],
                    imgs=[intra_img, extra_img],
                    models=[intra_model, extra_model],
                    row=row,
                    blur=blur[irow],
                )
            # turn off unused axes
            if nrows_plot < len(axdict[raft]):
                for j in range(nrows_plot, len(axdict[raft])):
                    for ax in axdict[raft][j]:
                        ax.axis("off")

        # add middle-axis text with the average Zernike value per corner

        # ---  Filter and average per detector ---

        # If information whether a given pair was used in Zernike average
        # is not present in the aggregate, print average of all Zernikes
        # That way we do not need another required input
        # (such as `aggregateAOSVisitTableAvg`)
        mask = aos_raw["used"] if "used" in aos_raw.columns else np.ones(len(aos_raw), dtype="bool")
        aos_used = aos_raw[mask]

        detectors = np.unique(aos_used["detector"])

        if "zk_deviation_CCS" in aos_raw.colnames:
            zkColname = "zk_deviation_CCS"
            zkTableTitle = "(deviation from intrinsic wavefront in CCS)"
        else:
            zkColname = "zk_CCS"
            zkTableTitle = "(as-measured in CCS, including intrinsics)"

        # Set name of created average column
        zkMeanColname = "zk_CCS_mean"
        rows = []
        for det in detectors:
            sel = aos_used["detector"] == det
            mean_zk = np.mean(aos_used[zkColname][sel], axis=0)
            rows.append((det, mean_zk))

        tab_avg = Table(rows=rows, names=("detector", zkMeanColname))

        # --- Get the Zernike indices (column labels) ---
        nollIndices = aos_raw.meta["nollIndices"]

        # Limit to first 15 indices (e.g., Z4–Z19)
        max_cols = 26
        nollIndices = nollIndices[:max_cols]
        col_labels = [f"Z{n}" for n in nollIndices]  # eg. Z4, Z15
        row_labels = [det[:3] for det in detectors]  # eg. R00, R40

        # --- Prepare data for table display ---
        table_data = []
        for det in detectors:
            mean_zk = tab_avg[zkMeanColname][tab_avg["detector"] == det][0]
            mean_zk = mean_zk[:max_cols]
            row = [f"{v:7.3f}" for v in mean_zk]  # format to 3 decimals
            table_data.append(row)

        # --- Create the table inside the middle_ax ---
        # If there are no used donut pairs in the visit there is nothing to
        # average, so leave the table area blank (matplotlib cannot draw a
        # table with no rows).
        if table_data:
            bbox = [0.05, 0.05, 0.85, 0.85]  # [xmin, ymin, width, height],
            table = middle_ax.table(
                cellText=table_data,
                rowLabels=row_labels,
                colLabels=col_labels,
                loc="left",
                cellLoc="center",
                bbox=bbox,
            )

            # Adjust table style
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Set monospace font for all cells FIRST
            for (row, col_idx), cell in table.get_celld().items():
                cell.get_text().set_fontfamily("monospace")
                # Center align all data cells
                if row > 0 and col_idx >= 0:  # data cells only
                    cell.get_text().set_horizontalalignment("center")

            # Loop through all cells
            for (row, col_idx), cell in table.get_celld().items():
                # Hide all lines first
                cell.visible_edges = ""

                # Keep horizontal line below header (row == 0)
                if row == 0:
                    cell.visible_edges += "B"  # bottom border

                # Keep vertical line after row labels (col == -1)
                if col_idx == -1:
                    cell.visible_edges += "R"  # right border

            # Move all data rows slightly down
            for row in range(1, len(row_labels) + 1):  # row=1..4 (data rows)
                for col in range(-1, len(col_labels)):
                    cell = table[(row, col)]
                    text = cell.get_text()
                    text.set_verticalalignment("top")  # align to top of cell
                    cell.set_height(cell.get_height() * 0.8)  # shrink cell to enhance offset

            # Adjust row label alignment and padding
            for row in range(1, len(row_labels) + 1):
                cell = table[(row, -1)]
                text = cell.get_text()
                text.set_horizontalalignment("right")  # right-align within cell
                cell.PAD = 0.2  # add more padding

        # Hide the axis frame
        middle_ax.axis("off")
        middle_ax.set_title(
            f"Average Zernike Coefficients per Detector {zkTableTitle}",
            fontsize=12,
            pad=10,
        )

        def format_group(
            vals: list[float],
            label: str,
            wrap_width: int = 2,
            rigid: bool = False,
            label_width: int = 20,
            prec: int = 3,
            max_int: int | None = None,
        ) -> list[str]:
            """
            Format DOF group for rigid-body or bending modes.
            """
            rows = []

            if rigid:
                # Single-value rigid-body row (decimal-aligned)
                formatted_str = f"{vals[0]:.{prec}f}"
                ip, fp = formatted_str.split(".")
                if max_int is None:
                    max_int = len(ip)
                row = f"{label:<{label_width}} {ip:>{max_int}}.{fp}"
                rows.append(row)

            else:
                # Bending-mode formatting (column-major)
                n = len(vals)
                labels = [f"b{i + 1}" for i in range(n)]
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
            ("M2 rx (asec)", [3]),
            ("M2 ry (asec)", [4]),
            ("Camera dz (microns)", [5]),
            ("Camera dx (microns)", [6]),
            ("Camera dy (microns)", [7]),
            ("Camera rx (asec)", [8]),
            ("Camera ry (asec)", [9]),
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
        y_start = 0.95
        y_step = 0.07  # tighter spacing so we can fit more

        # --- Precompute max integer width for rigid-body numbers ---
        rigid_vals = [states_val[i] for _, idxs in rigid_groups for i in idxs]
        formatted_all = [f"{v:.3f}" for v in rigid_vals]
        int_parts = [f.split(".")[0] for f in formatted_all]
        max_int_rigid = max(len(ip) for ip in int_parts)

        # Track y position per column separately
        ypos: dict[int, float] = {0: y_start, 1: y_start, 2: y_start, 3: y_start}

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
            if idxs[0] in [3, 4, 8, 9]:
                val *= 3600  # convert deg to asec for M2/camera rx, ry
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
        records: dict = {
            "filter": record.physical_filter,
            "observation reason": record.observation_reason,
            "science program": record.science_program,
            "elevation": (90 if record.zenith_angle is None else 90 - record.zenith_angle),
            "azimuth": 0 if record.azimuth is None else record.azimuth,
            "rotator": (0 if len(rotData) == 0 else rotData["actualPosition"].values.mean()),
            "Zernike plot range": f"{self.config.zkYmin:.1f} to {self.config.zkYmax:.1f} microns",
        }
        col = 3

        # Decide which keys are floats that should be decimal-aligned
        float_keys = {"elevation", "azimuth", "rotator"}

        # Format values so floats are aligned
        formatted_records: dict = {}
        for k, v in records.items():
            if str(k) in float_keys:
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
