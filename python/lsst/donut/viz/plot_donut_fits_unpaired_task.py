"""Task for plotting unpaired donut fits visualization."""

from pathlib import Path
from typing import Any, cast

import batoid
import danish
import numpy as np
import yaml
from astropy.table import QTable, Row, Table
from astropy.time import Time
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import least_squares

import lsst.afw.cameraGeom as Camera
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.daf.butler.dimensions import DimensionRecord
from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore, makeEfdClient
from lsst.ts.wep.estimation import DanishAlgorithm
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import getTaskInstrument
from lsst.utils.plotting.figures import make_figure
from lsst.utils.timer import timeMethod

try:
    from lsst.rubintv.production.uploaders import MultiUploader
    from lsst.rubintv.production.utils import getAutomaticLocationConfig, makePlotFile
except ImportError:
    MultiUploader = None

from .utilities import get_day_obs_seq_num_from_visitid, get_instrument_channel_name

__all__ = [
    "PlotDonutFitsUnpairedTaskConnections",
    "PlotDonutFitsUnpairedTaskConfig",
    "PlotDonutFitsUnpairedTask",
]


class PlotDonutFitsUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
):
    aggregateAOSRaw = ct.Input(
        doc="AOS raw catalog",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableRaw",
    )
    zernikeEstimateAvg = ct.Input(
        doc="Average Zernike estimates per detector",
        dimensions=("visit", "instrument"),
        storageClass="AstropyTable",
        name="aggregateAOSVisitTableAvg",
    )
    aggregateDonutTable = ct.Input(
        doc="Aggregated donut table",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
    )
    donutStampsUnpairedVisit = ct.Input(
        doc="Unpaired Donut Stamps",
        dimensions=("visit", "instrument"),
        storageClass="StampsBase",
        name="donutStampsUnpairedVisit",
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    donutFitsUnpaired = ct.Output(
        doc="Donut Fits Unpaired",
        dimensions=("visit", "instrument"),
        storageClass="Plot",
        name="donutFitsUnpaired",
    )


class PlotDonutFitsUnpairedTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotDonutFitsUnpairedTaskConnections,  # type: ignore
):
    doRubinTVUpload: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Upload to RubinTV",
        default=False,
    )
    matchThresholdPx: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Maximum AOS↔stamp centroid separation (pixels) for matching.",
        default=10.0,
    )
    duplicateThresholdPx: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Minimum centroid separation between selected stamps to avoid duplicates (px).",
        default=50.0,
    )
    defocusMm: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Detector defocus magnitude (meters) used for intra/extra models.",
        default=1.5e-3,
    )
    imgMaxFrac: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Quantile used for vmax in image display (0-1).",
        default=0.99,
    )
    imgMinFracScale: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Scale divisor for vmin = -vmax/imgMinFracScale.",
        default=10.0,
    )
    residualScaleDiv: pexConfig.Field = pexConfig.Field(
        dtype=float,
        doc="Residual scaling divisor for vmin/vmax = ±vmax/residualScaleDiv.",
        default=3.0,
    )
    enableEfdQuery: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        doc="Enable querying EFD for telescope control data.",
        default=True,
    )
    efdTopic: pexConfig.Field = pexConfig.Field(
        dtype=str,
        doc="EFD topic name for degree of freedom logevent.",
        default="lsst.sal.MTAOS.logevent_degreeOfFreedom",
    )
    nDonutsPerCorner: pexConfig.Field = pexConfig.Field(
        dtype=int,
        doc="Number of donuts per corner (integer, default: 4).\
        This sets the maximum number of donuts to plot in the visualization. \
        Limited by the 2x2 grid layout of the plot axes.",
        default=4,
    )
    zkYmin: pexConfig.Field = pexConfig.Field(
        dtype=float, doc="Lower limit on Zernike plot (default: -1 micron).", default=-1
    )
    zkYmax: pexConfig.Field = pexConfig.Field(
        dtype=float, doc="Upper limit on Zernike plot (default: +1 micron).", default=1
    )


class PlotDonutFitsUnpairedTask(pipeBase.PipelineTask):
    """Task for plotting unpaired donut fits.

    Inherits from PlotDonutFitsTask. This task reuses the initialization
    and setup from PlotDonutFitsTask
    while providing specialized functionality for unpaired donut visualization.
    """

    ConfigClass = PlotDonutFitsUnpairedTaskConfig
    _DefaultName = "plotDonutFitsUnpairedTask"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: PlotDonutFitsUnpairedTaskConfig = cast(PlotDonutFitsUnpairedTaskConfig, self.config)

        if self.config.doRubinTVUpload:
            if not MultiUploader:
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
        obsc = self.instrument.obscuration
        focal_length = self.instrument.focalLength
        r_outer = self.instrument.radius
        pixel_scale = self.instrument.pixelSize
        self.factory = danish.DonutFactory(
            R_outer=r_outer,
            R_inner=r_outer * obsc,
            mask_params=self.mask_params,
            focal_length=focal_length,
            pixel_scale=pixel_scale,
        )

        # Setup danish algo
        self.danish_algo = DanishAlgorithm()
        self.danish_model_keys = ["fwhm", "model_dx", "model_dy", "model_sky_level"]

        # Cache instrument obscuration and focal length
        self.obsc = self.instrument.obscuration
        self.fl = self.instrument.focalLength

    # Helpers
    def _safeNormalize(self, arr: np.ndarray) -> np.ndarray:
        try:
            total = float(np.nansum(arr))
        except Exception:
            return np.zeros_like(arr)
        if not np.isfinite(total) or total <= 0.0:
            return np.zeros_like(arr)
        return arr / total

    def _getCmap(self) -> LinearSegmentedColormap:
        # Cache custom colormap on first use
        if getattr(self, "_cached_cmap", None) is None:
            colors = [
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.0, 0.0, 0.0),
            ]
            positions = [0.0, 1 / 11, 1.0]
            self._cached_cmap = LinearSegmentedColormap.from_list(
                "cyan_white_magenta", list(zip(positions, colors))
            )
        return self._cached_cmap

    def _buildTelescopeModels(self, bandpass: str) -> tuple[batoid.Optic, batoid.Optic, batoid.Optic]:
        telescope = batoid.Optic.fromYaml(f"LSST_{bandpass}.yaml")
        dz = float(self.config.defocusMm)
        intra_telescope = telescope.withGloballyShiftedOptic(
            "Detector",
            [0, 0, -dz],
        )
        extra_telescope = telescope.withGloballyShiftedOptic(
            "Detector",
            [0, 0, +dz],
        )
        return telescope, intra_telescope, extra_telescope

    # Small plotting helpers to reduce duplication
    def _imshow_img(self, ax: Axes, img_norm: np.ndarray, vmax: float, cmap: LinearSegmentedColormap) -> None:
        ax.imshow(
            img_norm,
            cmap=cmap,
            vmin=-vmax / self.config.imgMinFracScale,
            vmax=vmax,
            interpolation="nearest",
        )

    def _imshow_residual(self, ax: Axes, residual: np.ndarray) -> None:
        vmax_residual = float(np.nanmax(np.abs(residual)))
        if not np.isfinite(vmax_residual) or vmax_residual == 0.0:
            vmax_residual = 1.0
        # Reduce dynamic range to 50% of the residual max to enhance contrast
        vmax_residual *= 0.5
        ax.imshow(
            residual,
            cmap="bwr",
            vmin=-vmax_residual,
            vmax=vmax_residual,
            interpolation="nearest",
        )

    def _plot_zernike_bars(
        self,
        ax: Axes,
        row: Row,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        fontsize: int = 10,
    ) -> None:
        if row is None or "zk_CCS" not in row.colnames:
            raise ValueError("Invalid row data for Zernike bar plot.")
        if row.meta is None:
            raise ValueError("Row metadata missing nollIndices for Zernike bar plot.")
        ax.bar(row.meta["nollIndices"], row["zk_CCS"], color="k")
        ax.axhline(0, color="k", lw=0.5)
        zk_max = np.nanmax(np.abs(row["zk_CCS"]))
        ylim_max = max(zk_max * 1.2, 0.5)
        ax.set_ylim(-ylim_max, ylim_max)
        ax.set_xlim(3.5, 28.5)
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        # Decorations
        marker_y = ylim_max * 0.88
        ax.scatter([4, 11, 22], [marker_y] * 3, marker="o", ec="k", c="none", s=10, lw=0.5)
        ax.scatter([7, 17], [marker_y] * 2, marker="$\u2191$", c="k", s=10, lw=0.5)
        ax.scatter([8, 16], [marker_y] * 2, marker="$\u2192$", c="k", s=10, lw=0.5)
        for j in [4, 11, 22]:
            ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.2, ec="none")
        for j in [5, 12, 23]:
            ax.axvspan(j - 0.5, j + 1.5, color="orange", alpha=0.2, ec="none")
        for j in [7, 16]:
            ax.axvspan(j - 0.5, j + 1.5, color="yellow", alpha=0.2, ec="none")
        for j in [9, 18]:
            ax.axvspan(j - 0.5, j + 1.5, color="green", alpha=0.2, ec="none")
        for j in [14, 25]:
            ax.axvspan(j - 0.5, j + 1.5, color="blue", alpha=0.2, ec="none")
        ax.axvspan(19.5, 21.5, color="indigo", alpha=0.2, ec="none")
        ax.axvspan(26.5, 28.5, color="violet", alpha=0.2, ec="none")

    def _render_combined_zk(self, ax: Axes, donut_data: list[dict]) -> None:
        """Render combined Zernike coefficients from multiple donuts.

        Parameters
        ----------
        ax : Axes
            Matplotlib Axes to render the plot on.
        donut_data : list of dict
            List of donut data dictionaries containing Zernike coefficients.
        """
        # Find noll indices and gather per-donut zk
        noll_indices = None
        zk_lists = []
        avg_zk = None
        manual_zk = None
        for d in donut_data:
            row = d.get("row")
            if row is not None and noll_indices is None:
                try:
                    noll_indices = row.meta["nollIndices"]
                except (KeyError, AttributeError, TypeError):
                    pass
            row = d.get("row")  # type: ignore[assignment]
            if row is not None:
                zk = row["zk_CCS"]
            else:
                zk = None
            if zk is not None:
                zk_lists.append(zk)
            if avg_zk is None and d.get("avg_zk_CCS") is not None:
                avg_zk = d.get("avg_zk_CCS")
            if manual_zk is None and d.get("manual_avg_zk") is not None:
                manual_zk = d.get("manual_avg_zk")

        if noll_indices is None or len(zk_lists) == 0:
            self._empty_cell(ax, "No ZK data")
            return

        # Background spans like other ZK plot
        for j in [4, 11, 22]:
            ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.2, ec="none")
        for j in [5, 12, 23]:
            ax.axvspan(j - 0.5, j + 1.5, color="orange", alpha=0.2, ec="none")
        for j in [7, 16]:
            ax.axvspan(j - 0.5, j + 1.5, color="yellow", alpha=0.2, ec="none")
        for j in [9, 18]:
            ax.axvspan(j - 0.5, j + 1.5, color="green", alpha=0.2, ec="none")
        for j in [14, 25]:
            ax.axvspan(j - 0.5, j + 1.5, color="blue", alpha=0.2, ec="none")
        ax.axvspan(19.5, 21.5, color="indigo", alpha=0.2, ec="none")
        ax.axvspan(26.5, 28.5, color="violet", alpha=0.2, ec="none")

        # Plot faded per-donut points
        for zk in zk_lists:
            ax.scatter(noll_indices, zk, s=10, color="dimgray", alpha=0.5, marker="x", zorder=1)

        # Emphasize manual average and corrected average
        if manual_zk is not None:
            ax.scatter(
                noll_indices,
                manual_zk,
                s=18,
                color="black",
                alpha=0.9,
                label="Manual Avg",
                zorder=3,
            )
        if avg_zk is not None:
            ax.scatter(
                noll_indices,
                avg_zk,
                s=18,
                color="tab:green",
                alpha=0.9,
                label="Corrected Avg",
                zorder=3,
            )

        # Axes styling: dynamic y-limits from data percentiles with padding
        all_vals = (
            zk_lists
            + ([manual_zk] if manual_zk is not None else [])
            + ([avg_zk] if avg_zk is not None else [])
        )
        # Compute vmax for fallback symmetric bounds
        vmax = 0.5
        try:
            conc = np.concatenate(all_vals)
            vmax = max(0.5, float(np.nanmax(np.abs(conc))))
        except Exception as e:
            self.log.warning(
                "Failed to compute vmax for combined ZK fallback; using default 0.5. Error: %s",
                e,
            )

        # Try dynamic y-limits from percentiles
        try:
            conc = np.concatenate(all_vals)
            y_min = float(np.nanpercentile(conc, 1))
            y_max = float(np.nanpercentile(conc, 99))
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                raise ValueError("bad range")
            pad = 0.1 * (y_max - y_min)
            y_min -= pad
            y_max += pad
        except Exception as e:
            self.log.warning(
                "Failed to compute dynamic y-limits for combined ZK; using symmetric bounds. Error: %s",
                e,
            )
            # fallback symmetric bounds
            y_min, y_max = -1.2 * vmax, 1.2 * vmax
        ax.set_xlim(3.5, 28.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Noll Index", fontsize=8)
        ax.set_ylabel("Coeff (μm)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    def _render_histogram(
        self,
        ax: Axes,
        donut_table_rows: np.ndarray,
        rows: np.ndarray,
        donut_data: list[dict],
        det_name: str,
    ) -> None:
        ood_scores = None
        if len(donut_table_rows) > 0 and "ood_score" in donut_table_rows.colnames:
            ood_scores = donut_table_rows["ood_score"]
            self.log.debug(f"Found ood_score in donut_table for {det_name}")
        elif len(rows) > 0 and "ood_score" in rows.colnames:
            ood_scores = rows["ood_score"]
            self.log.debug(f"Found ood_score in rows for {det_name}")

        if ood_scores is not None:
            ood_scores_valid = ood_scores[~np.isnan(ood_scores)]
            if len(ood_scores_valid) > 0:
                ax.hist(
                    ood_scores_valid,
                    bins=10,
                    color="skyblue",
                    edgecolor="black",
                    alpha=0.7,
                )
                mean_val = np.mean(ood_scores_valid)
                median_val = np.median(ood_scores_valid)
                std_val = np.std(ood_scores_valid)
                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.3f}",
                )
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.3f}",
                )
                ax.axvline(
                    mean_val + std_val,
                    color="orange",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"±Std: {std_val:.3f}",
                )
                ax.axvline(mean_val - std_val, color="orange", linestyle=":", linewidth=1.5)
                ax.set_xlabel("OOD Score", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.3)
                # Mark selected scores
                sel_scores = []
                for d in donut_data:
                    try:
                        sc = float(d.get("ood_score", np.nan))
                        if np.isfinite(sc):
                            sel_scores.append(sc)
                    except Exception:
                        continue
                added = False
                for sc in sel_scores:
                    ax.axvline(
                        sc,
                        color="k",
                        linestyle="-",
                        linewidth=1,
                        alpha=0.7,
                        label=("Selected" if not added else None),
                    )
                    added = True
                return
            ax.text(
                0.5,
                0.5,
                "No valid OOD scores",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return
        # No ood scores available
        ax.text(
            0.5,
            0.5,
            "OOD score not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _focal_color(self, det_name: str) -> str:
        return "blue" if det_name.endswith("SW1") else "red"

    def _empty_cell(self, ax: Axes, text: str, fontsize: int | None = None) -> None:
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=(fontsize or 8),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _render_corrected_and_manual_panels(
        self,
        axs: np.ndarray,
        col_idx: int,
        telescope: batoid.Optic,
        intra_telescope: batoid.Optic,
        extra_telescope: batoid.Optic,
        wavelength: float,
        det_name: str,
        data: dict,
        img: np.ndarray,
        img_norm: np.ndarray,
        model_norm: np.ndarray,
        vmax: float,
        vmax_residual: float | None,
        col_title: str,
    ) -> None:
        # Hold normalized models for cross-difference (Corr - Man)
        avg_model_norm_val = None
        manual_model_norm_val = None
        # Corrected (avg) panels
        if data.get("avg_zk_CCS") is not None and data.get("avg_row") is not None:
            try:
                avg_model, _ = self._generateAvgModel(
                    telescope,
                    intra_telescope,
                    extra_telescope,
                    data["avg_zk_CCS"],
                    data["avg_row"],
                    img,
                    wavelength,
                    data["row"]["thx_CCS"],
                    data["row"]["thy_CCS"],
                    det_name,
                )
                avg_model_norm = self._safeNormalize(avg_model)
                avg_model_norm_val = avg_model_norm
                self._imshow_img(axs[7][col_idx], avg_model_norm, vmax, self._getCmap())
                axs[7][col_idx].set_title(col_title, fontsize=9)
                axs[7][col_idx].set_xticks([])
                axs[7][col_idx].set_yticks([])

                avg_residual = img_norm - avg_model_norm
                self._imshow_residual(axs[8][col_idx], avg_residual)
                axs[8][col_idx].set_title(col_title, fontsize=9)
                axs[8][col_idx].set_xticks([])
                axs[8][col_idx].set_yticks([])
            except Exception:
                for r in [7, 8, 9]:
                    self._empty_cell(axs[r][col_idx], "Corrected\nError")
        else:
            for r in [7, 8, 9]:
                self._empty_cell(axs[r][col_idx], "No Corrected Data")

        # Manual panels
        if data.get("manual_avg_zk") is not None and data.get("manual_avg_row") is not None:
            try:
                manual_model, _ = self._generateAvgModel(
                    telescope,
                    intra_telescope,
                    extra_telescope,
                    data["manual_avg_zk"],
                    data["manual_avg_row"],
                    img,
                    wavelength,
                    data["row"]["thx_CCS"],
                    data["row"]["thy_CCS"],
                    det_name,
                )
                manual_model_norm = self._safeNormalize(manual_model)
                manual_model_norm_val = manual_model_norm
                self._imshow_img(axs[4][col_idx], manual_model_norm, vmax, self._getCmap())
                axs[4][col_idx].set_title(col_title, fontsize=9)
                axs[4][col_idx].set_xticks([])
                axs[4][col_idx].set_yticks([])

                manual_residual = img_norm - manual_model_norm
                self._imshow_residual(axs[5][col_idx], manual_residual)
                axs[5][col_idx].set_title(col_title, fontsize=9)
                axs[5][col_idx].set_xticks([])
                axs[5][col_idx].set_yticks([])

                manual_ind_diff = manual_model_norm - model_norm
                vmax_manual_diff = np.nanmax(np.abs(manual_ind_diff))
                axs[6][col_idx].imshow(
                    manual_ind_diff,
                    cmap="bwr",
                    vmin=-vmax_manual_diff,
                    vmax=vmax_manual_diff,
                    interpolation="nearest",
                )
                axs[6][col_idx].set_title(col_title, fontsize=9)
                axs[6][col_idx].set_xticks([])
                axs[6][col_idx].set_yticks([])
            except Exception as e:
                self.log.exception(
                    "Failed to generate manual avg model for donut %d: %s",
                    col_idx + 1,
                    str(e),
                )
                error_msg = str(e)[:50]
                for r in [4, 5, 6]:
                    self._empty_cell(axs[r][col_idx], f"Manual Avg\nError:\n{error_msg}")
        else:
            for r in [4, 5, 6]:
                self._empty_cell(axs[r][col_idx], "No Manual Avg")

        # Render Corr-Man (Corrected - Manual) in row 9 if both are available
        try:
            if (avg_model_norm_val is not None) and (manual_model_norm_val is not None):
                corr_man_diff = avg_model_norm_val - manual_model_norm_val
                vmax_corr_man = np.nanmax(np.abs(corr_man_diff))
                if not np.isfinite(vmax_corr_man) or vmax_corr_man == 0.0:
                    vmax_corr_man = 1.0
                axs[9][col_idx].imshow(
                    corr_man_diff,
                    cmap="bwr",
                    vmin=-vmax_corr_man,
                    vmax=vmax_corr_man,
                    interpolation="nearest",
                )
                axs[9][col_idx].set_title(col_title, fontsize=9)
                axs[9][col_idx].set_xticks([])
                axs[9][col_idx].set_yticks([])
            else:
                self._empty_cell(axs[9][col_idx], "No Corr-Man Data")
        except Exception:
            self._empty_cell(axs[9][col_idx], "Corr-Man Error")

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Get the inputs
        aos_raw = butlerQC.get(inputRefs.aggregateAOSRaw)
        zk_avg = butlerQC.get(inputRefs.zernikeEstimateAvg)
        aggregate_donut_table = butlerQC.get(inputRefs.aggregateDonutTable)
        donutStampsUnpaired = butlerQC.get(inputRefs.donutStampsUnpairedVisit)
        camera = butlerQC.get(inputRefs.camera)
        visit = inputRefs.aggregateAOSRaw.dataId["visit"]
        record = inputRefs.aggregateAOSRaw.dataId.records["visit"]

        day_obs, seq_num = get_day_obs_seq_num_from_visitid(visit)
        fig = self.run(
            aos_raw,
            zk_avg,
            aggregate_donut_table,
            donutStampsUnpaired,
            camera,
            day_obs,
            seq_num,
            record,
        )

        butlerQC.put(fig, outputRefs.donutFitsUnpaired)

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

    def getModelBase(
        self,
        telescope: batoid.Optic,
        defocused_telescope: batoid.Optic,
        row: Row,
        img: np.ndarray,
        wavelength: float,
        inex: str,
        thx: float | None = None,
        thy: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Generate theoretical donut model for single defocal position.

        Note: This method diverges from the parent class's `getModel`
        implementation (see `plot_aos_task.py`). Unlike paired donuts,
        unpaired donut stamps are not always centered from TARTs, so we
        need to refit the donut stamps (centers and FWHM) at the time of
        model generation using `least_squares` optimization. This is why
        we use `danish.SingleDonutModel` with refitting rather than the
        parent's approach that relies on pre-fitted metadata from the AOS
        catalog.

        Parameters
        ----------
        telescope : batoid.Optic
            In-focus telescope optical model
        defocused_telescope : batoid.Optic
            Defocused telescope optical model
        row : astropy.table.Row
            AOS catalog row with Zernike coefficients and metadata
        img : np.ndarray
            Observed donut image to fit
        wavelength : float
            Wavelength in meters
        inex : str
            Focal type identifier ("intra" or "extra")
        thx : float, optional
            Field angle x in CCS coordinates. If None, extracted from row.
        thy : float, optional
            Field angle y in CCS coordinates. If None, extracted from row.

        Returns
        -------
        model : np.ndarray
            Theoretical donut model image
        fwhm : float
            Fitted FWHM parameter from the optimization
        """
        if thx is None or thy is None:
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
                focal_length=self.fl,
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

    def getModelUnpaired(
        self,
        telescope: batoid.Optic,
        intra_telescope: batoid.Optic,
        extra_telescope: batoid.Optic,
        row: Row,
        img: np.ndarray,
        wavelength: float,
        detector_name: str,
    ) -> tuple[np.ndarray, float]:
        """Generate theoretical donut model for single defocal position.

        Parameters
        ----------
        telescope : batoid.Optic
            In-focus telescope optical model
        intra_telescope : batoid.Optic
            Intra-focal telescope optical model
        extra_telescope : batoid.Optic
            Extra-focal telescope optical model
        row : astropy.table.Row
            AOS catalog row with Zernike coefficients
        img : np.ndarray
            Observed donut image
        wavelength : float
            Wavelength in meters
        detector_name : str
            Detector name (e.g., "R00_SW0" or "R00_SW1")

        Returns
        -------
        model : np.ndarray
            Theoretical donut model
        fwhm : float
            Fitted FWHM parameter
        """
        # Determine which telescope model to use based on detector name
        if detector_name.endswith("SW1"):
            # Intra-focal detector
            defocused_telescope = intra_telescope
            focal_type = "intra"
            self.log.debug("Using intra-focal telescope model for detector %s", detector_name)
        elif detector_name.endswith("SW0"):
            # Extra-focal detector
            defocused_telescope = extra_telescope
            focal_type = "extra"
            self.log.debug("Using extra-focal telescope model for detector %s", detector_name)
        else:
            # Ambiguous detector naming: require explicit SW0/SW1 suffix
            raise ValueError(
                f"Unclear detector name '{detector_name}'. Expected suffix 'SW0' (extra) or 'SW1' (intra)."
            )

        # Extract CCS positions for unpaired (no suffix on field names)
        thx = row["thx_CCS"]
        thy = row["thy_CCS"]

        return self.getModelBase(
            telescope,
            defocused_telescope,
            row,
            img,
            wavelength,
            focal_type,
            thx=thx,
            thy=thy,
        )

    def _generateAvgModel(
        self,
        telescope: batoid.Optic,
        intra_telescope: batoid.Optic,
        extra_telescope: batoid.Optic,
        avg_zk_CCS: np.ndarray,
        avg_row: Row,
        img: np.ndarray,
        wavelength: float,
        thx_individual: float,
        thy_individual: float,
        detector_name: str,
    ) -> tuple[np.ndarray, float]:
        """Generate theoretical donut model using average Zernike coefficients.

        Parameters
        ----------
        telescope : batoid.Optic
            In-focus telescope optical model
        intra_telescope : batoid.Optic
            Intra-focal telescope optical model
        extra_telescope : batoid.Optic
            Extra-focal telescope optical model
        avg_zk_CCS : np.ndarray
            Average Zernike coefficients for the detector
        avg_row : astropy.table.Row
            Average AOS catalog row (for metadata like nollIndices)
        img : np.ndarray
            Observed donut image (for sizing)
        wavelength : float
            Wavelength in meters
        thx_individual : float
            Individual donut's CCS field angle x
        thy_individual : float
            Individual donut's CCS field angle y
        detector_name : str
            Detector name (e.g., "R00_SW0" or "R00_SW1")

        Returns
        -------
        model : np.ndarray
            Theoretical donut model
        fwhm : float
            Fitted FWHM parameter
        """
        # Determine which telescope model to use based on detector name
        if detector_name.endswith("SW1"):
            # Intra-focal detector
            defocused_telescope = intra_telescope
            focal_type = "intra"
        elif detector_name.endswith("SW0"):
            # Extra-focal detector
            defocused_telescope = extra_telescope
            focal_type = "extra"
        else:
            # Ambiguous detector naming: require explicit SW0/SW1 suffix
            raise ValueError(
                f"Unclear detector name '{detector_name}'. Expected suffix 'SW0' (extra) or 'SW1' (intra)."
            )

        # Create a temporary row-like object with average Zernikes
        # but individual field position
        class TempRow:
            def __init__(self, zk: np.ndarray, meta: dict) -> None:
                self.data = {"zk_CCS": zk}
                self.meta = meta

            def __getitem__(self, key: str) -> np.ndarray:
                return self.data[key]

        try:
            temp_row = TempRow(avg_zk_CCS, avg_row.meta)
        except AttributeError as e:
            msg = f"avg_row missing metadata; expected 'meta'. Type: {type(avg_row)}"
            self.log.error(msg)
            raise ValueError(msg) from e

        # Use avg Zernikes with individual position
        return self.getModelBase(
            telescope,
            defocused_telescope,
            temp_row,
            img,
            wavelength,
            focal_type,
            thx=thx_individual,
            thy=thy_individual,
        )

    def plotResults(
        self,
        axs: list[Axes],
        img: np.ndarray,
        model: np.ndarray,
        row: Row,
        blur: float,
        match_dist: float | None = None,
        focal_type: str | None = None,
    ) -> None:
        """Create 4-row visualization for unpaired donuts.

        Parameters
        ----------
        axs : list of Axes
            List of 4 axes for the visualization
        img : np.ndarray
            Observed donut image
        model : np.ndarray
            Theoretical donut model
        row : astropy.table.Row
            AOS catalog row
        blur : float
            Blur parameter for display
        match_dist : float, optional
            Distance between AOS centroid and donut stamp centroid
        focal_type : str, optional
            Focal type ("intra" or "extra")
        """
        # Custom colormap (cached)
        cmap = self._getCmap()

        # Normalize images
        img_norm = self._safeNormalize(img)
        model_norm = self._safeNormalize(model)
        residual = img_norm - model_norm
        vmax = np.nanquantile(img_norm, self.config.imgMaxFrac)

        # Row 1: Observed donut
        self._imshow_img(axs[0], img_norm, vmax, cmap)
        axs[0].text(5, 150, f"blur: {blur:5.3f}")
        if match_dist is not None:
            axs[0].text(5, 130, f"match: {match_dist:.1f}px", fontsize=8)
        if focal_type is not None:
            axs[0].text(
                5,
                110,
                f"focal: {focal_type}",
                fontsize=8,
                color="blue" if focal_type == "intra" else "red",
            )
        axs[0].set_title("Observed", fontsize=10)

        # Row 2: Model
        self._imshow_img(axs[1], model_norm, vmax, cmap)
        model_title = f"Model ({focal_type})" if focal_type else "Model"
        axs[1].set_title(model_title, fontsize=10)

        # Row 3: Residuals
        self._imshow_residual(axs[2], residual)
        axs[2].set_title("Residual", fontsize=10)

        # Row 4: Zernike coefficients
        self._plot_zernike_bars(
            axs[3],
            row,
            title="Zernike Coefficients",
            xlabel="Noll Index",
            ylabel="Coefficient (μm)",
        )

        # Color-code the right spine based on usage
        color = "green" if row["used"] else "red"
        axs[3].spines["right"].set_edgecolor(color)
        axs[3].spines["right"].set_linewidth(1.5)

        # Remove ticks from image plots
        for ax in axs[:3]:
            ax.set_xticks([])
            ax.set_yticks([])

    def plotResultsMultiDonut(
        self,
        axs: list[list[Axes]],
        donut_data: list[dict],
        telescope: batoid.Optic,
        intra_telescope: batoid.Optic,
        extra_telescope: batoid.Optic,
        wavelength: float,
        det_name: str,
    ) -> None:
        """Create per-detector 10x4 panels plus a histogram row.

        Parameters
        ----------
        axs : list[list[Axes]]
            11-row grid: rows 0–9 have 4 cols, row 10 spans full width
        donut_data : list of dict
            List of donut data dictionaries, each containing:
            - img: observed donut image
            - model: theoretical donut model (individual)
            - row: AOS catalog row
            - blur: blur parameter
            - match_dist: matching distance
            - focal_type: "intra" or "extra"
            - avg_zk_CCS: corrected (average) Zernike coefficients for detector
            - avg_row: corrected (average) AOS catalog row for detector
            - manual_avg_zk: manually averaged raw Zernikes for detector
            - manual_avg_row: row metadata for manual average
        telescope : batoid.Optic
            In-focus telescope optical model
        intra_telescope : batoid.Optic
            Intra-focal telescope optical model
        extra_telescope : batoid.Optic
            Extra-focal telescope optical model
        wavelength : float
            Wavelength in meters
        det_name : str
            Detector name for determining focal type
        """
        # Custom colormap (cached)
        cmap = self._getCmap()

        # First pass: normalize and compute global vmax for this detector
        normalized_data = []
        vmax_global = 0
        for i, data in enumerate(donut_data):
            if i >= self.config.nDonutsPerCorner:
                break

            img = data["img"]
            model = data["model"]

            img_norm = self._safeNormalize(img)
            model_norm = self._safeNormalize(model)
            residual = img_norm - model_norm
            vmax = np.nanquantile(img_norm, self.config.imgMaxFrac)
            vmax_global = max(vmax_global, vmax)

            normalized_data.append(
                {
                    "img_norm": img_norm,
                    "model_norm": model_norm,
                    "residual": residual,
                    "vmax": vmax,
                }
            )

        # Residuals use per-plot dynamic range (no shared scaling)

        # Process each donut for plotting
        for i, data in enumerate(donut_data):
            if i >= self.config.nDonutsPerCorner:  # Only process up to configured limit
                break

            img = data["img"]
            model = data["model"]
            row = data["row"]
            blur = data["blur"]
            match_dist = data["match_dist"]
            focal_type = data["focal_type"]
            # Column title: prefer OOD score if available and valid
            col_title = f"#{i + 1}"
            if "ood_score" in data:
                score_val = data["ood_score"]
                try:
                    score_float = float(score_val)
                except (TypeError, ValueError):
                    self.log.debug("Invalid ood_score value for donut %d: %r", i + 1, score_val)
                    score_float = np.nan
                if np.isfinite(score_float):
                    col_title = f"{score_float:.3f}"

            # Get normalized data from first pass
            img_norm = normalized_data[i]["img_norm"]
            model_norm = normalized_data[i]["model_norm"]
            residual = normalized_data[i]["residual"]
            vmax = normalized_data[i]["vmax"]

            # Row 0: Observed donuts (column i)
            self._imshow_img(axs[0][i], img_norm, vmax, cmap)
            axs[0][i].text(5, 150, f"blur: {blur:5.3f}", fontsize=8)
            axs[0][i].text(5, 130, f"match: {match_dist:.1f}px", fontsize=8)
            axs[0][i].text(
                5,
                110,
                f"focal: {focal_type}",
                fontsize=8,
                color="blue" if focal_type == "intra" else "red",
            )
            # annotate observed centroid in data pixels
            if "obs_centroid" in data:
                cx, cy = data["obs_centroid"]
                axs[0][i].text(
                    img_norm.shape[1] / 2,
                    img_norm.shape[0] / 2,
                    f"({cx:.0f},{cy:.0f})",
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4),
                )
            axs[0][i].set_title(col_title, fontsize=9)
            axs[0][i].set_xticks([])
            axs[0][i].set_yticks([])

            # Row 1: Models (column i)
            self._imshow_img(axs[1][i], model_norm, vmax, cmap)
            axs[1][i].set_title(col_title, fontsize=9)
            axs[1][i].set_xticks([])
            axs[1][i].set_yticks([])
            # annotate AOS centroid in data pixels
            if "aos_centroid" in data:
                cx, cy = data["aos_centroid"]
                axs[1][i].text(
                    model_norm.shape[1] / 2,
                    model_norm.shape[0] / 2,
                    f"({cx:.0f},{cy:.0f})",
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4),
                )

            # Row 2: Residuals (col i) with per-plot dynamic range
            self._imshow_residual(axs[2][i], residual)
            axs[2][i].set_title(col_title, fontsize=9)
            axs[2][i].set_xticks([])
            axs[2][i].set_yticks([])

            # Row 3: Zernike coefficients (column i)
            self._plot_zernike_bars(
                axs[3][i],
                row,
                title=col_title,
                xlabel="Noll Index",
                ylabel=("Coeff (μm)" if i == 0 else None),
                fontsize=9,
            )
            if i != 0:
                axs[3][i].set_yticks([])
                axs[3][i].set_ylabel("")

            # Color-code the right spine based on usage
            color = "green" if row["used"] else "red"
            axs[3][i].spines["right"].set_edgecolor(color)
            axs[3][i].spines["right"].set_linewidth(1.5)

            # Row 4: Corrected model (based on detector average Zernike)
            self._render_corrected_and_manual_panels(
                axs,
                i,
                telescope,
                intra_telescope,
                extra_telescope,
                wavelength,
                det_name,
                data,
                img,
                img_norm,
                model_norm,
                vmax,
                None,
                col_title,
            )

        # Fill empty columns if we have fewer than 4 donuts
        for i in range(len(donut_data), 4):
            for row_idx in range(10):
                axs[row_idx][i].text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=axs[row_idx][i].transAxes,
                )
                axs[row_idx][i].set_xticks([])
                axs[row_idx][i].set_yticks([])

    def run(
        self,
        aos_raw: Table,
        zk_avg: Table,
        aggregate_donut_table: QTable,
        donutStampsIntra: DonutStamps,
        camera: Camera,
        day_obs: int,
        seq_num: int,
        record: DimensionRecord | None,
    ) -> Figure:
        """Run the PlotDonutFitsUnpaired AOS task.

        Creates a figure of unpaired donuts, models, residuals, and Zernikes.

        Parameters
        ----------
        aos_raw: Astropy Table
            The AOS raw catalog.
        zk_avg: Astropy Table
            The AOS average catalog (one row per detector).
        aggregate_donut_table: Astropy QTable
            The aggregated donut table with additional fields like ood_score.
        donutStampsIntra: DonutStamps
            The unpaired donut stamps.
        camera: `lsst.afw.cameraGeom.Camera`
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
            If EFD query is enabled but no Butler record is provided.
        """
        # Get bandpass and wavelength
        bandpass = donutStampsIntra.getBandpasses()[0]
        assert all([bandpass == bp for bp in donutStampsIntra.getBandpasses()])
        wavelength = self.instrument.wavelength[bandpass]
        assert isinstance(wavelength, float)

        # Create telescope models with configured defocus for unpaired donuts
        telescope, intra_telescope, extra_telescope = self._buildTelescopeModels(bandpass)

        # Get telescope control data from EFD
        if record is None:
            raise RuntimeError("Butler record is required for EFD query but was not provided.")
        startTime = record.timespan.begin
        efd_topic = self.config.efdTopic
        states_val = np.empty(50)
        visit_logevent: int | str = "unknown"

        if self.config.enableEfdQuery:
            event = getMostRecentRowWithDataBefore(
                self.efd_client,
                efd_topic,
                timeToLookBefore=Time(startTime, scale="utc"),
            )
            for i in range(50):
                states_val[i] = event[f"aggregatedDoF{i}"]
            if "visitId" in event.keys():
                visit_logevent = event["visitId"]

        # Create figure with 2x4 grid for 8 detectors
        fig = make_figure(figsize=(20, 23))
        fig.suptitle(
            f"Unpaired Donut Fits - {day_obs} seq{seq_num}\n"
            f"Blue: Intra-focal detectors (SW1) | Red: Extra-focal detectors (SW0)",
            fontsize=15,
            fontweight="bold",
        )

        # Define detector names
        detector_names = [
            "R00_SW0",
            "R00_SW1",
            "R04_SW0",
            "R04_SW1",
            "R40_SW0",
            "R40_SW1",
            "R44_SW0",
            "R44_SW1",
        ]

        # Create grid layout: 2 rows x 4 columns with space for labels
        gs0 = GridSpec(
            nrows=2,
            ncols=4,
            left=0.045,  # Space for row labels on left
            right=0.985,
            bottom=0.055,
            top=0.955,  # tighter margins to maximize panel area
            wspace=0.06,
            hspace=0.08,
        )

        # Get blur information
        donut_blur = np.zeros(len(aos_raw))
        if "fwhm" in aos_raw.meta["estimatorInfo"].keys():
            fwhm_data = np.array(aos_raw.meta["estimatorInfo"].get("fwhm"))
            if len(fwhm_data) == len(aos_raw):
                donut_blur = fwhm_data
            elif len(fwhm_data) > 0:
                # If fwhm is shorter, try to broadcast or use first value
                # This handles cases where fwhm might be per-detector
                # or per-pair
                self.log.warning(
                    "fwhm array length (%d) does not match aos_raw length (%d). "
                    "Using first value for all rows.",
                    len(fwhm_data),
                    len(aos_raw),
                )
                donut_blur[:] = fwhm_data[0]
            # else: keep zeros if fwhm_data is empty

        # Prepare to compute precise row-label positions from actual axes
        top_row_centers = [None] * 12
        bottom_row_centers = [None] * 12

        # Process each detector
        for i, det_name in enumerate(detector_names):
            row_idx = i // 4
            col_idx = i % 4

            # Create subplot for this detector: 11 rows x 4 columns
            # (for 4 donuts + 1 histogram row)
            gs1 = GridSpecFromSubplotSpec(
                nrows=13,
                ncols=4,
                subplot_spec=gs0[row_idx, col_idx],
                hspace=0.06,
                wspace=0.04,
            )

            axs = []
            for j in range(10):  # First 10 rows: 4 columns each
                row_axs = []
                for k in range(4):  # 4 columns (donuts)
                    ax = fig.add_subplot(gs1[j, k])
                    row_axs.append(ax)
                axs.append(row_axs)

            # Row 10-11 (combined ZK): spans two rows across all 4 columns
            zk_combined_ax = fig.add_subplot(gs1[10:12, :])
            axs.append([zk_combined_ax])
            # Row 12 (histogram): spans all 4 columns
            histogram_ax = fig.add_subplot(gs1[12, :])
            axs.append([histogram_ax])

            # Get detector info (fail fast if missing)
            det = camera[det_name]
            det_id = det.getId()

            # Capture row centers (first column) for label alignment
            if col_idx == 0:
                for rr in range(len(axs)):
                    bbox = axs[rr][0].get_position()
                    center = bbox.y0 + bbox.height / 2
                    if row_idx == 0:
                        top_row_centers[rr] = center
                    else:
                        bottom_row_centers[rr] = center

            # Set detector title with focal type
            # Span across all columns in first row
            focal_color = "blue" if det_name.endswith("SW1") else "red"
            axs[0][0].set_title(f"ID: {det_id}", fontsize=10, fontweight="bold", color=focal_color)

            # Add colored border to indicate focal type
            for row_axs in axs:
                for ax in row_axs:
                    ax.spines["top"].set_color(focal_color)
                    ax.spines["top"].set_linewidth(1.5)
                    ax.spines["right"].set_color(focal_color)
                    ax.spines["right"].set_linewidth(1.5)
                    ax.spines["bottom"].set_color(focal_color)
                    ax.spines["bottom"].set_linewidth(1.5)
                    ax.spines["left"].set_color(focal_color)
                    ax.spines["left"].set_linewidth(1.5)

            # Get AOS data for this detector
            selected_rows = aos_raw["detector"] == det_name
            rows = aos_raw[selected_rows]
            blur = donut_blur[selected_rows]

            # Get donut table data (e.g., ood_score)
            donut_table_selected = aggregate_donut_table["detector"] == det_name
            donut_table_rows = aggregate_donut_table[donut_table_selected]

            # Get corrected average Zernikes (from aggregateAOSVisitTableAvg)
            avg_selected = zk_avg["detector"] == det_name
            if np.sum(avg_selected) > 0:
                avg_zk_for_detector = zk_avg[avg_selected]["zk_CCS"][0]  # Single row per detector
                avg_row_for_detector = zk_avg[avg_selected][0]  # Keep full row for metadata
            else:
                avg_zk_for_detector = None
                avg_row_for_detector = None

            # Compute manual average from raw individual Zernikes
            if len(rows) > 0:
                try:
                    manual_avg_zk = np.mean(rows["zk_CCS"], axis=0)
                    # Create row-like holder for manual average metadata
                    manual_avg_row = rows[0]  # Use first row for metadata
                    self.log.debug(
                        "Manual avg zk for %s: shape=%s, mean=%.3f",
                        det_name,
                        manual_avg_zk.shape,
                        float(np.mean(manual_avg_zk)),
                    )
                except Exception as e:
                    self.log.warning(f"Failed to compute manual average Zernikes for {det_name}: {e}")
                    manual_avg_zk = None
                    manual_avg_row = None
            else:
                manual_avg_zk = None
                manual_avg_row = None

            if len(rows) == 0:
                # No AOS data for this detector
                for row_axs in axs:
                    for ax in row_axs:
                        ax.text(
                            0.5,
                            0.5,
                            "No AOS Data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                continue

            # Get donut stamps for this detector
            idx_to_donuts = np.array(donutStampsIntra.metadata.getArray("DET_NAME")) == det_name
            donut_stamps_sel = np.array(donutStampsIntra)[idx_to_donuts]

            if len(donut_stamps_sel) == 0:
                # No donut stamps for this detector
                for row_axs in axs:
                    for ax in row_axs:
                        ax.text(
                            0.5,
                            0.5,
                            "No Donut Stamps",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                continue

            # Get centroid positions (computed later per-stamp when needed)

            # Donut-stamp-driven: try to collect up to 4 valid donuts
            # Skip stamps on modeling failure
            donut_data: list = []
            used_aos_indices: set[int] = set()  # ensure one-to-one stamp↔AOS mapping
            match_threshold_px = float(self.config.matchThresholdPx)
            dup_threshold_px = float(self.config.duplicateThresholdPx)  # avoid plotting near-duplicate stamps
            selected_stamp_positions: list[tuple[float, float]] = []

            for istamp, stamp in enumerate(donut_stamps_sel):
                if len(donut_data) >= 4:
                    break

                stamp_x = stamp.centroid_position.x
                stamp_y = stamp.centroid_position.y
                # Skip if this stamp is essentially a duplicate
                # of one already selected
                if any(
                    np.hypot(stamp_x - sx, stamp_y - sy) < dup_threshold_px
                    for sx, sy in selected_stamp_positions
                ):
                    continue
                # Find closest AOS catalog entry to this donut stamp
                dists = np.hypot(rows["centroid_x"] - stamp_x, rows["centroid_y"] - stamp_y)

                if len(dists) == 0:
                    self.log.info("No AOS data to match for stamp %d", istamp + 1)
                    continue  # No AOS data to match

                # Enforce unique AOS rows and proximity threshold
                candidate_indices = [i for i in range(len(rows)) if i not in used_aos_indices]
                if not candidate_indices:
                    self.log.info("No available AOS candidates for stamp %d", istamp + 1)
                    continue
                candidate_dists = dists[candidate_indices]
                rel_idx = int(np.argmin(candidate_dists))
                best_aos_idx = int(candidate_indices[rel_idx])
                min_dist = float(dists[best_aos_idx])
                if not np.isfinite(min_dist) or min_dist > match_threshold_px:
                    # too far or invalid match; skip
                    self.log.info(
                        "AOS match too far (%.1f px) or invalid for stamp %d",
                        min_dist,
                        istamp + 1,
                    )
                    continue
                row = rows[best_aos_idx]

                # Get detector orientation for image rotation
                nquarter = det.getOrientation().getNQuarter() % 4

                # Extract and rotate image
                img = np.rot90(stamp.stamp_im.image.array[1:, 1:], -nquarter).T

                # Generate model using the matched AOS catalog parameters;
                # skip on failure
                try:
                    model, fwhm = self.getModelUnpaired(
                        telescope,
                        intra_telescope,
                        extra_telescope,
                        row,
                        img,
                        wavelength,
                        det_name,
                    )
                except Exception as e:
                    self.log.exception(
                        "Skipping stamp %d for %s due to model error: %s",
                        istamp + 1,
                        det_name,
                        e,
                    )
                    continue

                # Store data for this donut
                donut_data.append(
                    {
                        "img": img,
                        "model": model,
                        "row": row,
                        # Prefer AOS ood_score; fallback via donut table
                        "ood_score": float(row["ood_score"]) if ("ood_score" in row.columns) else np.nan,
                        "blur": (float(blur[best_aos_idx]) if best_aos_idx < len(blur) else 0.0),
                        "match_dist": min_dist,
                        "focal_type": "intra" if det_name.endswith("SW1") else "extra",
                        "obs_centroid": (float(stamp_x), float(stamp_y)),
                        "aos_centroid": (
                            float(row["centroid_x"]),
                            float(row["centroid_y"]),
                        ),
                        "avg_zk_CCS": avg_zk_for_detector,
                        "avg_row": avg_row_for_detector,
                        "manual_avg_zk": manual_avg_zk,
                        "manual_avg_row": manual_avg_row,
                    }
                )
                # If NaN and table has ood_score, match nearest centroid
                if (
                    np.isnan(donut_data[-1]["ood_score"])
                    and len(donut_table_rows) > 0
                    and ("ood_score" in donut_table_rows.colnames)
                ):
                    try:
                        if ("centroid_x" in donut_table_rows.colnames) and (
                            "centroid_y" in donut_table_rows.colnames
                        ):
                            dtd = donut_table_rows
                            d_dists = np.hypot(dtd["centroid_x"] - stamp_x, dtd["centroid_y"] - stamp_y)
                            didx = int(np.argmin(d_dists))
                            donut_data[-1]["ood_score"] = float(dtd["ood_score"][didx])
                    except Exception as e:
                        self.log.warning(
                            "Failed to match ood_score from donut_table for %s stamp %d: %s",
                            det_name,
                            istamp + 1,
                            e,
                        )
                # Reserve this AOS entry so it isn't reused by another stamp
                used_aos_indices.add(best_aos_idx)
                # Remember this stamp position to avoid near-duplicates
                selected_stamp_positions.append((stamp_x, stamp_y))

            # Create visualization for all donuts
            self.plotResultsMultiDonut(
                axs,
                donut_data,
                telescope,
                intra_telescope,
                extra_telescope,
                wavelength,
                det_name,
            )

            # Plot combined ZK in row 10
            zk_combined_ax = axs[10][0]
            self._render_combined_zk(zk_combined_ax, donut_data)

            # Plot histogram of ood_score in row 11
            histogram_ax = axs[11][0]
            self._render_histogram(histogram_ax, donut_table_rows, rows, donut_data, det_name)

            # Add detector label at the top of the subplot (within the panel)
            focal_color = "blue" if det_name.endswith("SW1") else "red"

            # Add label at the top of the first row of the detector subplot
            axs[0][0].text(
                0.5,
                1.15,  # Position above the first row
                f"{det_name}",
                fontsize=14,
                fontweight="bold",
                color=focal_color,
                ha="center",
                va="center",
                transform=axs[0][0].transAxes,
            )

        # Add row labels on the left side
        row_labels = [
            "Obs",  # Observed donut
            "Ind Mdl",  # Individual model (from per-donut Zernikes)
            "Ind Res",  # Individual residual (Obs - Ind Model)
            "ZK",  # Zernike coefficients
            "Man Mdl",  # Manual average model (from manually averaged raw Zernikes)
            "Man Res",  # Manual average residual (Obs - Man Model)
            "Man-Ind",  # Manual-Individual difference (Man Model - Ind Model)
            "Corr Mdl",  # Corrected model (from detector-averaged Zernikes)
            "Corr Res",  # Corrected residual (Obs - Corr Model)
            "Corr-Man",  # Corrected-Manual difference (Corr Model - Man Model)
            "ZK Combo",  # Combined Zernike scatter for donuts, manual, corrected
            "OOD Hist",  # Out-of-distribution score histogram
        ]

        # Place labels using actual axis centers for perfect vertical alignment
        for i, label in enumerate(row_labels):
            if top_row_centers[i] is not None:
                fig.text(
                    0.02,
                    top_row_centers[i],
                    label,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    rotation=90,
                )
            if bottom_row_centers[i] is not None:
                fig.text(
                    0.02,
                    bottom_row_centers[i],
                    label,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    rotation=90,
                )

        # Add telescope control data at the very bottom of the figure
        fig.text(
            0.5,
            0.02,  # Bottom center of figure
            f"Telescope Control Data - Visit: {visit_logevent} | "
            f"Filter: {record.physical_filter} | "
            f"Elevation: {90 - record.zenith_angle if record.zenith_angle else 90:.1f}° | "
            f"Azimuth: {record.azimuth if record.azimuth else 0:.1f}°",
            ha="center",
            va="bottom",
            transform=fig.transFigure,
            fontsize=12,
            fontweight="bold",
        )

        return fig
