from collections import defaultdict
from functools import lru_cache

import astropy.units as u
import batoid
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
import numpy as np
from astropy.coordinates import Angle
from astropy.table import QTable, vstack
from batoid_rubin import LSSTBuilder
from galsim.zernike import zernikeBasis
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import Stamp, Stamps, SubtractBackgroundTask
from lsst.pex.exceptions import LengthError
from lsst.summit.utils.efdUtils import getEfdData, getMostRecentRowWithDataBefore
from lsst.ts.ofc import BendModeToForce, OFCData
from lsst_efd_client import EfdClient
from matplotlib.figure import Figure
from scipy.signal import correlate
from skimage.feature import peak_local_max

__all__ = [
    "HartmannSensitivityAnalysisConfig",
    "HartmannSensitivityAnalysisConnections",
    "HartmannSensitivityAnalysis",
]


def ccs_to_ocs(x_ccs: float, y_ccs: float, rtp: Angle) -> tuple[float, float]:
    crtp, srtp = np.cos(rtp), np.sin(rtp)
    x_ocs = x_ccs * crtp + y_ccs * srtp
    y_ocs = -x_ccs * srtp + y_ccs * crtp
    return x_ocs, y_ocs


def ocs_to_ccs(x_ocs: float, y_ocs: float, rtp: Angle) -> tuple[float, float]:
    return ccs_to_ocs(x_ocs, y_ocs, -rtp)


def ccs_to_dvcs(x_ccs: float, y_ccs: float) -> tuple[float, float]:
    return y_ccs, x_ccs


def dvcs_to_ccs(x_dvcs: float, y_dvcs: float) -> tuple[float, float]:
    return y_dvcs, x_dvcs


def trace_ocs_to_ccd(
    telescope, x_ocs: float, y_ocs: float, det, rtp: Angle
) -> tuple[float, float]:
    cr = batoid.RayVector.fromStop(
        x=0.0, y=0.0, optic=telescope, wavelength=700e-9, theta_x=x_ocs, theta_y=y_ocs
    )
    fp = telescope.trace(cr)  # OCS FOCAL_PLANE
    x_fp_ocs = fp.x[0] * 1e3  # m to mm
    y_fp_ocs = fp.y[0] * 1e3  # m to mm
    x_fp_ccs, y_fp_ccs = ocs_to_ccs(x_fp_ocs, y_fp_ocs, rtp)
    x_fp_dvcs, y_fp_dvcs = ccs_to_dvcs(x_fp_ccs, y_fp_ccs)
    x_ccd_dvcs, y_ccd_dvcs = det.transform(
        Point2D(x_fp_dvcs, y_fp_dvcs), FOCAL_PLANE, PIXELS
    )
    return x_ccd_dvcs, y_ccd_dvcs


@lru_cache
def get_m2_bmf():
    """Get the M2 BendModeToForce object"""
    return BendModeToForce("M2", OFCData("lsst"))


@lru_cache
def get_m1m3_bmf():
    """Get the M1M3 BendModeToForce object"""
    return BendModeToForce("M1M3", OFCData("lsst"))


def get_state(exposure, efd_client):
    visitInfo = exposure.info.getVisitInfo()
    begin = visitInfo.date.toAstropy()
    begin -= visitInfo.exposureTime * u.s / 2
    end = begin + visitInfo.exposureTime * u.s

    out = np.zeros(50, dtype=np.float64)

    m2_df = getMostRecentRowWithDataBefore(
        efd_client,
        "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
        timeToLookBefore=end,
        where=lambda df: df["salIndex"] == 2,
    )
    out[0] = m2_df["z"]
    out[1] = m2_df["x"]
    out[2] = m2_df["y"]
    out[3] = m2_df["u"]
    out[4] = m2_df["v"]

    cam_df = getMostRecentRowWithDataBefore(
        efd_client,
        "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
        timeToLookBefore=end,
        where=lambda df: df["salIndex"] == 1,
    )
    out[5] = cam_df["z"]
    out[6] = cam_df["x"]
    out[7] = cam_df["y"]
    out[8] = cam_df["u"]
    out[9] = cam_df["v"]

    m1m3_event = getMostRecentRowWithDataBefore(
        efd_client,
        "lsst.sal.MTM1M3.logevent_appliedActiveOpticForces",
        timeToLookBefore=end,
    )
    m1m3_forces = np.empty((156,))
    for i in range(156):
        m1m3_forces[i] = m1m3_event[f"zForces{i}"]
    out[10:30] = get_m1m3_bmf().bending_mode(m1m3_forces)

    m2_telemetry = QTable.from_pandas(
        getEfdData(
            efd_client,
            "lsst.sal.MTM2.axialForce",
            begin=begin,
            end=end,
        )
    )
    nrow = len(m2_telemetry)
    m2_forces = np.empty((nrow, 72), dtype=np.float64)
    for i in range(72):
        m2_forces[:, i] = m2_telemetry[f"applied{i}"]
    m2_forces = np.mean(m2_forces, axis=0)
    out[30:] = get_m2_bmf().bending_mode(m2_forces)

    return out


def get_parabola_vertex(v_m1, v_0, v_p1):
    denom = 2 * (v_m1 - 2 * v_0 + v_p1)
    if denom == 0:
        return 0
    return (v_m1 - v_p1) / denom


def get_offset(ref, comp, search_radius=10):
    # Determine offset via cross-correlation
    rsh = ref.shape
    csh = comp.shape

    # No implementation if ref larger than comp
    assert rsh[0] <= csh[0] and rsh[1] <= csh[1]

    # No implementation for even shapes
    assert rsh[0] % 2 == 1 and rsh[1] % 2 == 1
    assert csh[0] % 2 == 1 and csh[1] % 2 == 1

    max_search_x = (csh[1] - rsh[1]) // 2
    max_search_y = (csh[0] - rsh[0]) // 2

    if search_radius > max_search_x:
        clip_x = search_radius - max_search_x
        ref = ref[:, clip_x:-clip_x]
    if search_radius > max_search_y:
        clip_y = search_radius - max_search_y
        ref = ref[clip_y:-clip_y, :]
    rsh = ref.shape

    corr = np.zeros((2 * search_radius + 1, 2 * search_radius + 1))
    for j, dy in enumerate(range(-search_radius, search_radius + 1)):
        for i, dx in enumerate(range(-search_radius, search_radius + 1)):
            ymin = (csh[0] - rsh[0]) // 2 + dy
            ymax = ymin + rsh[0]
            xmin = (csh[1] - rsh[1]) // 2 + dx
            xmax = xmin + rsh[1]
            crop = comp[ymin:ymax, xmin:xmax]

            corr[j, i] = np.sum(ref * crop) / np.sqrt(np.sum(ref**2) * np.sum(crop**2))

    best_ji = np.unravel_index(np.argmax(corr), corr.shape)
    best_offset = (best_ji[0] - search_radius, best_ji[1] - search_radius)

    if best_ji[0] == 0 or best_ji[0] == 2 * search_radius:
        return (np.nan, np.nan)
    if best_ji[1] == 0 or best_ji[1] == 2 * search_radius:
        return (np.nan, np.nan)

    dx_sub = get_parabola_vertex(
        corr[best_ji[0], best_ji[1] - 1],
        corr[best_ji[0], best_ji[1]],
        corr[best_ji[0], best_ji[1] + 1],
    )
    dy_sub = get_parabola_vertex(
        corr[best_ji[0] - 1, best_ji[1]],
        corr[best_ji[0], best_ji[1]],
        corr[best_ji[0] + 1, best_ji[1]],
    )

    best_offset = (best_offset[0] + dy_sub, best_offset[1] + dx_sub)

    return best_offset


def match_patches(img, ref, xs, ys, patch_size, search_radius):
    dx_out = []
    dy_out = []

    for x, y in zip(xs, ys):
        xmin = x - patch_size // 2
        xmax = x + patch_size // 2 + 1
        ymin = y - patch_size // 2
        ymax = y + patch_size // 2 + 1
        ref_crop = ref[ymin:ymax, xmin:xmax]
        if xmin < 0 or ymin < 0 or xmax > ref.shape[1] or ymax > ref.shape[1]:
            dy_out.append(np.nan)
            dx_out.append(np.nan)
            continue

        xmin_crop = xmin - search_radius
        xmax_crop = xmax + search_radius
        ymin_crop = ymin - search_radius
        ymax_crop = ymax + search_radius
        if (
            xmin_crop < 0
            or ymin_crop < 0
            or xmax_crop > img.shape[1]
            or ymax_crop > img.shape[0]
        ):
            dy_out.append(np.nan)
            dx_out.append(np.nan)
            continue
        img_crop = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
        offset = get_offset(ref_crop, img_crop, search_radius)
        dy_out.append(offset[0])
        dx_out.append(offset[1])
    return dx_out, dy_out


def align_offsets(fx, fy, dfx, dfy):
    # Solve \Delta x = dx - y dtheta
    #       \Delta y = dy + x dtheta
    # for dx, dy, dtheta.
    design = np.zeros((2 * len(fx), 3))
    design[: len(fx), 0] = 1.0
    design[len(fx) :, 1] = 1.0
    design[: len(fx), 2] = -fy
    design[len(fx) :, 2] = fx
    target = np.concatenate([dfx, dfy])
    w = np.isfinite(target)
    solution, *_ = np.linalg.lstsq(design[w], target[w])
    delta = design @ solution
    return dfx - delta[: len(fx)], dfy - delta[len(fx) :]


def stamp_sets_to_stamps(stamp_sets):
    all_stamps = []
    metadata = defaultdict(list)
    for stamp_set in stamp_sets:
        nstamp = 1 + len(stamp_set["tests"])
        all_stamps.append(stamp_set["ref"])
        all_stamps.extend(stamp_set["tests"])
        metadata["EXP_ID"].extend([stamp_set["ref_id"]] + stamp_set["test_ids"])
        metadata["DONUT_ID"].extend([stamp_set["donut_id"]] * nstamp)
        metadata["REF_ID"].extend([stamp_set["ref_id"]] * nstamp)
        metadata["X_REF"].extend([stamp_set["x_ref_ccd_dvcs"]] * nstamp)
        metadata["Y_REF"].extend([stamp_set["y_ref_ccd_dvcs"]] * nstamp)
        metadata["OFFSET_X"].extend([0.0] + [off[1] for off in stamp_set["offsets"]])
        metadata["OFFSET_Y"].extend([0.0] + [off[0] for off in stamp_set["offsets"]])
    out = Stamps(all_stamps)
    for key, val in metadata.items():
        out.metadata[key] = np.array(val)

    return out


def stamps_to_stamp_sets(stamps):
    if len(stamps) == 0:
        return []
    metadata = stamps.metadata
    donut_ids = np.array(metadata.getArray("DONUT_ID"))
    exp_ids = np.array(metadata.getArray("EXP_ID"))
    ref_ids = np.array(metadata.getArray("REF_ID"))

    stamp_sets = []
    for donut_id in np.unique(donut_ids):
        ref_donut_idx = np.where((donut_ids == donut_id) & (exp_ids == ref_ids))[0][0]
        ref = stamps[ref_donut_idx]
        test_idxs = np.where((donut_ids == donut_id) & (exp_ids != ref_ids))[0]
        tests = [stamps[i] for i in test_idxs]
        offset_x = [metadata.getArray("OFFSET_X")[i] for i in test_idxs]
        offset_y = [metadata.getArray("OFFSET_Y")[i] for i in test_idxs]
        offsets = list(zip(offset_y, offset_x))

        stamp_set = {}
        stamp_set["donut_id"] = donut_id
        stamp_set["ref"] = ref
        stamp_set["x_ref_CCD_DVCS"] = metadata.getArray("X_REF")[ref_donut_idx]
        stamp_set["y_ref_CCD_DVCS"] = metadata.getArray("Y_REF")[ref_donut_idx]
        stamp_set["tests"] = tests
        stamp_set["offsets"] = offsets
        stamp_set["ref_id"] = metadata.getArray("REF_ID")[ref_donut_idx]
        stamp_set["test_ids"] = [metadata.getArray("EXP_ID")[i] for i in test_idxs]

        stamp_sets.append(stamp_set)

    return stamp_sets


def fit_displacements(
    fx,
    fy,
    dfx,
    dfy,
    radius,
):
    # Filter outliers in absolute displacement
    dfr = np.hypot(dfx, dfy)
    quantiles = np.nanquantile(dfr, [0.25, 0.5, 0.75])
    threshold = quantiles[0] + 4.0 * np.ptp(quantiles[[0, 2]])

    flag_0 = dfr < threshold

    # Fit Zks to inliers
    zkBasis = zernikeBasis(28, fx, fy, R_outer=radius, R_inner=radius * 0.62)
    wgood = np.isfinite(dfx) & np.isfinite(dfy) & flag_0
    dx_coefs, *_ = np.linalg.lstsq(zkBasis.T[wgood], dfx[wgood])
    dy_coefs, *_ = np.linalg.lstsq(zkBasis.T[wgood], dfy[wgood])
    dx_fit = zkBasis.T @ dx_coefs
    dy_fit = zkBasis.T @ dy_coefs

    # Filter again on residuals to Zk fit
    ddx = dfx - dx_fit
    ddy = dfy - dy_fit
    xquant = np.nanquantile(ddx, [0.25, 0.5, 0.75])
    yquant = np.nanquantile(ddy, [0.25, 0.5, 0.75])
    xiqr = np.ptp(xquant[[0, 2]])
    yiqr = np.ptp(yquant[[0, 2]])
    xmed = xquant[1]
    ymed = yquant[1]
    xgood = (ddx > xmed - 3 * xiqr) & (ddx < xmed + 3 * xiqr)
    ygood = (ddy > ymed - 3 * yiqr) & (ddy < ymed + 3 * yiqr)
    good = np.isfinite(dfx) & np.isfinite(dfy) & xgood & ygood

    # Fit once more for the plot
    dx_coefs, *_ = np.linalg.lstsq(zkBasis.T[good], dfx[good])
    dy_coefs, *_ = np.linalg.lstsq(zkBasis.T[good], dfy[good])
    dx_fit = zkBasis.T @ dx_coefs
    dy_fit = zkBasis.T @ dy_coefs
    return dx_fit, dy_fit, good


class HartmannSensitivityAnalysisConnections(
    pipeBase.PipelineTaskConnections, dimensions=("group", "instrument", "detector")
):
    exposures = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
        multiple=True,
    )
    hartmann_stamps = ct.Output(
        doc="Output Hartmann stamps",
        dimensions=("group", "instrument", "detector"),
        storageClass="Stamps",
        name="hartmann_stamps",
    )
    hartmann_detection_table = ct.Output(
        doc="Output Hartmann detection table",
        dimensions=("group", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_detection_table",
    )
    hartmann_analysis_table = ct.Output(
        doc="Output Hartmann sensitivity analysis table",
        dimensions=("group", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_analysis_table",
    )
    hartmann_unfiltered_plot = ct.Output(
        doc="Output Hartmann unfiltered plot",
        dimensions=("group", "instrument", "detector"),
        storageClass="Plot",
        name="hartmann_unfiltered_plot",
    )
    hartmann_filtered_plot = ct.Output(
        doc="Output Hartmann filtered plot",
        dimensions=("group", "instrument", "detector"),
        storageClass="Plot",
        name="hartmann_filtered_plot",
    )


class HartmannSensitivityAnalysisConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=HartmannSensitivityAnalysisConnections,
):
    ref_index = pexConfig.Field[int](
        doc="Index of the reference exposure within the group",
        default=-1,
    )
    m2_dz = pexConfig.Field[float](
        doc="Defocus offset for M2 in mm",
        default=4.0,
    )
    cam_dz = pexConfig.Field[float](
        doc="Defocus offset for the camera in mm",
        default=4.0,
    )
    min_flux = pexConfig.Field[float](
        doc="Minimum flux for analysis",
        default=3e6,
    )
    max_inner_ratio = pexConfig.Field[float](
        doc="Maximum inner/total flux ratio for analysis",
        default=0.03,
    )
    max_outer_ratio = pexConfig.Field[float](
        doc="Maximum outer/total flux ratio for analysis",
        default=0.03,
    )
    do_plot = pexConfig.Field[bool](
        doc="Whether to make plots",
        default=True,
    )
    bin_size = pexConfig.Field[int](
        doc="Bin size for running initial detection",
        default=8,
    )
    rng_seed = pexConfig.Field[int](
        doc="Random number seed",
        default=57721,
    )
    n_pupil_positions = pexConfig.Field[int](
        doc="Number of random pupil positions to choose",
        default=500,
    )
    max_donuts = pexConfig.Field[int](
        doc="Maximum number of donuts to analyze",
        default=3,
    )
    max_exp_plot = pexConfig.Field[int](
        doc="Maximum number of exposures to plot",
        default=4,
    )
    isr = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="Instrument signature removal task",
    )
    subtractBackground = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )

    def setDefaults(self):
        self.isr.doAmpOffset = False
        self.isr.ampOffset.doApplyAmpOffset = False
        # Turn off slow steps in ISR but mask saturated pixels
        self.isr.doBrighterFatter = False
        self.isr.doSaturation = True
        self.isr.crosstalk.doQuadraticCrosstalkCorrection = False
        self.isr.qa.saveStats = False
        self.isr.doStandardStatistics = False
        self.isr.doInterpolate = False
        self.isr.doVariance = False
        self.isr.doDeferredCharge = False
        self.isr.doDefect = False
        self.isr.doApplyGains = True
        self.isr.doBias = False
        self.isr.doFlat = True
        self.isr.doDark = False
        self.isr.doLinearize = False
        self.isr.doSuspect = False
        self.isr.doSetBadRegions = False
        self.isr.doBootstrap = False
        self.isr.doCrosstalk = False
        self.isr.doITLEdgeBleedMask = False


class HartmannSensitivityAnalysis(
    pipeBase.PipelineTask,
):
    ConfigClass = HartmannSensitivityAnalysisConfig
    _DefaultName = "HartmannSensitivityAnalysis"

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.makeSubtask("isr")
        self.makeSubtask("subtractBackground")
        self._display = display

        self._defocus = abs(self.config.m2_dz + self.config.cam_dz)
        self._donut_diam = 680.0 / 8.0 * self._defocus
        self._donut_radius = self._donut_diam / 2
        self._binned_template_size = int(self._donut_diam * 1.15 / self.config.bin_size)
        if self._binned_template_size % 2 == 0:
            self._binned_template_size += 1
        self._template_size = int(self._donut_diam * 1.15)
        if self._template_size % 2 == 0:
            self._template_size += 1

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, exposures, run_isr=False, **isr_kwargs):
        config = self.config

        reference_exposure, test_exposures = self.prepare_exposures(
            exposures, config.ref_index, run_isr=run_isr, **isr_kwargs
        )
        detections = self.detect(reference_exposure)

        if self._display is not None:
            self.update_display(reference_exposure, detections)

        if sum(detections["use"]) == 0:
            stamp_sets = []
        else:
            efd_client = EfdClient("usdf_efd")
            ref_state = get_state(reference_exposure, efd_client)
            test_states = [get_state(exp, efd_client) for exp in test_exposures]

            # Make optics
            band = reference_exposure.filter.bandLabel
            ref_telescope = self.get_telescope(band, ref_state - ref_state)
            test_telescopes = [
                self.get_telescope(band, ts - ref_state) for ts in test_states
            ]

            # Initial alignment of donuts
            stamp_sets = self.get_aligned_stamp_sets(
                reference_exposure,
                test_exposures,
                ref_telescope,
                test_telescopes,
                detections,
            )

        patch_table = self.match_all_patches(stamp_sets)
        self.remove_net_shift_and_rotation(patch_table)
        self.fit_displacements(patch_table)

        if self.config.do_plot:
            self.log.info("Making plots")
            initial_fig = self.plot_initial(stamp_sets, patch_table)
            filtered_fig = self.plot_filtered(stamp_sets, patch_table)
        else:
            initial_fig = None
            filtered_fig = None

        return pipeBase.Struct(
            hartmann_detection_table=detections,
            hartmann_stamps=stamp_sets_to_stamps(stamp_sets),
            stamp_sets=stamp_sets,
            hartmann_analysis_table=patch_table,
            hartmann_unfiltered_plot=initial_fig,
            hartmann_filtered_plot=filtered_fig,
        )

    def prepare_exposures(self, exposures, ref_index, run_isr=False, **isr_kwargs):
        exposures.sort(key=lambda exp: exp.getInfo().getVisitInfo().id)
        if run_isr:
            self.log.info("Running ISR on %d exposures", len(exposures))
            exposures = [self.isr.run(exp, **isr_kwargs).exposure for exp in exposures]
        self.log.info("Subtracting background")
        for exposure in exposures:
            self.subtractBackground.run(exposure=exposure)

        # Sort exposures and pick reference
        if ref_index < 0:
            ref_index = len(exposures) + ref_index
        reference_exposure = exposures[ref_index]
        ref_id = reference_exposure.info.getVisitInfo().id
        self.log.info("Using exposure %d as reference", ref_id)
        test_exposures = [exp for i, exp in enumerate(exposures) if i != ref_index]
        self.log.info("Using %d test exposures", len(test_exposures))
        for exp in test_exposures:
            test_id = exp.info.getVisitInfo().id
            self.log.info("  test exposure: %d", test_id)
        return reference_exposure, test_exposures

    def detect(self, exposure):
        visit_info = exposure.info.getVisitInfo()
        rsp = visit_info.boresightRotAngle.asDegrees() * u.deg
        q = visit_info.boresightParAngle.asDegrees() * u.deg
        rtp = q - rsp - 90.0*u.deg

        template = np.zeros(
            (self._binned_template_size, self._binned_template_size), dtype=float
        )
        y, x = np.ogrid[
            -self._binned_template_size // 2 : self._binned_template_size // 2,
            -self._binned_template_size // 2 : self._binned_template_size // 2,
        ]
        r = np.hypot(x, y)
        binned_radius = self._donut_radius / self.config.bin_size
        template[r < binned_radius] = 1.0
        template[r < binned_radius * 0.62] = 0.0
        inner_hole = np.zeros_like(template)
        inner_hole[r < binned_radius * 0.55] = 1.0
        outer_annulus = np.zeros_like(template)
        outer_annulus[(r >= binned_radius * 1.05) & (r < binned_radius * 1.15)] = 1.0

        exp = exposure.clone()
        mi = exp.getMaskedImage()
        binned = afwMath.binImage(mi, self.config.bin_size)
        exp.setMaskedImage(binned)
        arr = exp.image.array
        mask = exp.mask.array

        # Histogram equalize since we care more about connected points above
        # threshold than actual flux values.
        cdf = np.nanquantile(arr, np.linspace(0, 1, 256))
        heq = np.digitize(arr, cdf)
        det = correlate(heq, template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(2 * 0.8 * binned_radius),
            exclude_border=int(binned_radius * 1.15),
        )

        table = QTable()
        table["idx"] = np.arange(len(peaks), dtype=np.int32)
        table["x_ref_ccd_dvcs"] = (peaks[:, 1] * self.config.bin_size).astype(np.int32)
        table["y_ref_ccd_dvcs"] = (peaks[:, 0] * self.config.bin_size).astype(np.int32)
        fluxes = []
        inner_fluxes = []
        outer_fluxes = []
        x_field_ccs_list = []
        y_field_ccs_list = []
        x_field_ocs_list = []
        y_field_ocs_list = []
        for peak in peaks:
            x_ccd_dvcs = peak[1] * self.config.bin_size
            y_ccd_dvcs = peak[0] * self.config.bin_size
            # Get OCS coordinates
            x_field_dvcs, y_field_dvcs = exposure.getDetector().transform(
                Point2D(x_ccd_dvcs, y_ccd_dvcs), PIXELS, FIELD_ANGLE
            )
            x_field_ccs, y_field_ccs = dvcs_to_ccs(x_field_dvcs, y_field_dvcs)
            x_field_ocs, y_field_ocs = ccs_to_ocs(x_field_ccs, y_field_ccs, rtp)
            x_field_ccs_list.append(x_field_ccs)
            y_field_ccs_list.append(y_field_ccs)
            x_field_ocs_list.append(x_field_ocs)
            y_field_ocs_list.append(y_field_ocs)

            xmin = peak[1] - self._binned_template_size // 2
            xmax = peak[1] + self._binned_template_size // 2 + 1
            ymin = peak[0] - self._binned_template_size // 2
            ymax = peak[0] + self._binned_template_size // 2 + 1
            stamp = arr[ymin:ymax, xmin:xmax]
            bad_mask_planes = ["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA"]
            bitmask = exp.mask.getPlaneBitMask(bad_mask_planes)
            msk = (mask[ymin:ymax, xmin:xmax] & bitmask) != 0

            use = xmin >= 0
            use = use and ymin >= 0
            use = use and xmax <= arr.shape[1]
            use = use and ymax <= arr.shape[0]
            use = use and np.sum(msk) == 0
            if not use:
                fluxes.append(np.nan)
                inner_fluxes.append(np.nan)
                outer_fluxes.append(np.nan)
                continue

            # Guess that a reasonable background estimate is the median of the
            # inner and outer pixels
            bkg = np.nanmedian(stamp * (np.maximum(inner_hole, outer_annulus)))
            stamp = stamp - bkg
            fluxes.append(np.nansum(stamp * template))
            inner_fluxes.append(np.nansum(stamp * inner_hole))
            outer_fluxes.append(np.nansum(stamp * outer_annulus))
        table["x_ref_field_ocs"] = np.array(x_field_ocs_list, dtype=np.float32)
        table["y_ref_field_ocs"] = np.array(y_field_ocs_list, dtype=np.float32)
        table["x_ref_field_ccs"] = np.array(x_field_ccs_list, dtype=np.float32)
        table["y_ref_field_ccs"] = np.array(y_field_ccs_list, dtype=np.float32)
        table["flux"] = np.array(fluxes, dtype=np.float32)
        table["inner_flux"] = np.array(inner_fluxes, dtype=np.float32)
        table["outer_flux"] = np.array(outer_fluxes, dtype=np.float32)
        table["inner_ratio"] = table["inner_flux"] / table["flux"]
        table["outer_ratio"] = table["outer_flux"] / table["flux"]
        table["use"] = table["flux"] > self.config.min_flux
        table["use"] &= table["inner_ratio"] < self.config.max_inner_ratio
        table["use"] &= table["outer_ratio"] < self.config.max_outer_ratio

        # Filter to max_donuts brightest
        subtable = table[table["use"]]
        subtable.sort(keys="flux", reverse=True)
        idxs = subtable["idx"][: self.config.max_donuts]
        table["use"][~np.isin(table["idx"], idxs)] = False

        self.log.info("Detected %d donuts", len(table))
        self.log.info("Selected %d donuts for analysis", np.sum(table["use"]))

        return table

    def get_telescope(self, band, state):
        # Handle sign flips and unit conversions
        state = np.array(state)
        state[[0, 1, 3, 5, 6, 8]] *= -1  # z and x are flipped
        state[30:] *= -1  # M2 modes are flipped
        state[[3, 4, 8, 9]] *= 3600  # deg to arcsec

        fiducial = (
            batoid.Optic.fromYaml(f"LSST_{band}.yaml")
            .withGloballyShiftedOptic("M2", [0, 0, self.config.m2_dz * 1e-3])
            .withGloballyShiftedOptic("LSSTCamera", [0, 0, self.config.cam_dz * 1e-3])
        )

        builder = LSSTBuilder(
            fiducial,
            fea_dir="/home/j/jmeyers3/.local/batoid_rubin_data",
            bend_dir="/home/j/jmeyers3/.local/batoid_rubin_data",
        )
        builder = builder.with_aos_dof(state)
        return builder.build()

    def get_aligned_stamp_sets(
        self,
        reference_exposure,
        test_exposures,
        reference_telescope,
        test_telescopes,
        detections,
    ):
        self.log.info("Aligning detections")
        visit_info = reference_exposure.info.getVisitInfo()
        rsp = visit_info.boresightRotAngle.asDegrees() * u.deg
        q = visit_info.boresightParAngle.asDegrees() * u.deg
        rtp = q - rsp - 90.0 * u.deg
        stamp_size = self._template_size
        stamp_sets = []
        for detection in detections:
            if not detection["use"]:
                continue
            self.log.info(
                "  Aligning detection %d at (x,y)=(%d,%d)",
                detection["idx"],
                detection["x_ref_ccd_dvcs"],
                detection["y_ref_ccd_dvcs"],
            )
            x_ref_ccd_dvcs, y_ref_ccd_dvcs = detection[["x_ref_ccd_dvcs", "y_ref_ccd_dvcs"]]
            x_ref_field_ocs, y_ref_field_ocs = detection[["x_ref_field_ocs", "y_ref_field_ocs"]]
            x_ref_predict, y_ref_predict = trace_ocs_to_ccd(
                reference_telescope,
                x_ref_field_ocs,
                y_ref_field_ocs,
                reference_exposure.getDetector(),
                rtp
            )

            # Extract stamp from reference
            xmin = x_ref_ccd_dvcs - stamp_size // 2
            xmax = x_ref_ccd_dvcs + stamp_size // 2 + 1
            ymin = y_ref_ccd_dvcs - stamp_size // 2
            ymax = y_ref_ccd_dvcs + stamp_size // 2 + 1
            box = Box2I(Point2I(xmin, ymin), Extent2I(xmax - xmin, ymax - ymin))
            try:
                ref_stamp = Stamp(reference_exposure.maskedImage[box])
            except LengthError:
                detection["use"] = False
                self.log.warn(
                    "  Could not extract reference stamp for detection %d; skipping",
                    detection["idx"],
                )
                continue
            ref_stamp_arr = ref_stamp.stamp_im.image.array

            offsets = []
            test_stamps = []
            for test_exposure, test_telescope in zip(test_exposures, test_telescopes):
                # Get expected position in test exposure
                x_test_predict, y_test_predict = trace_ocs_to_ccd(
                    test_telescope,
                    x_ref_field_ocs,
                    y_ref_field_ocs,
                    test_exposure.getDetector(),
                    rtp
                )
                test_xmin = xmin + int(round(x_test_predict - x_ref_predict))
                test_xmax = xmax + int(round(x_test_predict - x_ref_predict))
                test_ymin = ymin + int(round(y_test_predict - y_ref_predict))
                test_ymax = ymax + int(round(y_test_predict - y_ref_predict))
                test_box = Box2I(
                    Point2I(test_xmin, test_ymin),
                    Extent2I(test_xmax - test_xmin, test_ymax - test_ymin),
                )
                if (
                    test_xmin < 0
                    or test_ymin < 0
                    or test_xmax > test_exposure.image.array.shape[1]
                    or test_ymax > test_exposure.image.array.shape[0]
                ):
                    detection["use"] = False
                    self.log.warn(
                        "  Aligned stamp for detection %d in exposure %d would be out of bounds; skipping",
                        detection["idx"],
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                test_stamp = Stamp(test_exposure.maskedImage[test_box])
                test_stamp_arr = test_stamp.stamp_im.image.array
                offset = get_offset(ref_stamp_arr, test_stamp_arr, search_radius=60)
                if not np.isfinite(offset).all():
                    detection["use"] = False
                    self.log.warn(
                        "  Could not align detection %d in exposure %d; skipping",
                        detection["idx"],
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                offsets.append(offset)
                test_box = Box2I(
                    Point2I(
                        test_xmin + int(round(offset[1])),
                        test_ymin + int(round(offset[0])),
                    ),
                    Extent2I(test_xmax - test_xmin, test_ymax - test_ymin),
                )
                try:
                    test_stamp = Stamp(test_exposure.maskedImage[test_box])
                except LengthError:
                    detection["use"] = False
                    self.log.warn(
                        "  Could not extract aligned stamp for detection %d in exposure %d; skipping",
                        detection["idx"],
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                test_stamps.append(test_stamp)
            if not detection["use"]:
                continue
            test_stamps = Stamps(test_stamps)
            stamp_sets.append(
                dict(
                    donut_id=detection["idx"],
                    ref=ref_stamp,
                    x_ref_ccd_dvcs=x_ref_ccd_dvcs,
                    y_ref_ccd_dvcs=y_ref_ccd_dvcs,
                    tests=test_stamps,
                    offsets=offsets,
                    ref_id=reference_exposure.info.getVisitInfo().id,
                    test_ids=[exp.info.getVisitInfo().id for exp in test_exposures],
                )
            )
        return stamp_sets

    def choose_random_pupil_locations(self, stamp_set, n_positions):
        # Use danish to match pupil locations to image locations
        stamp = stamp_set["ref"]
        radius = self._donut_radius
        template_size = self._template_size
        template = np.zeros((template_size, template_size), dtype=bool)
        y, x = np.ogrid[
            -template_size // 2 : template_size // 2,
            -template_size // 2 : template_size // 2,
        ]
        r = np.hypot(x, y)
        template[r < radius] = True
        template[r < radius * 0.62] = False

        arr = stamp.stamp_im.image.array
        donut_mean = np.nanmean(arr[template])
        donut_iqr = np.ptp(np.nanquantile(arr[template], [0.75, 0.25]))
        threshold = donut_mean - 1.25 * donut_iqr

        expanded_template = np.zeros_like(template, dtype=bool)
        expanded_template[r < radius * 1.1] = True
        expanded_template[r < radius * 0.62 * 0.9] = False

        rng = np.random.Generator(np.random.PCG64(self.config.rng_seed))
        wy, wx = np.where((arr > threshold) & (expanded_template))
        x = x.ravel()[wx]
        y = y.ravel()[wy]
        w = rng.choice(len(x), size=n_positions, replace=False)
        x = x[w]
        y = y[w]
        # Parallel to DVCS
        return x, y

    def match_all_patches(
        self,
        stamp_sets,
    ):
        patch_table = None
        for stamp_set in stamp_sets:
            self.log.info("Matching patches in donut %d", stamp_set["donut_id"])
            fx, fy = self.choose_random_pupil_locations(
                stamp_set, self.config.n_pupil_positions
            )
            donut_patch_table = QTable()
            donut_patch_table["fx"] = fx
            donut_patch_table["fy"] = fy
            for iexp, test_stamp in enumerate(stamp_set["tests"]):
                self.log.info("  Matching in exposure %d", stamp_set["test_ids"][iexp])
                dfx, dfy = match_patches(
                    stamp_set["ref"].stamp_im.image.array,
                    test_stamp.stamp_im.image.array,
                    fx + self._template_size // 2,
                    fy + self._template_size // 2,
                    patch_size=30,
                    search_radius=10,
                )
                donut_patch_table[f"dfx_{iexp}"] = dfx
                donut_patch_table[f"dfy_{iexp}"] = dfy
            donut_patch_table["donut_id"] = stamp_set["donut_id"]
            if patch_table is None:
                patch_table = donut_patch_table
            else:
                patch_table = vstack([patch_table, donut_patch_table])
        return patch_table

    def remove_net_shift_and_rotation(
        self,
        patch_table,
    ):
        if patch_table is None:
            return
        self.log.info("Removing net shifts and rotations")
        idxs = [int(col[4:]) for col in patch_table.colnames if col.startswith("dfx_")]
        for idx in idxs:
            patch_table[f"dfx_{idx}_aligned"] = np.nan
            patch_table[f"dfy_{idx}_aligned"] = np.nan
        for donut_id in np.unique(patch_table["donut_id"]):
            wdonut = np.where(patch_table["donut_id"] == donut_id)[0]
            fx = patch_table["fx"][wdonut]
            fy = patch_table["fy"][wdonut]

            for idx in idxs:
                dfx = patch_table[f"dfx_{idx}"][wdonut]
                dfy = patch_table[f"dfy_{idx}"][wdonut]

                dfx_aligned, dfy_aligned = align_offsets(fx, fy, dfx, dfy)
                patch_table[f"dfx_{idx}_aligned"][wdonut] = dfx_aligned
                patch_table[f"dfy_{idx}_aligned"][wdonut] = dfy_aligned

    def update_display(
        self,
        exposure,
        donutCatalog,
    ):
        if self._display is None:
            raise RuntimeError("No display set")
        self._display.mtv(exposure)

        for idx, source in enumerate(donutCatalog):
            x, y = source["x_ref_ccd_dvcs"], source["y_ref_ccd_dvcs"]
            use = source["flux"] > self.config.min_flux
            use = use and source["inner_ratio"] < self.config.max_inner_ratio
            use = use and source["outer_ratio"] < self.config.max_outer_ratio
            color = "green" if use else "red"
            self._display.dot("o", x, y, size=self._donut_radius, ctype=color)
            self._display.dot(str(idx), x, y, ctype=color)

    def plot_initial(self, stamp_sets, patch_table):
        ndonut = len(stamp_sets)
        if ndonut == 0:
            nexp = 0
        else:
            nexp = len(stamp_sets[0]["tests"])

        figsize = (3 * self.config.max_exp_plot, 3 * self.config.max_donuts)
        fig = Figure(figsize=figsize, dpi=200)
        grispec_kw = dict(
            left=0.04, right=0.98, bottom=0.02, top=0.96, wspace=0.01, hspace=0.01
        )
        axs = fig.subplots(
            nrows=self.config.max_donuts,
            ncols=self.config.max_exp_plot,
            gridspec_kw=grispec_kw,
            squeeze=False,
        )
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idonut in range(self.config.max_donuts):
            for iexp in range(nexp):
                ax = axs[idonut][iexp]
                if idonut >= ndonut or iexp >= nexp:
                    fig.delaxes(ax)
                    continue
                stamp_set = stamp_sets[idonut]
                if idonut == 0:
                    ax.set_title(stamp_set["test_ids"][iexp])
                if iexp == 0:
                    label = (idonut, (int(stamp_set["x_ref_ccd_dvcs"]), int(stamp_set["y_ref_ccd_dvcs"])))
                    ax.set_ylabel(label)
                coords = patch_table[patch_table["donut_id"] == stamp_set["donut_id"]]
                fx = np.array(coords["fx"])
                fy = np.array(coords["fy"])
                dfx = np.array(coords[f"dfx_{iexp}_aligned"])
                dfy = np.array(coords[f"dfy_{iexp}_aligned"])

                ax = axs[idonut][iexp]
                vmax = np.quantile(stamp_set["ref"].stamp_im.image.array, 0.99)
                resid = (
                    stamp_set["ref"].stamp_im.image.array
                    - stamp_set["tests"][iexp].stamp_im.image.array
                )

                ext = resid.shape[0] // 2
                ax.imshow(
                    resid,
                    origin="lower",
                    cmap="bwr",
                    vmin=-vmax,
                    vmax=vmax,
                    extent=[-ext, ext, -ext, ext],
                )
                Q = ax.quiver(
                    fx,
                    fy,
                    dfx,
                    dfy,
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                )
                ax.quiverkey(Q, 0.12, 0.88, 3, "3 pixels")
        return fig

    def fit_displacements(
        self,
        patch_table,
    ):
        if patch_table is None:
            return
        donut_ids = np.unique(patch_table["donut_id"])
        ndonut = len(donut_ids)
        nexp = len(
            [
                col
                for col in patch_table.colnames
                if col.startswith("dfx_") and col.endswith("_aligned")
            ]
        )
        self.log.info(
            "Fitting displacements for %d donuts and %d exposures", ndonut, nexp
        )

        # Make room in table
        for iexp in range(nexp):
            patch_table[f"dfx_{iexp}_fit"] = np.nan
            patch_table[f"dfy_{iexp}_fit"] = np.nan
            patch_table[f"use_fit_{iexp}"] = False

        for donut_id in donut_ids:
            select = patch_table["donut_id"] == donut_id
            fx = patch_table["fx"][select]
            fy = patch_table["fy"][select]
            for iexp in range(nexp):
                dfx = patch_table[f"dfx_{iexp}_aligned"][select]
                dfy = patch_table[f"dfy_{iexp}_aligned"][select]
                dx_fit, dy_fit, use = fit_displacements(
                    fx, fy, dfx, dfy, self._donut_radius
                )
                patch_table[f"dfx_{iexp}_fit"][select] = dx_fit
                patch_table[f"dfy_{iexp}_fit"][select] = dy_fit
                patch_table[f"use_fit_{iexp}"][select] = use

    def plot_filtered(self, stamp_sets, patch_table):
        ndonut = len(stamp_sets)
        if ndonut == 0:
            nexp = 0
        else:
            nexp = len(stamp_sets[0]["tests"])

        figsize = (3 * self.config.max_exp_plot, 3 * self.config.max_donuts)
        fig = Figure(figsize=figsize, dpi=200)
        grispec_kw = dict(
            left=0.04, right=0.98, bottom=0.02, top=0.96, wspace=0.01, hspace=0.01
        )
        axs = fig.subplots(
            nrows=self.config.max_donuts,
            ncols=self.config.max_exp_plot,
            gridspec_kw=grispec_kw,
            squeeze=False,
        )
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idonut in range(self.config.max_donuts):
            for iexp in range(nexp):
                ax = axs[idonut][iexp]
                if idonut >= ndonut or iexp >= nexp:
                    fig.delaxes(ax)
                    continue
                stamp_set = stamp_sets[idonut]
                if idonut == 0:
                    ax.set_title(stamp_set["test_ids"][iexp])
                if iexp == 0:
                    label = (idonut, (int(stamp_set["x_ref_ccd_dvcs"]), int(stamp_set["y_ref_ccd_dvcs"])))
                    ax.set_ylabel(label)
                coords = patch_table[patch_table["donut_id"] == stamp_set["donut_id"]]
                fx = np.array(coords["fx"])
                fy = np.array(coords["fy"])
                dfx = np.array(coords[f"dfx_{iexp}_aligned"])
                dfy = np.array(coords[f"dfy_{iexp}_aligned"])
                dfx_fit = np.array(coords[f"dfx_{iexp}_fit"])
                dfy_fit = np.array(coords[f"dfy_{iexp}_fit"])
                wgood = np.array(coords[f"use_fit_{iexp}"], dtype=bool)

                ax = axs[idonut][iexp]
                vmax = np.quantile(stamp_set["ref"].stamp_im.image.array, 0.99)
                resid = (
                    stamp_set["ref"].stamp_im.image.array
                    - stamp_set["tests"][iexp].stamp_im.image.array
                )

                ext = resid.shape[0] // 2
                ax.imshow(
                    resid,
                    origin="lower",
                    cmap="bwr",
                    vmin=-vmax,
                    vmax=vmax,
                    extent=[-ext, ext, -ext, ext],
                )
                Q = ax.quiver(
                    fx[wgood],
                    fy[wgood],
                    dfx[wgood],
                    dfy[wgood],
                    color="g",
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                    width=0.002,
                )
                ax.quiver(
                    fx[~wgood],
                    fy[~wgood],
                    dfx[~wgood],
                    dfy[~wgood],
                    color="r",
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                    width=0.002,
                )
                ax.quiver(
                    fx[wgood],
                    fy[wgood],
                    dfx_fit[wgood],
                    dfy_fit[wgood],
                    color="k",
                    alpha=0.4,
                    width=0.004,
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                )
                ax.quiverkey(Q, 0.12, 0.88, 3, "3 pixels")
        return fig
