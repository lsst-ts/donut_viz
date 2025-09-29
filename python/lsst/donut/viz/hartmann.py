import numpy as np
from collections import defaultdict

from astropy.table import QTable
from skimage.feature import peak_local_max
from scipy.signal import correlate

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct

from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import SubtractBackgroundTask, Stamp, Stamps


__all__ = [
    "HartmannSensitivityAnalysisConfig",
    "HartmannSensitivityAnalysisConnections",
    "HartmannSensitivityAnalysis",
]


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
            ymin = (csh[0]-rsh[0])//2 + dy
            ymax = ymin + rsh[0]
            xmin = (csh[1]-rsh[1])//2 + dx
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
        corr[best_ji[0], best_ji[1] + 1]
    )
    dy_sub = get_parabola_vertex(
        corr[best_ji[0] - 1, best_ji[1]],
        corr[best_ji[0], best_ji[1]],
        corr[best_ji[0] + 1, best_ji[1]]
    )

    best_offset = (best_offset[0] + dy_sub, best_offset[1] + dx_sub)

    return best_offset


def match_patches(img, ref, xs, ys, patch_size, search_radius):
    dx_out = []
    dy_out = []

    for x, y in zip(xs, ys):
        ref_crop = ref[
            y - patch_size // 2:y + patch_size // 2 + 1,
            x - patch_size // 2:x + patch_size // 2 + 1,
        ]
        img_crop = img[
            y - (patch_size + 2 * search_radius) // 2:y + (patch_size + 2 * search_radius) // 2 + 1,
            x - (patch_size + 2 * search_radius) // 2:x + (patch_size + 2 * search_radius) // 2 + 1,
        ]
        try:
            offset = get_offset(ref_crop, img_crop, search_radius)
        except:
            offset = np.nan, np.nan
        dy_out.append(offset[0])
        dx_out.append(offset[1])
    return dx_out, dy_out


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
        metadata["REF_X"].extend([stamp_set["ref_x"]] * nstamp)
        metadata["REF_Y"].extend([stamp_set["ref_y"]] * nstamp)
        metadata["OFFSET_X"].extend([0.0] + [off[1] for off in stamp_set["offsets"]])
        metadata["OFFSET_Y"].extend([0.0] + [off[0] for off in stamp_set["offsets"]])
    out = Stamps(all_stamps)
    for key, val in metadata.items():
        out.metadata[key] = np.array(val)

    return out


def stamps_to_stamp_sets(stamps):
    metadata = stamps.metadata
    donut_ids = np.array(metadata.getArray("DONUT_ID"))
    exp_ids = np.array(metadata.getArray("EXP_ID"))
    ref_ids = np.array(metadata.getArray("REF_ID"))

    stamp_sets = []
    for donut_id in np.unique(donut_ids):
        ref_donut_idx = np.where(
            (donut_ids == donut_id) & (exp_ids == ref_ids)
        )[0][0]
        ref = stamps[ref_donut_idx]
        test_idxs = np.where(
            (donut_ids == donut_id) & (exp_ids != ref_ids)
        )[0]
        tests = [stamps[i] for i in test_idxs]
        offset_x = [metadata.getArray("OFFSET_X")[i] for i in test_idxs]
        offset_y = [metadata.getArray("OFFSET_Y")[i] for i in test_idxs]
        offsets = list(zip(offset_y, offset_x))

        stamp_set = {}
        stamp_set["donut_id"] = donut_id
        stamp_set["ref"] = ref
        stamp_set["ref_x"] = metadata.getArray("REF_X")[ref_donut_idx]
        stamp_set["ref_y"] = metadata.getArray("REF_Y")[ref_donut_idx]
        stamp_set["tests"] = tests
        stamp_set["offsets"] = offsets
        stamp_set["ref_id"] = metadata.getArray("REF_ID")[ref_donut_idx]
        stamp_set["test_ids"] = [metadata.getArray("EXP_ID")[i] for i in test_idxs]

        stamp_sets.append(stamp_set)
        # metadata = stamps.metadata
        # donut_ids = np.unique(metadata["DONUT_ID"])
        # stamp_sets = []
        # for donut_id in donut_ids:
        #     ref_donut_idx = np.where(
        #         (metadata["DONUT_ID"] == donut_id) & (metadata["EXP_ID"] == metadata["REF_ID"])
        #     )[0][0]
        #     ref = stamps[ref_donut_idx]
        #     test_idxs = np.where(
        #         (metadata["DONUT_ID"] == donut_id) & (metadata["EXP_ID"] != metadata["REF_ID"])
        #     )[0]
        #     tests = [stamps[i] for i in test_idxs]
        #     offset_x = metadata["OFFSET_X"][test_idxs]
        #     offset_y = metadata["OFFSET_Y"][test_idxs]
        #     offsets = list(zip(offset_y, offset_x))

        #     stamp_set = {}
        #     stamp_set["donut_id"] = donut_id
        #     stamp_set["ref"] = ref
        #     stamp_set["ref_x"] = metadata["REF_X"][ref_donut_idx]
        #     stamp_set["ref_y"] = metadata["REF_Y"][ref_donut_idx]
        #     stamp_set["tests"] = tests
        #     stamp_set["offsets"] = offsets
        #     stamp_set["ref_id"] = metadata["REF_ID"][ref_donut_idx]
        #     stamp_set["exp_ids"] = [metadata["EXP_ID"][i] for i in test_idxs]

        #     stamp_sets.append(stamp_set)

    return stamp_sets

class HartmannSensitivityAnalysisConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("group", "instrument", "detector")
):
    exposures = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="raw",
        multiple=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    # hartmannDonuts = ct.Output(
    #     doc="Output Hartmann donuts",
    #     dimensions=("group", "instrument", "detector"),
    #     storageClass="DonutStamps",
    #     name="hartmann_donuts",
    # )
    # hartmannPlot = ct.Output(
    #     doc="Output Hartmann sensitivity analysis plot",
    #     dimensions=("group", "instrument", "detector"),
    #     storageClass="Plot",
    #     name="hartmann_sensitivity_plot",
    # )
    # hartmannTable = ct.Output(
    #     doc="Output Hartmann sensitivity analysis table",
    #     dimensions=("group", "instrument", "detector"),
    #     storageClass="AstropyQTable",
    #     name="hartmann_sensitivity_table",
    # )


class HartmannSensitivityAnalysisConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=HartmannSensitivityAnalysisConnections
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
        results = self.run(**inputs)

    def run(
        self,
        exposures,
        camera=None,
        skip_isr=False,
        **isr_kwargs
    ):
        config = self.config

        exposures.sort(key=lambda exp: exp.getInfo().getVisitInfo().id)
        if not skip_isr:
            self.log.info("Running ISR on %d exposures", len(exposures))
            exposures = [self.isr.run(exp, **isr_kwargs).exposure for exp in exposures]
        for exposure in exposures:
            self.log.info("Subtracting background")
            self.subtractBackground.run(exposure=exposure)

        ref_index = config.ref_index
        if ref_index < 0:
            ref_index = len(exposures) + ref_index
        reference_exposure = exposures[ref_index]
        ref_id = reference_exposure.info.getVisitInfo().id
        self.log.info("Using exposure %d as reference", ref_id)
        test_exposures = [
            exp for i, exp in enumerate(exposures) if i != ref_index
        ]
        self.log.info("Using %d test exposures", len(test_exposures))
        for exp in test_exposures:
            test_id = exp.info.getVisitInfo().id
            self.log.info("  test exposure: %d", test_id)

        detections = self.detect(reference_exposure)
        self.log.info("Detected %d donuts", len(detections))
        self.log.info("Selected %d donuts for analysis", np.sum(detections["use"]))

        if self._display is not None:
            self.update_display(reference_exposure, detections)

        self.log.info("Aligning stamps")
        stamp_sets = self.get_aligned_stamp_sets(
            reference_exposure,
            test_exposures,
            detections,
        )
        for idonut, stamp_set in enumerate(stamp_sets):
            self.log.info(
                "Processing stamp set %d", idonut
            )
            fx, fy = self.choose_random_pupil_locations(
                stamp_set["ref"],
                self.config.n_pupil_positions
            )
            stamp_set["coords"] = QTable()
            stamp_set["coords"]["fx"] = fx
            stamp_set["coords"]["fy"] = fy
            for iexp, test_stamp in enumerate(stamp_set["tests"]):
                dfx, dfy = match_patches(
                    stamp_set["ref"].stamp_im.image.array,
                    test_stamp.stamp_im.image.array,
                    fx + self._template_size // 2,
                    fy + self._template_size // 2,
                    patch_size=30,
                    search_radius=10,
                )
                stamp_set["coords"][f"dfx_{iexp}"] = dfx
                stamp_set["coords"][f"dfy_{iexp}"] = dfy

        # Make plots
        # Write outputs

        # Measure offsets between reference and test
        return pipeBase.Struct(
            detections=detections,
            stamp_sets=stamp_sets,
            stamps=stamp_sets_to_stamps(stamp_sets),
        )

    def detect(self, exposure):
        template = np.zeros(
            (self._binned_template_size, self._binned_template_size),
            dtype=float
        )
        y, x = np.ogrid[
            -self._binned_template_size // 2 : self._binned_template_size // 2,
            -self._binned_template_size // 2 : self._binned_template_size // 2
        ]
        r = np.hypot(x, y)
        binned_radius = self._donut_radius / self.config.bin_size
        template[r < binned_radius] = 1.0
        template[r < binned_radius*0.62] = 0.0
        inner_hole = np.zeros_like(template)
        inner_hole[r < binned_radius*0.55] = 1.0
        outer_annulus = np.zeros_like(template)
        outer_annulus[(r >= binned_radius*1.05) & (r < binned_radius*1.15)] = 1.0

        exp = exposure.clone()
        mi = exp.getMaskedImage()
        binned = afwMath.binImage(mi, self.config.bin_size)
        exp.setMaskedImage(binned)
        arr = exp.image.array
        mask = exp.mask.array

        # Histogram equalize since we care more about connected points above threshold
        # than actual flux values.
        cdf = np.nanquantile(arr, np.linspace(0, 1, 256))
        heq = np.digitize(arr, cdf)
        det = correlate(heq, template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(2 * 0.8 * binned_radius),
            exclude_border=int(binned_radius*1.15)
        )

        table = QTable()
        table["idx"] = np.arange(len(peaks), dtype=np.int32)
        table["ref_x"] = (peaks[:, 1] * self.config.bin_size).astype(np.int32)
        table["ref_y"] = (peaks[:, 0] * self.config.bin_size).astype(np.int32)
        fluxes = []
        inner_fluxes = []
        outer_fluxes = []
        for peak in peaks:
            xmin = peak[1] - self._binned_template_size // 2
            xmax = peak[1] + self._binned_template_size // 2 + 1
            ymin = peak[0] - self._binned_template_size // 2
            ymax = peak[0] + self._binned_template_size // 2 + 1
            stamp = arr[ymin:ymax, xmin:xmax]
            bad_mask_planes = ['BAD', 'CR', 'INTRP', 'SAT', 'SUSPECT', 'NO_DATA']
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

            # Guess that a reasonable background estimate is the median of the inner
            # and outer pixels
            bkg = np.nanmedian(stamp*(np.maximum(inner_hole, outer_annulus)))
            stamp = stamp - bkg
            fluxes.append(np.nansum(stamp*template))
            inner_fluxes.append(np.nansum(stamp*inner_hole))
            outer_fluxes.append(np.nansum(stamp*outer_annulus))
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
        idxs = subtable["idx"][:self.config.max_donuts]
        table["use"][~np.isin(table["idx"], idxs)] = False

        return table

    def get_aligned_stamp_sets(
        self,
        reference_exposure,
        test_exposures,
        detections,
    ):
        stamp_size = self._template_size
        stamp_sets = []
        for detection in detections:
            if not detection["use"]:
                continue
            self.log.info(
                "Aligning detection %d at (x,y)=(%d,%d)",
                detection["idx"],
                detection["ref_x"],
                detection["ref_y"]
            )
            ref_x, ref_y = detection[["ref_x", "ref_y"]]

            # Extract stamp from reference
            xmin = ref_x - stamp_size // 2
            xmax = ref_x + stamp_size // 2 + 1
            ymin = ref_y - stamp_size // 2
            ymax = ref_y + stamp_size // 2 + 1
            box = Box2I(Point2I(xmin, ymin), Extent2I(xmax-xmin, ymax-ymin))
            ref_stamp = Stamp(reference_exposure.maskedImage[box])
            ref_stamp_arr = ref_stamp.stamp_im.image.array

            offsets = []
            test_stamps = []
            for test_exposure in test_exposures:
                test_stamp_arr = test_exposure.image.array[ymin:ymax, xmin:xmax]
                offset = get_offset(ref_stamp_arr, test_stamp_arr, search_radius=30)
                offsets.append(offset)
                test_box = Box2I(
                    Point2I(
                        xmin + int(round(offset[1])),
                        ymin + int(round(offset[0]))
                    ),
                    Extent2I(xmax - xmin, ymax - ymin)
                )
                test_stamps.append(Stamp(test_exposure.maskedImage[test_box]))
            test_stamps = Stamps(test_stamps)
            stamp_sets.append(
                dict(
                    donut_id=detection["idx"],
                    ref=ref_stamp,
                    ref_x=ref_x,
                    ref_y=ref_y,
                    tests=test_stamps,
                    offsets=offsets,
                    ref_id=reference_exposure.info.getVisitInfo().id,
                    test_ids=[exp.info.getVisitInfo().id for exp in test_exposures],
                )
            )
        return stamp_sets

    def choose_random_pupil_locations(
        self,
        stamp,
        n_positions
    ):
        radius = self._donut_radius
        template_size = self._template_size
        template = np.zeros((template_size, template_size), dtype=bool)
        y, x = np.ogrid[
            -template_size // 2 : template_size // 2,
            -template_size // 2 : template_size // 2
        ]
        r = np.hypot(x, y)
        template[r < radius] = True
        template[r < radius*0.62] = False

        arr = stamp.stamp_im.image.array
        donut_mean = np.nanmean(arr[template])
        donut_iqr = np.ptp(np.nanquantile(arr[template], [0.75, 0.25]))
        threshold = donut_mean - 1.25 * donut_iqr

        expanded_template = np.zeros_like(template, dtype=bool)
        expanded_template[r < radius*1.1] = True
        expanded_template[r < radius*0.62*0.9] = False

        rng = np.random.Generator(np.random.PCG64(self.config.rng_seed))
        wx, wy = np.where(
            (arr > threshold) & (expanded_template)
        )
        x = x.ravel()[wx]
        y = y.ravel()[wy]
        w = rng.choice(len(x), size=n_positions, replace=False)
        x = x[w]
        y = y[w]
        return x, y

    def update_display(
        self,
        exposure,
        donutCatalog,
    ):
        if self._display is None:
            raise RuntimeError("No display set")
        self._display.mtv(exposure)

        for idx, source in enumerate(donutCatalog):
            x, y = source["ref_x"], source["ref_y"]
            use = source["flux"] > self.config.min_flux
            use = use and source["inner_ratio"] < self.config.max_inner_ratio
            use = use and source["outer_ratio"] < self.config.max_outer_ratio
            color = "green" if use else "red"
            self._display.dot(
                "o", x, y,
                size=self._donut_radius,
                ctype=color
            )
            self._display.dot(str(idx), x, y, ctype=color)

    def plot(
        self, referenceExposure, testExposures, offsetTable,
    ):
        pass
