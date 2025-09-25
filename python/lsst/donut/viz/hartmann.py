import numpy as np

from astropy.table import QTable
from skimage.feature import peak_local_max
from scipy.signal import correlate

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct

from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import SubtractBackgroundTask


__all__ = [
    "HartmannSensitivityAnalysisConfig",
    "HartmannSensitivityAnalysisConnections",
    "HartmannSensitivityAnalysis",
]


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
    do_plot = pexConfig.Field[bool](
        doc="Whether to make plots",
        default=True,
    )
    bin_size = pexConfig.Field[int](
        doc="Bin size for running initial detection",
        default=8,
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
        self._donut_diam = int(680 / 8 * self._defocus / self.config.bin_size)
        self._donut_radius = self._donut_diam / 2

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
            exposures = [self.isr.run(exp, **isr_kwargs).exposure for exp in exposures]
        for exposure in exposures:
            self.subtractBackground.run(exposure=exposure)

        ref_index = config.ref_index
        if ref_index < 0:
            ref_index = len(exposures) + ref_index
        reference_exposure = exposures[ref_index]
        test_exposures = [
            exp for i, exp in enumerate(exposures) if i != ref_index
        ]

        detections = self.detect(reference_exposure)

        if self._display is not None:
            self.update_display(reference_exposure, detections)

        # Loop over test exposures and measure offsets
        # Make plots
        # Write outputs

        # Measure offsets between reference and test
        return pipeBase.Struct(
            detections=detections,
        )

    def detect(self, exposure):
        diam = self._donut_diam
        radius = self._donut_radius
        template = np.zeros((diam, diam), dtype=float)
        y, x = np.ogrid[-diam // 2 : diam // 2, -diam // 2 : diam // 2]
        r = np.hypot(x, y)
        template[r < radius] = 1.0
        template[r < radius*0.62] = 0.0
        hole = np.zeros_like(template)
        hole[r < radius*0.62] = 1.0

        exp = exposure.clone()
        mi = exp.getMaskedImage()
        binned = afwMath.binImage(mi, self.config.bin_size)
        exp.setMaskedImage(binned)
        arr = exp.image.array

        # Histogram equalize since we care more about connected points above threshold
        # than actual flux values.
        cdf = np.nanquantile(arr, np.linspace(0, 1, 256))
        heq = np.digitize(arr, cdf)
        det = correlate(heq, template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(0.8 * diam),
            exclude_border=int(radius)
        )

        table = QTable()
        table["idx"] = np.arange(len(peaks), dtype=np.int32)
        table["centroid_x"] = (peaks[:, 1] * self.config.bin_size).astype(np.int32)
        table["centroid_y"] = (peaks[:, 0] * self.config.bin_size).astype(np.int32)
        fluxes = []
        hole_fluxes = []
        for peak in peaks:
            stamp = arr[
                peak[0] - diam // 2 : peak[0] + diam // 2 + 1,
                peak[1] - diam // 2 : peak[1] + diam // 2 + 1,
            ]
            fluxes.append(np.nansum(stamp*template))
            hole_fluxes.append(np.nansum(stamp*hole))
        table["flux"] = np.array(fluxes, dtype=np.float32)
        table["hole_flux"] = np.array(hole_fluxes, dtype=np.float32)
        table["ratio"] = table["hole_flux"] / table["flux"]
        return table

    def update_display(
        self,
        exposure,
        donutCatalog,
    ):
        if self._display is None:
            raise RuntimeError("No display set")
        self._display.mtv(exposure)

        for idx, source in enumerate(donutCatalog):
            x, y = source["centroid_x"], source["centroid_y"]
            if source["flux"] > 3e6 and abs(source["ratio"]) < 0.1:
                color = "green"
            else:
                color="red"
            self._display.dot(
                "o", x, y,
                size=self._donut_radius*self.config.bin_size,
                ctype=color
            )
            self._display.dot(str(idx), x, y, ctype=color)

    def plot(
        self, referenceExposure, testExposures, offsetTable,
    ):
        pass
