from pathlib import Path

import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct

from lsst.afw.table import SourceTable
from lsst.ip.isr import IsrTask
from lsst.meas.algorithms import SourceDetectionTask, SubtractBackgroundTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.base import IdGenerator, SingleFrameMeasurementTask
from lsst.summit.utils.quickLook import QuickLookIsrTask
from lsst.utils import getPackageDir

__all__ = [
    "HartmannSensitivityAnalysisConfig",
    "HartmannSensitivityAnalysisConnections",
    "HartmannSensitivityAnalysis",
]


class HartmannSensitivityAnalysisConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("group", "instrument", "detector")
):
    # exposures = ct.Input(
    #     doc="Input exposure to make measurements on",
    #     dimensions=("exposure", "detector", "instrument"),
    #     storageClass="Exposure",
    #     name="post_isr_image",
    #     multiple=True,
    # )
    exposures = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="raw",
        multiple=True,
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
        default=16,
    )
    installPsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Install a PSF model",
    )
    background = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Estimate background",
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask, doc="Measure sources"
    )

    def setDefaults(self):
        self.installPsf.fwhm = 5.0
        self.installPsf.width = 21

        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        self.detection.reEstimateBackground = False
        self.detection.doTempLocalBackground = False
        self.detection.minPixels = 40

        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_FPPosition",
            "base_SdssCentroid",
            "ext_shapeHSM_HsmSourceMoments",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
        ]
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"


class HartmannSensitivityAnalysis(
    pipeBase.PipelineTask,
):
    ConfigClass = HartmannSensitivityAnalysisConfig
    _DefaultName = "HartmannSensitivityAnalysis"

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.schema = SourceTable.makeMinimalSchema()
        self.makeSubtask("installPsf")
        self.makeSubtask("background")
        self.makeSubtask("detection", schema=self.schema)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

        self._display = display

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        exposures = [butlerQC.get(ref) for ref in inputRefs.exposures]
        results = self.run(exposures)

    def run_ISR(self, exposure):
        isrConfig = IsrTask.ConfigClass()
        packageDir = getPackageDir("summit_utils")
        isrConfig.load(Path(packageDir) / "config" / "quickLookIsr.py")
        quickLookIsrConfig = QuickLookIsrTask.ConfigClass()
        quickLookIsrConfig.doRepairCosmics = False
        quickLookIsrTask = QuickLookIsrTask(config=quickLookIsrConfig)
        return quickLookIsrTask.run(exposure, isrBaseConfig=isrConfig).outputExposure

    def run(
        self,
        exposures,
        skip_isr=False,
    ):
        config = self.config

        exposures.sort(key=lambda exp: exp.getInfo().getVisitInfo().id)
        if not skip_isr:
            exposures = [self.run_ISR(exp) for exp in exposures]
        ref_index = config.ref_index
        if ref_index < 0:
            ref_index = len(exposures) + ref_index
        reference_exposure = exposures[ref_index]
        test_exposures = [
            exp for i, exp in enumerate(exposures) if i != ref_index
        ]
        sources = self.detect(reference_exposure)


        # Run detection on reference to pick out suitable giant donuts.
        # Loop over others, passing to .run()
        # Make plots
        # Write outputs

    # Measure offsets between reference and test
        pass

    def detect(self, exposure):
        idGenerator = IdGenerator()
        sourceIdFactory = idGenerator.make_table_id_factory()
        table = SourceTable.make(self.schema, sourceIdFactory)
        # table.setMetadata(self.algMetadata)

        exposure = exposure.clone()
        binned = afwMath.binImage(exposure.getMaskedImage(), self.config.bin_size)
        exposure.setMaskedImage(binned)
        self.installPsf.run(exposure)
        self.background.run(exposure)
        sourceCat = self.detection.run(table=table, exposure=exposure, doSmooth=True).sources

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=idGenerator.catalog_id)

        if self._display is not None:
            self.update_display(exposure, sourceCat)
        return sourceCat

    def update_display(
        self,
        exposure,
        sourceCat,
    ):
        if self._display is None:
            raise RuntimeError("No display set")
        self._display.mtv(exposure)

        for idx, source in enumerate(sourceCat):
            x, y = source.getCentroid()
            sh = source.getShape()
            self._display.dot(sh, x, y)
            self._display.dot(str(idx), x, y)

    def plot(
        self, referenceExposure, testExposures, offsetTable,
    ):
        pass
