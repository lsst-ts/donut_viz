import os
from copy import copy

import matplotlib
import numpy as np

from lsst.daf.butler import Butler
from lsst.donut.viz import (
    AggregateDonutStampsTask,
    AggregateDonutStampsTaskConfig,
    AggregateDonutTablesCwfsTask,
    AggregateDonutTablesCwfsTaskConfig,
    AggregateZernikeTablesTask,
    AggregateZernikeTablesTaskConfig,
    PlotCwfsPairingTask,
    PlotCwfsPairingTaskConfig,
    PlotDonutCwfsTask,
    PlotDonutCwfsTaskConfig,
    PlotDonutFitsTask,
    PlotPsfZernTask,
    PlotPsfZernTaskConfig,
)
from lsst.donut.viz.utilities import get_day_obs_seq_num_from_visitid
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task import DonutStamps
from lsst.ts.wep.utils import (
    convertDictToVisitInfo,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)
from lsst.utils.tests import TestCase


class TestDonutVizPipeline(TestCase):
    test_data_dir: str
    test_repo_dir: str
    test_run_name: str
    camera_name: str
    meta_keys: list[str]
    extra_detector_names: list[str]
    intra_detector_names: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        wep_module_dir = getModulePath()
        cls.test_data_dir = os.path.join(wep_module_dir, "tests", "testData")
        cls.test_repo_dir = os.path.join(cls.test_data_dir, "gen3TestRepo")
        cls.meta_keys = [
            "alt",
            "az",
            "dec",
            "mjd",
            "parallacticAngle",
            "ra",
            "rotAngle",
            "rotTelPos",
            "visit",
            "nollIndices",
            "band",
        ]

        cls.butler = Butler.from_config(cls.test_repo_dir)
        cls.test_run_name = "test_run_1"
        registry = cls.butler.registry
        collections_list = list(registry.queryCollections())
        if cls.test_run_name in collections_list:
            clean_up_cmd = writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name)
            runProgram(clean_up_cmd)
        cls.extra_detector_names = ["R00_SW0"]  # Only one detector pair used in tests
        cls.intra_detector_names = ["R00_SW1"]

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.camera_name = "LSSTCam"
        donut_viz_dir = os.getenv("DONUT_VIZ_DIR")
        if donut_viz_dir is None:
            raise RuntimeError("Environment variable DONUT_VIZ_DIR must be set for tests")
        test_pipeline = os.path.join(
            donut_viz_dir,
            "pipelines",
            "tests",
            "cwfsWcsCatalogPipeline.yaml",
        )

        pipe_cmd = writePipetaskCmd(
            cls.test_repo_dir,
            cls.test_run_name,
            instrument,
            collections,
            pipelineYaml=test_pipeline,
        )
        # Make sure we are using the right exposure+detector combinations
        pipe_cmd += ' -d "exposure IN (4021123106000) AND '
        pipe_cmd += 'detector IN (191, 192)"'
        runProgram(pipe_cmd)

    def setUp(self) -> None:
        self.butler = Butler.from_config(self.test_repo_dir)
        self.registry = self.butler.registry

    @classmethod
    def tearDownClass(cls) -> None:
        clean_up_cmd = writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name)
        runProgram(clean_up_cmd)

    def testAggregateZernikesAvg(self) -> None:
        average_dataset_list = list(
            self.butler.query_datasets("aggregateZernikesAvg", collections=self.test_run_name)
        )
        self.assertEqual(len(average_dataset_list), 1)
        self.assertEqual(average_dataset_list[0].dataId["visit"], 4021123106000)
        agg_zern_avg = self.butler.get(average_dataset_list[0])
        self.assertEqual(len(agg_zern_avg), 1)
        self.assertCountEqual(agg_zern_avg["detector"], self.extra_detector_names)
        self.assertCountEqual(agg_zern_avg.meta.keys(), self.meta_keys + ["estimatorInfo"])

    def testAggregateZernikesRaw(self) -> None:
        raw_dataset_list = list(
            self.butler.query_datasets("aggregateZernikesRaw", collections=self.test_run_name)
        )
        self.assertEqual(len(raw_dataset_list), 1)
        self.assertEqual(raw_dataset_list[0].dataId["visit"], 4021123106000)
        agg_zern_raw = self.butler.get(raw_dataset_list[0])
        self.assertEqual(len(agg_zern_raw), 2)
        self.assertCountEqual(
            agg_zern_raw["detector"],
            sorted([det for det in self.extra_detector_names for _ in range(2)]),
        )
        self.assertCountEqual(agg_zern_raw.meta.keys(), self.meta_keys + ["estimatorInfo"])

    def testAggregateDonuts(self) -> None:
        donut_table_list = list(
            self.butler.query_datasets("aggregateDonutTable", collections=self.test_run_name)
        )
        self.assertEqual(len(donut_table_list), 1)
        self.assertEqual(donut_table_list[0].dataId["visit"], 4021123106000)
        agg_donut_table = self.butler.get(donut_table_list[0])
        self.assertEqual(len(agg_donut_table), 4)
        all_detectors = self.extra_detector_names + self.intra_detector_names
        self.assertCountEqual(
            agg_donut_table["detector"],
            sorted([det for det in all_detectors for _ in range(2)]),
        )
        self.assertCountEqual(agg_donut_table["focusZ"].value, [1.5] * 2 + [-1.5] * 2)
        self.assertCountEqual(agg_donut_table.meta.keys(), ["visitInfo"])
        donut_meta_keys = self.meta_keys + ["focusZ"]
        donut_meta_keys.remove("nollIndices")
        donut_meta_keys.remove("band")  # this is not present in donutTable
        self.assertCountEqual(agg_donut_table.meta["visitInfo"].keys(), donut_meta_keys)

    def testAggregateDonutStamps(self) -> None:
        intra_dataset_list = list(
            self.butler.query_datasets("donutStampsIntraVisit", collections=self.test_run_name)
        )
        extra_dataset_list = list(
            self.butler.query_datasets("donutStampsExtraVisit", collections=self.test_run_name)
        )
        self.assertEqual(len(intra_dataset_list), 1)
        self.assertEqual(len(extra_dataset_list), 1)
        self.assertEqual(intra_dataset_list[0].dataId["visit"], 4021123106000)
        self.assertEqual(extra_dataset_list[0].dataId["visit"], 4021123106000)
        intra_donuts = self.butler.get(intra_dataset_list[0])
        extra_donuts = self.butler.get(extra_dataset_list[0])
        self.assertEqual(len(intra_donuts), 1)
        self.assertEqual(len(extra_donuts), 1)
        intra_meta = intra_donuts.metadata.toDict()
        extra_meta = extra_donuts.metadata.toDict()
        self.assertEqual(intra_meta["DET_NAME"], self.intra_detector_names[0])
        self.assertEqual(intra_meta["DFC_TYPE"], "intra")
        self.assertEqual(extra_meta["DET_NAME"], self.extra_detector_names[0])
        self.assertEqual(extra_meta["DFC_TYPE"], "extra")

    def testAggregateAOSVisitTableRaw(self) -> None:
        raw_visit_table_list = list(
            self.butler.query_datasets("aggregateAOSVisitTableRaw", collections=self.test_run_name)
        )
        self.assertEqual(len(raw_visit_table_list), 1)
        self.assertEqual(raw_visit_table_list[0].dataId["visit"], 4021123106000)
        raw_visit_table = self.butler.get(raw_visit_table_list[0])
        self.assertCountEqual(raw_visit_table.meta.keys(), self.meta_keys + ["estimatorInfo"])
        raw_zern_table = self.butler.get(
            "aggregateZernikesRaw",
            dataId=raw_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(raw_zern_table), len(raw_visit_table))
        np.testing.assert_array_equal(raw_zern_table["zk_CCS"], raw_visit_table["zk_CCS"])
        np.testing.assert_array_equal(raw_zern_table["zk_intrinsic_CCS"], raw_visit_table["zk_intrinsic_CCS"])
        np.testing.assert_array_equal(raw_zern_table["zk_deviation_CCS"], raw_visit_table["zk_deviation_CCS"])
        donut_table = self.butler.get(
            "aggregateDonutTable",
            dataId=raw_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(donut_table) / 2, len(raw_visit_table))
        self.assertCountEqual(
            donut_table["coord_ra"][donut_table["focusZ"].value == -1.5].value,
            raw_visit_table["coord_ra_intra"].value,
        )

    def testAggregateAOSVisitTableAvg(self) -> None:
        avg_visit_table_list = list(
            self.butler.query_datasets("aggregateAOSVisitTableAvg", collections=self.test_run_name)
        )
        self.assertEqual(len(avg_visit_table_list), 1)
        self.assertEqual(avg_visit_table_list[0].dataId["visit"], 4021123106000)
        avg_visit_table = self.butler.get(avg_visit_table_list[0])
        self.assertCountEqual(avg_visit_table.meta.keys(), self.meta_keys + ["estimatorInfo"])
        avg_zern_table = self.butler.get(
            "aggregateZernikesAvg",
            dataId=avg_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(avg_zern_table), len(avg_visit_table))
        np.testing.assert_array_equal(avg_zern_table["zk_CCS"], avg_visit_table["zk_CCS"])
        np.testing.assert_array_equal(avg_zern_table["zk_intrinsic_CCS"], avg_visit_table["zk_intrinsic_CCS"])
        np.testing.assert_array_equal(avg_zern_table["zk_deviation_CCS"], avg_visit_table["zk_deviation_CCS"])
        donut_table = self.butler.get(
            "aggregateDonutTable",
            dataId=avg_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        np.testing.assert_array_equal(
            np.mean(donut_table["thx_CCS"][donut_table["detector"] == "R22_S11"]),
            avg_visit_table["thx_CCS"][avg_visit_table["detector"] == "R22_S11"],
        )

    def testPlotAOSTasks(self) -> None:
        # Test that plots exist in butler
        measured_dataset_list = list(
            self.butler.query_datasets("measuredZernikePyramid", collections=self.test_run_name)
        )
        self.assertEqual(len(measured_dataset_list), 1)
        self.assertEqual(measured_dataset_list[0].dataId["visit"], 4021123106000)

        intrinsic_dataset_list = list(
            self.butler.query_datasets("intrinsicZernikePyramid", collections=self.test_run_name)
        )

        self.assertEqual(len(intrinsic_dataset_list), 1)
        self.assertEqual(intrinsic_dataset_list[0].dataId["visit"], 4021123106000)

        residual_dataset_list = list(
            self.butler.query_datasets("residualZernikePyramid", collections=self.test_run_name)
        )
        self.assertEqual(len(residual_dataset_list), 1)
        self.assertEqual(residual_dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotCwfsPairingTask(self) -> None:
        # Test that plots exist in butler
        dataset_list = list(self.butler.query_datasets("pairingPlot", collections=self.test_run_name))
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotCwfsPairingTaskRunMissingData(self) -> None:
        # Get only one detector
        dataset_list = list(self.butler.query_datasets("post_isr_image", collections=self.test_run_name))

        images = {}
        # pick just one of two available detectors
        for dataset in dataset_list[:1]:
            det = dataset.dataId["detector"]
            images[det] = self.butler.get(dataset).image.array
        table = self.butler.get(
            self.butler.query_datasets("aggregateAOSVisitTableRaw", collections=self.test_run_name)[0]
        )
        visit = dataset_list[0].dataId["exposure"]

        # Run the plotting task
        config = PlotCwfsPairingTaskConfig()
        config.doRubinTVUpload = False
        camera = LsstCam().getCamera()
        task = PlotCwfsPairingTask(config=config)
        taskOut = task.run(images, table, camera, int(visit))
        self.assertIsInstance(taskOut, matplotlib.figure.Figure)

    def testPlotDonutFitsTask(self) -> None:
        # Test that plots exist in butler
        dataset_list = list(self.butler.query_datasets("donutFits", collections=self.test_run_name))
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotDonutFitsTaskRunMissingMetadata(self) -> None:
        table_ref = list(
            self.butler.registry.queryDatasets(
                "aggregateAOSVisitTableRaw", collections=self.test_run_name
            ).expanded()
        )[0]
        aos_raw = self.butler.get(table_ref)
        stamps_extra = self.butler.get(
            self.butler.query_datasets("donutStampsExtraVisit", collections=self.test_run_name)[0]
        )
        stamps_intra = self.butler.get(
            self.butler.query_datasets("donutStampsIntraVisit", collections=self.test_run_name)[0]
        )
        camera = LsstCam().getCamera()

        self.task = PlotDonutFitsTask()
        day_obs, seq_num = get_day_obs_seq_num_from_visitid(4021123106000)
        record = table_ref.dataId.records["visit"]

        aos_raw.meta["estimatorInfo"] = {"fwhm": [1.5, 1.5]}
        # Test that median shifts are output to log
        with self.assertLogs(logger=self.task.log.logger, level="WARNING") as cm:
            self.task.run(
                aos_raw,
                stamps_intra,
                stamps_extra,
                camera,
                day_obs,
                seq_num,
                record,
            )
        records = cm.records  # always a list of LogRecord objects
        self.assertEqual(len(records), 2)

        for idx, rec in enumerate(records):
            expected = (
                "No model plot produced for R00, "
                f"donut index: {idx}. Required metadata for danish model not found in "
                "aggregateAOSVisitTableRaw."
            )
            # "rec.getMessage()" gives the real log message
            self.assertEqual(rec.levelname, "WARNING")
            self.assertEqual(rec.getMessage(), expected)

        # Test getModel function
        err_msg = str(
            "danish_meta must contain the following keys: "
            + "['fwhm', 'model_dx', 'model_dy', 'model_sky_level'], but only contains: {'fwhm'}"
        )
        with self.assertRaises(ValueError) as cm2:
            self.task.getModel(
                aos_raw[0]["zk_deviation_CCS"],
                aos_raw[0]["zk_intrinsic_CCS"],
                aos_raw.meta["nollIndices"],
                aos_raw.meta["estimatorInfo"],
                stamps_extra,
                stamps_intra,
            )
        self.assertEqual(str(cm2.exception), err_msg)

    def testPlotDonutCwfsTask(self) -> None:
        # Test that plots exist in butler
        dataset_list = list(self.butler.query_datasets("donutPlot", collections=self.test_run_name))
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotDonutCwfsRunMissingData(self) -> None:
        # Aggregate only 3 of 4 detectors
        intra_datasets = self.butler.query_datasets("donutStampsIntra", collections=self.test_run_name)
        extra_datasets = self.butler.query_datasets("donutStampsExtra", collections=self.test_run_name)
        quality_datasets = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)
        donut_stamps_intra = [self.butler.get(dataset) for dataset in intra_datasets]
        donut_stamps_extra = [self.butler.get(dataset) for dataset in extra_datasets]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]

        donut_stamps_intra = [copy(donut_stamps_intra[0]) for i in range(4)]
        donut_stamps_extra = [copy(donut_stamps_extra[0]) for i in range(4)]
        quality_tables = [copy(quality_tables[0]) for i in range(4)]

        # Remove all rows in first table
        quality_tables[0].remove_rows(np.arange(4))

        # Test that outputs are still created with only 3 detectors
        agg_donut_config = AggregateDonutStampsTaskConfig()
        agg_donut_task = AggregateDonutStampsTask(config=agg_donut_config)
        task_out = agg_donut_task.run(donut_stamps_intra, donut_stamps_extra, quality_tables)
        self.assertEqual(len(task_out.intra), 3)
        self.assertEqual(len(task_out.extra), 3)

        # Run the plotting task
        inst = str(intra_datasets[0].dataId["instrument"])
        plot_config = PlotDonutCwfsTaskConfig()
        plot_task = PlotDonutCwfsTask(config=plot_config)
        taskOut = plot_task.run(task_out.intra, task_out.extra, inst)
        self.assertIsInstance(taskOut, matplotlib.figure.Figure)

    def testPlotPsfZernTask(self) -> None:
        # Test that plots exist in butler
        psf_zern_dataset_list = list(
            self.butler.query_datasets("psfFromZernPanel", collections=self.test_run_name)
        )
        self.assertEqual(len(psf_zern_dataset_list), 1)
        self.assertEqual(psf_zern_dataset_list[0].dataId["visit"], 4021123106000)

    def testAggDonutStampsRunMissingData(self) -> None:
        intra_datasets = self.butler.query_datasets("donutStampsIntra", collections=self.test_run_name)
        extra_datasets = self.butler.query_datasets("donutStampsExtra", collections=self.test_run_name)
        quality_datasets = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)
        donut_stamps_intra = [self.butler.get(dataset) for dataset in intra_datasets]
        donut_stamps_extra = [self.butler.get(dataset) for dataset in extra_datasets]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]
        donut_stamps_intra = [copy(donut_stamps_intra[0]) for i in range(4)]
        donut_stamps_extra = [copy(donut_stamps_extra[0]) for i in range(4)]
        quality_tables = [copy(quality_tables[0]) for i in range(4)]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 4)
        # Remove all rows in first table
        quality_tables[0].remove_rows(np.arange(4))

        # Test that outputs are still created
        agg_donut_config = AggregateDonutStampsTaskConfig()
        agg_donut_task = AggregateDonutStampsTask(config=agg_donut_config)
        task_out = agg_donut_task.run(donut_stamps_intra, donut_stamps_extra, quality_tables)
        self.assertEqual(len(task_out), 2)

    def testAggDonutStampsMetadata(self) -> None:
        intra_datasets = self.butler.query_datasets("donutStampsIntra", collections=self.test_run_name)
        extra_datasets = self.butler.query_datasets("donutStampsExtra", collections=self.test_run_name)
        intra_agg_datasets = self.butler.query_datasets(
            "donutStampsIntraVisit", collections=self.test_run_name
        )
        intra_agg_stamps = self.butler.get(intra_agg_datasets[0])
        extra_agg_datasets = self.butler.query_datasets(
            "donutStampsExtraVisit", collections=self.test_run_name
        )
        extra_agg_stamps = self.butler.get(extra_agg_datasets[0])
        donut_stamps_intra = self.butler.get(intra_datasets[0])
        donut_stamps_extra = self.butler.get(extra_datasets[0])
        visit_keys = [
            "VISIT",
            "BORESIGHT_ROT_ANGLE_RAD",
            "BORESIGHT_PAR_ANGLE_RAD",
            "BORESIGHT_RA_RAD",
            "BORESIGHT_DEC_RAD",
            "MJD",
            "BANDPASS",
            "BORESIGHT_ALT_RAD",
            "BORESIGHT_AZ_RAD",
        ]

        # Assert that all visit_keys are present in aggregated metadata
        intra_meta_keys = list(intra_agg_stamps.metadata.toDict().keys())
        for key in visit_keys:
            self.assertIn(key, intra_meta_keys)
        extra_meta_keys = list(extra_agg_stamps.metadata.toDict().keys())
        for key in visit_keys:
            self.assertIn(key, extra_meta_keys)

        # Test that values are correctly set
        for key in visit_keys[:7]:
            self.assertEqual(intra_agg_stamps.metadata[key], donut_stamps_intra.metadata[key])
            self.assertEqual(extra_agg_stamps.metadata[key], donut_stamps_extra.metadata[key])
            self.assertEqual(intra_agg_stamps.metadata[key], extra_agg_stamps.metadata[key])
        # Separate out BORESIGHT_ALT_RAD and BORESIGHT_AZ_RAD
        # which should be nan
        for key in visit_keys[7:]:
            self.assertTrue(
                np.isnan(intra_agg_stamps.metadata[key]) and np.isnan(donut_stamps_intra.metadata[key])
            )
            self.assertTrue(
                np.isnan(extra_agg_stamps.metadata[key]) and np.isnan(donut_stamps_extra.metadata[key])
            )

    def testAggDonutStampsSingleStamp(self) -> None:
        intra_datasets = self.butler.query_datasets("donutStampsIntra", collections=self.test_run_name)
        extra_datasets = self.butler.query_datasets("donutStampsExtra", collections=self.test_run_name)
        quality_datasets = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)
        donut_stamps_intra = [self.butler.get(dataset) for dataset in intra_datasets]
        donut_stamps_extra = [self.butler.get(dataset) for dataset in extra_datasets]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 4)
        # Leave only one DonutStamp in one of the DonutStamps objects
        quality_tables[0].remove_rows(3)
        donut_stamps_intra_new = DonutStamps(donut_stamps_intra[0][:1])
        for key in donut_stamps_intra[0].metadata.keys():
            donut_stamps_intra_new.metadata[key] = donut_stamps_intra[0].metadata[key]
        donut_stamps_intra[0] = donut_stamps_intra_new

        # Test that outputs are still created
        agg_donut_config = AggregateDonutStampsTaskConfig()
        agg_donut_task = AggregateDonutStampsTask(config=agg_donut_config)
        task_out = agg_donut_task.run(donut_stamps_intra, donut_stamps_extra, quality_tables)
        self.assertEqual(len(task_out), 2)

    def testAggZernikeTablesRunMissingData(self) -> None:
        zernike_tables = self.butler.query_datasets("zernikes", collections=self.test_run_name)
        zernike_table_list = [self.butler.get(zernike_table) for zernike_table in zernike_tables]
        zernike_table_list = [copy(zernike_table_list[0]) for i in range(4)]
        # First check that original dataset is length more than 0
        self.assertEqual(len(zernike_table_list[0]), 3)
        # Remove all rows in first table
        zernike_table_list[0].remove_rows(np.arange(3))

        # Test that outputs are still created
        agg_zern_task = AggregateZernikeTablesTask(config=AggregateZernikeTablesTaskConfig())
        agg_zern_task_out = agg_zern_task.run(zernike_table_list)
        self.assertEqual(len(agg_zern_task_out.raw), 6)
        self.assertEqual(len(agg_zern_task_out.avg), 3)

    def testAggDonutTablesRunMissingDate(self) -> None:
        donutTables = self.butler.query_datasets("donutTable", collections=self.test_run_name)
        qualityTables = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)

        visitInfoDict = dict()
        for donutTableRef in donutTables:
            table = self.butler.get(donutTableRef)
            visit_id = table.meta["visit_info"]["visit_id"]
            if visit_id in visitInfoDict:
                continue
            visitInfoDict[visit_id] = convertDictToVisitInfo(table.meta["visit_info"])
        donutTablesDict = {ref.dataId["detector"]: self.butler.get(ref) for ref in donutTables}
        qualityTablesDict = {ref.dataId["detector"]: self.butler.get(ref) for ref in qualityTables}
        camera_object = self.butler.get(
            "camera", dataId={"instrument": "LSSTCam"}, collections="LSSTCam/calib"
        )

        # Remove all extra-focal donuts from one detector
        qualityTablesDict[191].remove_rows(np.where(qualityTablesDict[191]["DEFOCAL_TYPE"] == "extra"))

        task_aggr = AggregateDonutTablesCwfsTask(config=AggregateDonutTablesCwfsTaskConfig())
        aggTable = task_aggr.run(camera_object, donutTablesDict, qualityTablesDict)
        self.assertEqual(len(aggTable.aggregateDonutTable), 2)

    def testPlotPsfZernTaskMissingData(self) -> None:
        # Test that if detectors have different numbers of zernikes
        # the plot still gets made.
        zernike_datasets = self.butler.query_datasets("zernikes", collections=self.test_run_name)
        zernikes = [self.butler.get(dataset) for dataset in zernike_datasets]
        zernikes = [copy(zernikes[0]) for i in range(4)]
        zernikes_missing_data = copy(zernikes)
        zernikes_missing_data[0].remove_rows(np.arange(len(zernikes_missing_data[0])))
        task = PlotPsfZernTask(config=PlotPsfZernTaskConfig())
        for input_data in [zernikes, zernikes_missing_data]:
            try:
                task.run(zernikes)
            except Exception:
                self.fail(f"Unexpected exception raised with input {input_data}")
