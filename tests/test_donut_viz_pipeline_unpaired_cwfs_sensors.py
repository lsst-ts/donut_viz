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
    AggregateDonutStampsUnpairedTask,
    PlotCwfsPairingTask,
    PlotCwfsPairingTaskConfig,
    PlotDonutCwfsTask,
    PlotDonutUnpairedCwfsTask,
    PlotDonutCwfsTaskConfig,
    PlotPsfZernTask,
    PlotPsfZernTaskConfig,
)
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
    @classmethod
    def setUpClass(cls):
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
        ]

        cls.butler = Butler(cls.test_repo_dir)
        cls.test_run_name = "test_run_1"
        registry = cls.butler.registry
        collections_list = list(registry.queryCollections())
        if cls.test_run_name in collections_list:
            clean_up_cmd = writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name)
            runProgram(clean_up_cmd)
        cls.extraDetectorNames = ["R00_SW0"]  # Only one detector pair used in tests
        cls.intraDetectorNames = ["R00_SW1"]

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.camera_name = "LSSTCam"
        test_pipeline = os.path.join(
            os.getenv("DONUT_VIZ_DIR"),
            "pipelines",
            "tests",
            "cwfsUnpairedPipeline.yaml",
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

    @classmethod
    def tearDownClass(cls):
        clean_up_cmd = writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name)
        runProgram(clean_up_cmd)

    def testAggregateZernikesAvg(self):
        average_dataset_list = list(
            self.butler.query_datasets(
                "aggregateZernikesAvg", collections=self.test_run_name
            )
        )
        self.assertEqual(len(average_dataset_list), 1)
        self.assertEqual(average_dataset_list[0].dataId["visit"], 4021123106000)
        agg_zern_avg = self.butler.get(average_dataset_list[0])
        self.assertEqual(len(agg_zern_avg), 2)
        self.assertCountEqual(
            agg_zern_avg["detector"], self.extraDetectorNames + self.intraDetectorNames
        )
        self.assertCountEqual(agg_zern_avg.meta.keys(), self.meta_keys)

    def testAggregateZernikesRaw(self):
        raw_dataset_list = list(
            self.butler.query_datasets(
                "aggregateZernikesRaw", collections=self.test_run_name
            )
        )
        self.assertEqual(len(raw_dataset_list), 1)
        self.assertEqual(raw_dataset_list[0].dataId["visit"], 4021123106000)
        agg_zern_raw = self.butler.get(raw_dataset_list[0])
        self.assertEqual(len(agg_zern_raw), 4)
        self.assertCountEqual(
            agg_zern_raw["detector"],
            [self.extraDetectorNames[0]] * 2 + [self.intraDetectorNames[0]] * 2,
        )
        self.assertCountEqual(
            agg_zern_raw.meta.keys(), self.meta_keys + ["estimatorInfo"]
        )

    def testAggregateDonuts(self):
        donut_table_list = list(
            self.butler.query_datasets(
                "aggregateDonutTable", collections=self.test_run_name
            )
        )
        self.assertEqual(len(donut_table_list), 1)
        self.assertEqual(donut_table_list[0].dataId["visit"], 4021123106000)
        agg_donut_table = self.butler.get(donut_table_list[0])
        self.assertEqual(len(agg_donut_table), 4)
        all_detectors = self.extraDetectorNames + self.intraDetectorNames
        self.assertCountEqual(
            agg_donut_table["detector"],
            sorted([det for det in all_detectors for _ in range(2)]),
        )
        self.assertCountEqual(agg_donut_table["focusZ"].value, [1.5] * 2 + [-1.5] * 2)
        self.assertCountEqual(agg_donut_table.meta.keys(), ["visitInfo"])
        donut_meta_keys = self.meta_keys + ["focusZ"]
        donut_meta_keys.remove("nollIndices")
        self.assertCountEqual(agg_donut_table.meta["visitInfo"].keys(), donut_meta_keys)

    def testAggregateDonutStamps(self):
        dataset_list = list(
            self.butler.query_datasets(
                "donutStampsUnpairedVisit", collections=self.test_run_name
            )
        )
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)
        unpaired_donuts = self.butler.get(dataset_list[0])
        self.assertEqual(len(unpaired_donuts), 2)

    def testAggregateAOSVisitTableRaw(self):
        raw_visit_table_list = list(
            self.butler.query_datasets(
                "aggregateAOSVisitTableRaw", collections=self.test_run_name
            )
        )
        self.assertEqual(len(raw_visit_table_list), 1)
        self.assertEqual(raw_visit_table_list[0].dataId["visit"], 4021123106000)
        raw_visit_table = self.butler.get(raw_visit_table_list[0])
        self.assertCountEqual(
            raw_visit_table.meta.keys(), self.meta_keys + ["estimatorInfo"]
        )
        raw_zern_table = self.butler.get(
            "aggregateZernikesRaw",
            dataId=raw_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(raw_zern_table), len(raw_visit_table))
        np.testing.assert_array_equal(
            raw_zern_table["zk_CCS"], raw_visit_table["zk_CCS"]
        )
        donut_table = self.butler.get(
            "aggregateDonutTable",
            dataId=raw_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(donut_table), len(raw_visit_table))
        self.assertCountEqual(
            donut_table["coord_ra"].value,
            raw_visit_table["coord_ra"].value,
        )

    def testAggregateAOSVisitTableAvg(self):
        avg_visit_table_list = list(
            self.butler.query_datasets(
                "aggregateAOSVisitTableAvg", collections=self.test_run_name
            )
        )
        self.assertEqual(len(avg_visit_table_list), 1)
        self.assertEqual(avg_visit_table_list[0].dataId["visit"], 4021123106000)
        avg_visit_table = self.butler.get(avg_visit_table_list[0])
        self.assertCountEqual(avg_visit_table.meta.keys(), self.meta_keys)
        avg_zern_table = self.butler.get(
            "aggregateZernikesAvg",
            dataId=avg_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        self.assertEqual(len(avg_zern_table), len(avg_visit_table))
        np.testing.assert_array_equal(
            avg_zern_table["zk_CCS"], avg_visit_table["zk_CCS"]
        )
        donut_table = self.butler.get(
            "aggregateDonutTable",
            dataId=avg_visit_table_list[0].dataId,
            collections=self.test_run_name,
        )
        np.testing.assert_array_equal(
            np.mean(donut_table["thx_CCS"][donut_table["detector"] == "R22_S11"]),
            avg_visit_table["thx_CCS"][avg_visit_table["detector"] == "R22_S11"],
        )

    def testPlotAOSTasks(self):
        # Test that plots exist in butler
        measured_dataset_list = list(
            self.butler.query_datasets(
                "measuredZernikePyramid", collections=self.test_run_name
            )
        )
        self.assertEqual(len(measured_dataset_list), 1)
        self.assertEqual(measured_dataset_list[0].dataId["visit"], 4021123106000)

        intrinsic_dataset_list = list(
            self.butler.query_datasets(
                "intrinsicZernikePyramid", collections=self.test_run_name
            )
        )

        self.assertEqual(len(intrinsic_dataset_list), 1)
        self.assertEqual(intrinsic_dataset_list[0].dataId["visit"], 4021123106000)

        residual_dataset_list = list(
            self.butler.query_datasets(
                "residualZernikePyramid", collections=self.test_run_name
            )
        )
        self.assertEqual(len(residual_dataset_list), 1)
        self.assertEqual(residual_dataset_list[0].dataId["visit"], 4021123106000)

    # def testPlotCwfsPairingTask(self):
    #     # Test that plots exist in butler
    #     dataset_list = list(
    #         self.butler.query_datasets("pairingPlot", collections=self.test_run_name)
    #     )
    #     self.assertEqual(len(dataset_list), 1)
    #     self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    # def testPlotCwfsPairingTaskRunMissingData(self):
    #     # Get only one detector
    #     dataset_list = list(
    #         self.butler.query_datasets("post_isr_image", collections=self.test_run_name)
    #     )

    #     images = {}
    #     # pick just one of two available detectors
    #     for dataset in dataset_list[:1]:
    #         det = dataset.dataId["detector"]
    #         images[det] = self.butler.get(dataset).image.array
    #     table = self.butler.get(
    #         self.butler.query_datasets(
    #             "aggregateAOSVisitTableRaw", collections=self.test_run_name
    #         )[0]
    #     )
    #     visit = dataset_list[0].dataId["exposure"]

    #     # Run the plotting task
    #     config = PlotCwfsPairingTaskConfig()
    #     config.doRubinTVUpload = False
    #     camera = LsstCam().getCamera()
    #     task = PlotCwfsPairingTask(config=config)
    #     taskOut = task.run(images, table, camera, visit)
    #     self.assertIsInstance(taskOut, matplotlib.figure.Figure)

    # def testPlotDonutFitsTask(self):
    #     # Test that plots exist in butler
    #     dataset_list = list(
    #         self.butler.query_datasets("donutFits", collections=self.test_run_name)
    #     )
    #     self.assertEqual(len(dataset_list), 1)
    #     self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotDonutUnpairedCwfsTask(self):
        # Test that plots exist in butler
        dataset_list = list(
            self.butler.query_datasets("donutPlot", collections=self.test_run_name)
        )
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotDonutUnpairedCwfsRunMissingData(self):
        # Aggregate only 3 of 4 detectors
        config = AggregateDonutStampsTaskConfig()
        task = AggregateDonutStampsTask(config=config)
        unpaired_datasets = self.butler.query_datasets(
            "donutStamps", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
        donut_stamps_unpaired = [
            self.butler.get(dataset) for dataset in unpaired_datasets
        ]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]

        donut_stamps_unpaired = [copy(donut_stamps_unpaired[0]) for i in range(4)]
        quality_tables = [copy(quality_tables[0]) for i in range(4)]

        # Remove all rows in first table
        quality_tables[0].remove_rows(np.arange(2))

        # Test that outputs are still created with only 3 detectors
        agg_donut_task = AggregateDonutStampsUnpairedTask()
        unpaired_stamps_miss = agg_donut_task.run(donut_stamps_unpaired, quality_tables)
        self.assertEqual(len(unpaired_stamps_miss), 3)

        # Run the plotting task
        inst = unpaired_datasets[0].dataId["instrument"]
        task = PlotDonutUnpairedCwfsTask()
        taskOut = task.run(unpaired_stamps_miss, inst)
        self.assertIsInstance(taskOut, matplotlib.figure.Figure)

    # def testPlotPsfZernTask(self):
    #     # Test that plots exist in butler
    #     psf_zern_dataset_list = list(
    #         self.butler.query_datasets(
    #             "psfFromZernPanel", collections=self.test_run_name
    #         )
    #     )
    #     self.assertEqual(len(psf_zern_dataset_list), 1)
    #     self.assertEqual(psf_zern_dataset_list[0].dataId["visit"], 4021123106000)

    def testAggDonutStampsRunMissingData(self):
        unpaired_datasets = self.butler.query_datasets(
            "donutStamps", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
        donut_stamps_unpaired = [
            self.butler.get(dataset) for dataset in unpaired_datasets
        ]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]
        donut_stamps_unpaired = [copy(donut_stamps_unpaired[0]) for i in range(4)]
        quality_tables = [copy(quality_tables[0]) for i in range(4)]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 2)
        # Remove all rows in first table
        quality_tables[0].remove_rows(np.arange(2))

        # Test that outputs are still created
        agg_donut_task = AggregateDonutStampsUnpairedTask()
        task_out = agg_donut_task.run(donut_stamps_unpaired, quality_tables)
        # Should still have output from 3 sets of stamps
        # while skipping the one with no quality rows
        self.assertEqual(len(task_out), 3)

    def testAggDonutStampsMetadata(self):
        unpaired_datasets = self.butler.query_datasets(
            "donutStamps", collections=self.test_run_name
        )
        agg_datasets = self.butler.query_datasets(
            "donutStampsUnpairedVisit", collections=self.test_run_name
        )
        agg_stamps = self.butler.get(agg_datasets[0])
        donut_stamps_unpaired = self.butler.get(unpaired_datasets[0])
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
        agg_meta_keys = list(agg_stamps.metadata.toDict().keys())
        for key in visit_keys:
            self.assertIn(key, agg_meta_keys)

        # Test that values are correctly set
        for key in visit_keys[:7]:
            self.assertEqual(
                agg_stamps.metadata[key], donut_stamps_unpaired.metadata[key]
            )

        # Separate out BORESIGHT_ALT_RAD and BORESIGHT_AZ_RAD
        # which should be nan
        for key in visit_keys[7:]:
            self.assertTrue(
                np.isnan(agg_stamps.metadata[key])
                and np.isnan(donut_stamps_unpaired.metadata[key])
            )

    def testAggDonutStampsSingleStamp(self):
        unpaired_datasets = self.butler.query_datasets(
            "donutStamps", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
        donut_stamps_unpaired = [
            self.butler.get(dataset) for dataset in unpaired_datasets
        ]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 2)
        # Leave only one DonutStamp in one of the DonutStamps objects
        quality_tables[0].remove_rows(1)
        donut_stamps_intra_new = DonutStamps(donut_stamps_unpaired[0][:1])
        for key in donut_stamps_unpaired[0].metadata.keys():
            donut_stamps_intra_new.metadata[key] = donut_stamps_unpaired[0].metadata[
                key
            ]
        donut_stamps_unpaired[0] = donut_stamps_intra_new

        # Test that outputs are still created
        agg_donut_task = AggregateDonutStampsUnpairedTask()
        task_out = agg_donut_task.run(donut_stamps_unpaired, quality_tables)
        self.assertEqual(len(task_out), 2)

    def testAggZernikeTablesRunMissingData(self):
        zernike_tables = self.butler.query_datasets(
            "zernikes", collections=self.test_run_name
        )
        zernike_table_list = [
            self.butler.get(zernike_table) for zernike_table in zernike_tables
        ]
        zernike_table_list = [copy(zernike_table_list[0]) for i in range(4)]
        # First check that original dataset is length more than 0
        self.assertEqual(len(zernike_table_list[0]), 3)
        # Remove all rows in first table
        zernike_table_list[0].remove_rows(np.arange(3))

        # Test that outputs are still created
        agg_zern_task = AggregateZernikeTablesTask(
            config=AggregateZernikeTablesTaskConfig()
        )
        raw_out, avg_out = agg_zern_task.run(zernike_table_list)
        self.assertEqual(len(raw_out), 6)
        self.assertEqual(len(avg_out), 3)

    def testAggDonutTablesRunMissingDate(self):
        donutTables = self.butler.query_datasets(
            "donutTable", collections=self.test_run_name
        )
        qualityTables = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )

        visitInfoDict = dict()
        for donutTableRef in donutTables:
            table = self.butler.get(donutTableRef)
            visit_id = table.meta["visit_info"]["visit_id"]
            if visit_id in visitInfoDict:
                continue
            visitInfoDict[visit_id] = convertDictToVisitInfo(table.meta["visit_info"])
        donutTables = {
            ref.dataId["detector"]: self.butler.get(ref) for ref in donutTables
        }
        qualityTables = {
            ref.dataId["detector"]: self.butler.get(ref) for ref in qualityTables
        }
        camera = self.butler.get(
            "camera", dataId={"instrument": "LSSTCam"}, collections="LSSTCam/calib"
        )

        # Remove all extra-focal donuts from one detector
        qualityTables[191].remove_rows(
            np.where(qualityTables[191]["DEFOCAL_TYPE"] == "extra")
        )

        task = AggregateDonutTablesCwfsTask(config=AggregateDonutTablesCwfsTaskConfig())
        agg_donut_table = task.run(camera, donutTables, qualityTables)
        self.assertEqual(len(agg_donut_table), 2)

    # def testPlotPsfZernTaskMissingData(self):
    #     # Test that if detectors have different numbers of zernikes
    #     # the plot still gets made.
    #     zernike_datasets = self.butler.query_datasets(
    #         "zernikes", collections=self.test_run_name
    #     )
    #     zernikes = [self.butler.get(dataset) for dataset in zernike_datasets]
    #     zernikes = [copy(zernikes[0]) for i in range(4)]
    #     zernikes_missing_data = copy(zernikes)
    #     zernikes_missing_data[0].remove_rows(np.arange(len(zernikes_missing_data[0])))
    #     task = PlotPsfZernTask(config=PlotPsfZernTaskConfig())
    #     for input_data in [zernikes, zernikes_missing_data]:
    #         try:
    #             task.run(zernikes)
    #         except Exception:
    #             self.fail(f"Unexpected exception raised with input {input_data}")
