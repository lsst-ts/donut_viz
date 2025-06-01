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
    PlotDonutCwfsTask,
    PlotDonutCwfsTaskConfig,
    PlotPsfZernTask,
    PlotPsfZernTaskConfig,
)
from lsst.ts.wep.task import DonutStamps
from lsst.ts.wep.task.generateDonutCatalogUtils import convertDictToVisitInfo
from lsst.ts.wep.utils import (
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
        self.assertEqual(len(agg_zern_avg), 1)
        self.assertCountEqual(agg_zern_avg["detector"], self.extraDetectorNames)
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
        self.assertEqual(len(agg_zern_raw), 2)
        self.assertCountEqual(
            agg_zern_raw["detector"],
            sorted([det for det in self.extraDetectorNames for _ in range(2)]),
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
        intra_dataset_list = list(
            self.butler.query_datasets(
                "donutStampsIntraVisit", collections=self.test_run_name
            )
        )
        extra_dataset_list = list(
            self.butler.query_datasets(
                "donutStampsExtraVisit", collections=self.test_run_name
            )
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
        self.assertEqual(intra_meta["DET_NAME"], self.intraDetectorNames[0])
        self.assertEqual(intra_meta["DFC_TYPE"], "intra")
        self.assertEqual(extra_meta["DET_NAME"], self.extraDetectorNames[0])
        self.assertEqual(extra_meta["DFC_TYPE"], "extra")

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
        self.assertEqual(len(donut_table) / 2, len(raw_visit_table))
        self.assertCountEqual(
            donut_table["coord_ra"][donut_table["focusZ"].value == -1.5].value,
            raw_visit_table["coord_ra_intra"].value,
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

    def testPlotDonutCwfsTask(self):
        # Test that plots exist in butler
        dataset_list = list(
            self.butler.query_datasets("donutPlot", collections=self.test_run_name)
        )
        self.assertEqual(len(dataset_list), 1)
        self.assertEqual(dataset_list[0].dataId["visit"], 4021123106000)

    def testPlotDonutCwfsRunMissingData(self):
        # Aggregate only 3 of 4 detectors
        config = AggregateDonutStampsTaskConfig()
        task = AggregateDonutStampsTask(config=config)
        intra_datasets = self.butler.query_datasets(
            "donutStampsIntra", collections=self.test_run_name
        )
        extra_datasets = self.butler.query_datasets(
            "donutStampsExtra", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
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
        intra_stamps_miss, extra_stamps_miss = agg_donut_task.run(
            donut_stamps_intra, donut_stamps_extra, quality_tables
        )
        self.assertEqual(len(intra_stamps_miss), 3)
        self.assertEqual(len(extra_stamps_miss), 3)

        # Run the plotting task
        inst = intra_datasets[0].dataId["instrument"]
        config = PlotDonutCwfsTaskConfig()
        task = PlotDonutCwfsTask(config=config)
        taskOut = task.run(intra_stamps_miss, extra_stamps_miss, inst)
        self.assertIsInstance(taskOut, matplotlib.figure.Figure)

    def testPlotPsfZernTask(self):
        # Test that plots exist in butler
        psf_zern_dataset_list = list(
            self.butler.query_datasets(
                "psfFromZernPanel", collections=self.test_run_name
            )
        )
        self.assertEqual(len(psf_zern_dataset_list), 1)
        self.assertEqual(psf_zern_dataset_list[0].dataId["visit"], 4021123106000)

    def testAggDonutStampsRunMissingData(self):
        intra_datasets = self.butler.query_datasets(
            "donutStampsIntra", collections=self.test_run_name
        )
        extra_datasets = self.butler.query_datasets(
            "donutStampsExtra", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
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
        task_out = agg_donut_task.run(
            donut_stamps_intra, donut_stamps_extra, quality_tables
        )
        self.assertEqual(len(task_out), 2)

    def testAggDonutStampsMetadata(self):
        intra_datasets = self.butler.query_datasets(
            "donutStampsIntra", collections=self.test_run_name
        )
        extra_datasets = self.butler.query_datasets(
            "donutStampsExtra", collections=self.test_run_name
        )
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
            self.assertEqual(
                intra_agg_stamps.metadata[key], donut_stamps_intra.metadata[key]
            )
            self.assertEqual(
                extra_agg_stamps.metadata[key], donut_stamps_extra.metadata[key]
            )
            self.assertEqual(
                intra_agg_stamps.metadata[key], extra_agg_stamps.metadata[key]
            )
        # Separate out BORESIGHT_ALT_RAD and BORESIGHT_AZ_RAD
        # which should be nan
        for key in visit_keys[7:]:
            self.assertTrue(
                np.isnan(intra_agg_stamps.metadata[key])
                and np.isnan(donut_stamps_intra.metadata[key])
            )
            self.assertTrue(
                np.isnan(extra_agg_stamps.metadata[key])
                and np.isnan(donut_stamps_extra.metadata[key])
            )

    def testAggDonutStampsSingleStamp(self):
        intra_datasets = self.butler.query_datasets(
            "donutStampsIntra", collections=self.test_run_name
        )
        extra_datasets = self.butler.query_datasets(
            "donutStampsExtra", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets(
            "donutQualityTable", collections=self.test_run_name
        )
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
        task_out = agg_donut_task.run(
            donut_stamps_intra, donut_stamps_extra, quality_tables
        )
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

    def testPlotPsfZernTaskMissingData(self):
        # Test that if detectors have different numbers of zernikes
        # the plot still gets made.
        zernike_datasets = self.butler.query_datasets(
            "zernikes", collections=self.test_run_name
        )
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
