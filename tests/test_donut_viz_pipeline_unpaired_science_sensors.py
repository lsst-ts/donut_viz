import os
from copy import copy

import numpy as np

from lsst.daf.butler import Butler
from lsst.donut.viz import (
    AggregateDonutStampsUnpairedTask,
    AggregateDonutTablesUnpairedTask,
    AggregateZernikeTablesTask,
    AggregateZernikeTablesTaskConfig,
)
from lsst.ts.wep.task import DonutStamps
from lsst.ts.wep.utils import (
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
            "scienceUnpairedPipeline.yaml",
        )

        pipe_cmd = writePipetaskCmd(
            cls.test_repo_dir,
            cls.test_run_name,
            instrument,
            collections,
            pipelineYaml=test_pipeline,
        )
        # Make sure we are using the right exposure+detector combinations
        pipe_cmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipe_cmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
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
        self.assertEqual(len(average_dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in average_dataset_list], [4021123106001, 4021123106002]
        )
        for dataset in average_dataset_list:
            agg_zern_avg = self.butler.get(dataset)
            self.assertEqual(len(agg_zern_avg), 2)
            self.assertCountEqual(agg_zern_avg["detector"], ["R22_S10", "R22_S11"])
            self.assertCountEqual(agg_zern_avg.meta.keys(), self.meta_keys + ["estimatorInfo"])

    def testAggregateZernikesRaw(self) -> None:
        raw_dataset_list = list(
            self.butler.query_datasets("aggregateZernikesRaw", collections=self.test_run_name)
        )
        self.assertEqual(len(raw_dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in raw_dataset_list], [4021123106001, 4021123106002]
        )
        for dataset in raw_dataset_list:
            agg_zern_raw = self.butler.get(dataset)
            self.assertEqual(len(agg_zern_raw), 6)
            self.assertCountEqual(
                agg_zern_raw["detector"],
                ["R22_S10"] * 3 + ["R22_S11"] * 3,
            )
            self.assertCountEqual(agg_zern_raw.meta.keys(), self.meta_keys + ["estimatorInfo"])

    def testAggregateDonuts(self) -> None:
        donut_table_list = list(
            self.butler.query_datasets("aggregateDonutTable", collections=self.test_run_name)
        )
        self.assertEqual(len(donut_table_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in donut_table_list], [4021123106001, 4021123106002]
        )
        for dataset in donut_table_list:
            agg_donut_table = self.butler.get(dataset)
            self.assertEqual(len(agg_donut_table), 6)
            self.assertCountEqual(agg_donut_table["detector"], ["R22_S10"] * 3 + ["R22_S11"] * 3)
            self.assertCountEqual(
                agg_donut_table["focusZ"].value,
                [1.5] * 6 if dataset.dataId["visit"] == 4021123106001 else [-1.5] * 6,
            )
            self.assertCountEqual(agg_donut_table.meta.keys(), ["visitInfo"])
            donut_meta_keys = self.meta_keys + ["focusZ"]
            donut_meta_keys.remove("nollIndices")
            donut_meta_keys.remove("band")  # not in donutTable metadata
            self.assertCountEqual(agg_donut_table.meta["visitInfo"].keys(), donut_meta_keys)

    def testAggregateDonutStamps(self) -> None:
        dataset_list = list(
            self.butler.query_datasets("donutStampsUnpairedVisit", collections=self.test_run_name)
        )
        self.assertEqual(len(dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in dataset_list], [4021123106001, 4021123106002]
        )
        for dataset in dataset_list:
            unpaired_donuts = self.butler.get(dataset)
            self.assertEqual(len(unpaired_donuts), 2)

    def testAggregateAOSVisitTableRaw(self) -> None:
        raw_visit_table_list = list(
            self.butler.query_datasets("aggregateAOSVisitTableRaw", collections=self.test_run_name)
        )
        self.assertEqual(len(raw_visit_table_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in raw_visit_table_list], [4021123106001, 4021123106002]
        )
        for dataset in raw_visit_table_list:
            raw_visit_table = self.butler.get(dataset)
            self.assertCountEqual(raw_visit_table.meta.keys(), self.meta_keys + ["estimatorInfo"])
            raw_zern_table = self.butler.get(
                "aggregateZernikesRaw",
                dataId=dataset.dataId,
                collections=self.test_run_name,
            )
            self.assertEqual(len(raw_zern_table), len(raw_visit_table))
            np.testing.assert_array_equal(raw_zern_table["zk_CCS"], raw_visit_table["zk_CCS"])
            np.testing.assert_array_equal(
                raw_zern_table["zk_intrinsic_CCS"], raw_visit_table["zk_intrinsic_CCS"]
            )
            np.testing.assert_array_equal(
                raw_zern_table["zk_deviation_CCS"], raw_visit_table["zk_deviation_CCS"]
            )
            donut_table = self.butler.get(
                "aggregateDonutTable",
                dataId=dataset.dataId,
                collections=self.test_run_name,
            )
            self.assertEqual(len(donut_table), len(raw_visit_table))
            self.assertCountEqual(
                donut_table["coord_ra"].value,
                raw_visit_table["coord_ra"].value,
            )

    def testAggregateAOSVisitTableAvg(self) -> None:
        avg_visit_table_list = list(
            self.butler.query_datasets("aggregateAOSVisitTableAvg", collections=self.test_run_name)
        )
        self.assertEqual(len(avg_visit_table_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in avg_visit_table_list], [4021123106001, 4021123106002]
        )
        for dataset in avg_visit_table_list:
            avg_visit_table = self.butler.get(dataset)
            self.assertCountEqual(avg_visit_table.meta.keys(), self.meta_keys + ["estimatorInfo"])
            avg_zern_table = self.butler.get(
                "aggregateZernikesAvg",
                dataId=dataset.dataId,
                collections=self.test_run_name,
            )
            self.assertEqual(len(avg_zern_table), len(avg_visit_table))
            np.testing.assert_array_equal(avg_zern_table["zk_CCS"], avg_visit_table["zk_CCS"])
            np.testing.assert_array_equal(
                avg_zern_table["zk_intrinsic_CCS"], avg_visit_table["zk_intrinsic_CCS"]
            )
            np.testing.assert_array_equal(
                avg_zern_table["zk_deviation_CCS"], avg_visit_table["zk_deviation_CCS"]
            )
            donut_table = self.butler.get(
                "aggregateDonutTable",
                dataId=dataset.dataId,
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
        self.assertEqual(len(measured_dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in measured_dataset_list], [4021123106001, 4021123106002]
        )

        intrinsic_dataset_list = list(
            self.butler.query_datasets("intrinsicZernikePyramid", collections=self.test_run_name)
        )

        self.assertEqual(len(intrinsic_dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in intrinsic_dataset_list], [4021123106001, 4021123106002]
        )

        residual_dataset_list = list(
            self.butler.query_datasets("residualZernikePyramid", collections=self.test_run_name)
        )
        self.assertEqual(len(residual_dataset_list), 2)
        self.assertCountEqual(
            [dataset.dataId["visit"] for dataset in residual_dataset_list], [4021123106001, 4021123106002]
        )

    def testAggDonutStampsRunMissingData(self) -> None:
        unpaired_datasets = self.butler.query_datasets(
            "donutStampsScienceSensor", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)
        donut_stamps_unpaired = [self.butler.get(dataset) for dataset in unpaired_datasets]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]
        donut_stamps_unpaired = [copy(donut_stamps_unpaired[0]) for i in range(4)]
        quality_tables = [copy(quality_tables[0]) for i in range(4)]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 3)
        # Remove all rows in first table
        quality_tables[0].remove_rows(np.arange(3))

        # Test that outputs are still created
        agg_donut_task = AggregateDonutStampsUnpairedTask()
        task_out = agg_donut_task.run(donut_stamps_unpaired, quality_tables)
        # Should still have output from 3 sets of stamps
        # while skipping the one with no quality rows
        self.assertEqual(len(task_out.stamps), 3)

    def testAggDonutStampsMetadata(self) -> None:
        unpaired_datasets = self.butler.query_datasets(
            "donutStampsScienceSensor", collections=self.test_run_name
        )
        agg_datasets = self.butler.query_datasets("donutStampsUnpairedVisit", collections=self.test_run_name)
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
            self.assertEqual(agg_stamps.metadata[key], donut_stamps_unpaired.metadata[key])

        # Separate out BORESIGHT_ALT_RAD and BORESIGHT_AZ_RAD
        # which should be nan
        for key in visit_keys[7:]:
            self.assertTrue(
                np.isnan(agg_stamps.metadata[key]) and np.isnan(donut_stamps_unpaired.metadata[key])
            )

    def testAggDonutStampsSingleStamp(self) -> None:
        unpaired_datasets = self.butler.query_datasets(
            "donutStampsScienceSensor", collections=self.test_run_name
        )
        quality_datasets = self.butler.query_datasets("donutQualityTable", collections=self.test_run_name)
        donut_stamps_unpaired = [self.butler.get(dataset) for dataset in unpaired_datasets]
        quality_tables = [self.butler.get(dataset) for dataset in quality_datasets]

        # First check that original dataset is length more than 0
        self.assertEqual(len(quality_tables[0]), 3)
        # Leave only one DonutStamp in one of the DonutStamps objects
        quality_tables[0].remove_rows(np.arange(1, 3))
        donut_stamps_intra_new = DonutStamps(donut_stamps_unpaired[0][:1])
        for key in donut_stamps_unpaired[0].metadata.keys():
            donut_stamps_intra_new.metadata[key] = donut_stamps_unpaired[0].metadata[key]
        donut_stamps_unpaired[0] = donut_stamps_intra_new

        # Test that outputs are still created
        agg_donut_task = AggregateDonutStampsUnpairedTask()
        task_out = agg_donut_task.run(donut_stamps_unpaired, quality_tables)
        self.assertEqual(len(task_out.stamps), 4)

    def testAggZernikeTablesRunMissingData(self) -> None:
        zernike_tables = self.butler.query_datasets("zernikes", collections=self.test_run_name)
        zernike_table_list = [self.butler.get(zernike_table) for zernike_table in zernike_tables]
        zernike_table_list = [copy(zernike_table_list[0]) for i in range(4)]
        # First check that original dataset is length more than 0
        self.assertEqual(len(zernike_table_list[0]), 4)
        # Remove all rows in first table
        zernike_table_list[0].remove_rows(np.arange(4))

        # Test that outputs are still created
        agg_zern_task = AggregateZernikeTablesTask(config=AggregateZernikeTablesTaskConfig())
        agg_out = agg_zern_task.run(zernike_table_list)
        self.assertEqual(len(agg_out.raw), 9)
        self.assertEqual(len(agg_out.avg), 3)

    def testAggDonutTablesRunMissingData(self) -> None:
        donutTables = self.butler.query_datasets(
            "donutTable",
            collections=self.test_run_name,
            where="instrument='LSSTCam' AND visit IN (4021123106001)",
        )
        qualityTables = self.butler.query_datasets(
            "donutQualityTable",
            collections=self.test_run_name,
            where="instrument='LSSTCam' AND visit IN (4021123106001)",
        )

        donutTablesDict = {ref.dataId["detector"]: self.butler.get(ref) for ref in donutTables}
        qualityTablesDict = {ref.dataId["detector"]: self.butler.get(ref) for ref in qualityTables}
        camera = self.butler.get("camera", dataId={"instrument": "LSSTCam"}, collections="LSSTCam/calib")

        # Remove all extra-focal donuts from one detector
        qualityTablesDict[94].remove_rows(np.arange(3))

        task = AggregateDonutTablesUnpairedTask()
        agg_donut_table = task.run(camera, donutTablesDict, qualityTablesDict)
        self.assertEqual(len(agg_donut_table.aggregateDonutTable), 3)
