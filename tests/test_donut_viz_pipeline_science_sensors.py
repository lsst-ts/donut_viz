import os

import numpy as np
from lsst.daf.butler import Butler
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

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.camera_name = "LSSTCam"
        test_pipeline = os.path.join(
            os.getenv("DONUT_VIZ_DIR"),
            "pipelines",
            "scienceDirectDetectPipeline.yaml",
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
        agg_zern_avg = self.butler.get(average_dataset_list[0])
        self.assertEqual(len(agg_zern_avg), 2)
        self.assertCountEqual(agg_zern_avg["detector"], ["R22_S10", "R22_S11"])
        self.assertCountEqual(agg_zern_avg.meta.keys(), self.meta_keys)

    def testAggregateZernikesRaw(self):
        raw_dataset_list = list(
            self.butler.query_datasets(
                "aggregateZernikesRaw", collections=self.test_run_name
            )
        )
        self.assertEqual(len(raw_dataset_list), 1)
        agg_zern_raw = self.butler.get(raw_dataset_list[0])
        self.assertEqual(len(agg_zern_raw), 6)
        self.assertCountEqual(
            agg_zern_raw["detector"], ["R22_S10"] * 3 + ["R22_S11"] * 3
        )
        self.assertCountEqual(agg_zern_raw.meta.keys(), self.meta_keys)

    def testAggregateDonuts(self):
        donut_table_list = list(
            self.butler.query_datasets(
                "aggregateDonutTable", collections=self.test_run_name
            )
        )
        self.assertEqual(len(donut_table_list), 1)
        agg_donut_table = self.butler.get(donut_table_list[0])
        self.assertEqual(len(agg_donut_table), 12)
        self.assertCountEqual(
            agg_donut_table["detector"], ["R22_S10"] * 6 + ["R22_S11"] * 6
        )
        self.assertCountEqual(agg_donut_table["focusZ"].value, [1.5] * 6 + [-1.5] * 6)
        self.assertCountEqual(
            agg_donut_table.meta.keys(), ["extra", "intra", "average"]
        )
        donut_meta_keys = self.meta_keys + ["focusZ"]
        donut_meta_keys.remove("nollIndices")
        for key in ["extra", "intra"]:
            self.assertCountEqual(agg_donut_table.meta[key].keys(), donut_meta_keys)
        donut_meta_keys.remove("focusZ")
        donut_meta_keys.remove("visit")
        self.assertCountEqual(agg_donut_table.meta["average"], donut_meta_keys)

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
        intra_donuts = self.butler.get(intra_dataset_list[0])
        extra_donuts = self.butler.get(extra_dataset_list[0])
        self.assertEqual(len(intra_donuts), 2)
        self.assertEqual(len(extra_donuts), 2)
        intra_meta = intra_donuts.metadata.toDict()
        extra_meta = extra_donuts.metadata.toDict()
        self.assertCountEqual(intra_meta["DET_NAME"], ["R22_S10", "R22_S11"])
        self.assertCountEqual(intra_meta["DFC_TYPE"], ["intra"] * 2)
        self.assertCountEqual(extra_meta["DET_NAME"], ["R22_S10", "R22_S11"])
        self.assertCountEqual(extra_meta["DFC_TYPE"], ["extra"] * 2)

    def testAggregateAOSVisitTableRaw(self):
        raw_visit_table_list = list(
            self.butler.query_datasets(
                "aggregateAOSVisitTableRaw", collections=self.test_run_name
            )
        )
        self.assertEqual(len(raw_visit_table_list), 1)
        raw_visit_table = self.butler.get(raw_visit_table_list[0])
        self.assertCountEqual(raw_visit_table.meta.keys(), self.meta_keys)
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

        intrinsic_dataset_list = list(
            self.butler.query_datasets(
                "intrinsicZernikePyramid", collections=self.test_run_name
            )
        )
        self.assertEqual(len(intrinsic_dataset_list), 1)

        residual_dataset_list = list(
            self.butler.query_datasets(
                "residualZernikePyramid", collections=self.test_run_name
            )
        )
        self.assertEqual(len(residual_dataset_list), 1)

    def testDonutPlotTask(self):
        # Test that plots exist in butler
        intra_dataset_list = list(
            self.butler.query_datasets("donutPlotIntra", collections=self.test_run_name)
        )
        self.assertEqual(len(intra_dataset_list), 1)

        extra_dataset_list = list(
            self.butler.query_datasets("donutPlotExtra", collections=self.test_run_name)
        )
        self.assertEqual(len(extra_dataset_list), 1)

    def testPlotPsfZernTask(self):
        # Test that plots exist in butler
        psf_zern_dataset_list = list(
            self.butler.query_datasets(
                "psfFromZernPanel", collections=self.test_run_name
            )
        )
        self.assertEqual(len(psf_zern_dataset_list), 1)
