# This file is part of donut_viz.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for the AuxTel monolith pipeline.

``test_pipelines.py`` already checks that every production pipeline parses and
expands into a PipelineGraph, which covers this file's *syntax*. What it cannot
check is the AuxTel-specific *content*: the pipeline would still parse if
``donutDiameter`` reverted to the LSSTCam-sized default that clips an AuxTel
donut, or if ``opticalModel`` were left at ``offAxis``, for which there is no
batoid fit for AuxTel. Both would silently ruin the fit, so they are asserted
here.

``gen3TestRepo`` now carries a LATISS CWFS pair, so
``TestDonutVizPipelineAuxTelRun`` below actually executes the pipeline. The
pair (20260614 seq 17/18, HD 94473) was picked so the fit is well behaved: the
donut sits 50 px from the boresight, keeping it inside the ``onAxis`` model's
validity domain, and Z4 = -330 nm is a real signal rather than noise about
zero. Only gains and overscan are needed for LATISS ISR, so no flat, bias or
linearizer is staged.

A complementary execution test over many on-sky pairs lives in ts_wep as
``tests/task/test_latissMonolithTask.py::TestLatissMonolithTaskOnSky``, gated
on access to ``/repo/main``.
"""

import os
import unittest
from pathlib import Path
from typing import cast

import numpy as np

from lsst.daf.butler import Butler
from lsst.pipe.base import Pipeline, PipelineGraph
from lsst.ts.wep.task import LatissMonolithTaskConfig
from lsst.ts.wep.utils import runProgram, writeCleanUpRepoCmd, writePipetaskCmd
from lsst.utils import getPackageDir
from lsst.utils.tests import TestCase

PIPELINE_NAME = "auxtelMonolithBase.yaml"
TASK_LABEL = "latissMonolithTask"

# The LATISS pair staged in ts_wep's gen3TestRepo. The task keys its outputs on
# the extra-focal visit, following ts_wep's paired convention.
EXTRA_VISIT = 2026061400018
INTRA_VISIT = 2026061400017

# What the same pair gives on /repo/main under the RSO-873 stack. Tolerances
# are loose enough to absorb a danish or scipy point release, but tight enough
# that either of the two regressions the monolith works around -- the 43.5x OPD
# zkRef scaling, and stamps no longer being peak-normalized -- would blow
# straight through them.
EXPECTED_Z4_NM = -330.0
EXPECTED_COST = 2542.6
EXPECTED_FWHM_ARCSEC = 1.92


class TestDonutVizPipelineAuxTel(TestCase):
    """Check the AuxTel monolith pipeline's structure and LATISS config."""

    butler: Butler
    pipeline: Pipeline
    pipeline_graph: PipelineGraph

    @classmethod
    def setUpClass(cls) -> None:
        wep_module_dir = getPackageDir("ts_wep")
        test_repo_dir = os.path.join(wep_module_dir, "tests", "testData", "gen3TestRepo")
        cls.butler = Butler.from_config(test_repo_dir)

        donut_viz_dir = os.getenv("DONUT_VIZ_DIR")
        if donut_viz_dir is None:
            raise RuntimeError("Environment variable DONUT_VIZ_DIR must be set for tests")
        pipeline_path = Path(donut_viz_dir) / "pipelines" / "production" / "lsstcam_usdf" / PIPELINE_NAME
        cls.pipeline = Pipeline.fromFile(pipeline_path.as_posix())
        cls.pipeline_graph = cls.pipeline.to_graph(registry=cls.butler.registry)

    def testPipelineExpands(self) -> None:
        self.assertIsInstance(self.pipeline, Pipeline)
        self.assertIsInstance(self.pipeline_graph, PipelineGraph)

    def testIsASingleTaskPipeline(self) -> None:
        """The whole point of a monolith: one task, one quantum."""
        labels = list(self.pipeline_graph.tasks)
        self.assertEqual(labels, [TASK_LABEL])

    def testInstrumentIsLatiss(self) -> None:
        self.assertEqual(self.pipeline.getInstrument(), "lsst.obs.lsst.Latiss")

    def testQuantumIsPerDetectorNotPerVisit(self) -> None:
        """The task pairs two exposures itself, so it cannot be visit-keyed.

        Its outputs are visit-dimensioned, but the quantum is not.
        """
        task_node = self.pipeline_graph.tasks[TASK_LABEL]
        self.assertEqual(set(task_node.dimensions.names), {"instrument", "detector"})

    def testConnections(self) -> None:
        task_node = self.pipeline_graph.tasks[TASK_LABEL]
        self.assertEqual(set(task_node.inputs), {"raws"})
        self.assertEqual(set(task_node.prerequisite_inputs), {"camera"})
        self.assertEqual(set(task_node.outputs), {"zernikes", "donutStampsExtra", "donutStampsIntra"})

    def testLatissSpecificConfigIsPinned(self) -> None:
        """The values the AuxTel fit depends on, which syntax checks miss."""
        # PipelineGraph types .config as the base PipelineTaskConfig, so narrow
        # it to the concrete class for the attribute checks below.
        config = cast(LatissMonolithTaskConfig, self.pipeline_graph.tasks[TASK_LABEL].config)

        # There is no off-axis batoid fit for AuxTel, so offAxis is wrong.
        self.assertEqual(config.opticalModel, "onAxis")

        # An AuxTel donut spans 194 px, so the ts_wep default of 160 clips it.
        # 228 is what latiss_wep_align derives for dz = 0.8.
        self.assertEqual(config.donutDiameter, 228)
        self.assertGreater(config.donutDiameter, 194)

        self.assertEqual(list(config.nollIndices), list(range(4, 23)))

    def testSubsetAndStep(self) -> None:
        subsets = self.pipeline.subsets
        self.assertIn("step1-latiss", subsets)
        self.assertEqual(set(subsets["step1-latiss"]), {TASK_LABEL})


class TestDonutVizPipelineAuxTelRun(TestCase):
    """Execute the pipeline on the LATISS pair staged in ts_wep's gen3TestRepo.

    Both exposures must be in the same quantum -- the task pairs them itself --
    so the data query names them explicitly rather than relying on a day_obs.
    """

    test_repo_dir: str
    test_run_name: str
    butler: Butler

    @classmethod
    def setUpClass(cls) -> None:
        wep_module_dir = getPackageDir("ts_wep")
        cls.test_repo_dir = os.path.join(wep_module_dir, "tests", "testData", "gen3TestRepo")
        cls.test_run_name = "test_run_latiss_monolith"

        butler = Butler.from_config(cls.test_repo_dir)
        if cls.test_run_name in list(butler.registry.queryCollections()):
            runProgram(writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name))

        donut_viz_dir = os.getenv("DONUT_VIZ_DIR")
        if donut_viz_dir is None:
            raise RuntimeError("Environment variable DONUT_VIZ_DIR must be set for tests")
        pipeline_path = Path(donut_viz_dir) / "pipelines" / "production" / "lsstcam_usdf" / PIPELINE_NAME

        # LATISS ISR needs only raws plus the camera from the curated calibs.
        pipe_cmd = writePipetaskCmd(
            cls.test_repo_dir,
            cls.test_run_name,
            "lsst.obs.lsst.Latiss",
            "LATISS/raw/all,LATISS/calib",
            pipelineYaml=pipeline_path.as_posix(),
        )
        pipe_cmd += f' -d "exposure IN ({INTRA_VISIT}, {EXTRA_VISIT}) AND detector = 0"'
        runProgram(pipe_cmd)

        cls.butler = Butler.from_config(cls.test_repo_dir, collections=[cls.test_run_name])

    @classmethod
    def tearDownClass(cls) -> None:
        runProgram(writeCleanUpRepoCmd(cls.test_repo_dir, cls.test_run_name))

    @property
    def data_id(self) -> dict:
        return {"instrument": "LATISS", "detector": 0, "visit": EXTRA_VISIT}

    def testOutputsAreKeyedOnTheExtraFocalVisit(self) -> None:
        """One quantum in, one set of outputs out, keyed on the extra visit."""
        for dataset_type in ("zernikes", "donutStampsExtra", "donutStampsIntra"):
            refs = list(self.butler.query_datasets(dataset_type, collections=self.test_run_name))
            self.assertEqual(len(refs), 1, f"{dataset_type} should be written exactly once")
            self.assertEqual(refs[0].dataId["visit"], EXTRA_VISIT)

    def testStampsAreAuxTelSized(self) -> None:
        """228 px, not the LSSTCam-sized 160 that would clip a LATISS donut."""
        for dataset_type, expected_visit in (
            ("donutStampsExtra", EXTRA_VISIT),
            ("donutStampsIntra", INTRA_VISIT),
        ):
            stamps = self.butler.get(dataset_type, dataId=self.data_id)
            self.assertEqual(len(stamps), 1)
            self.assertEqual(stamps[0].stamp_im.image.array.shape, (228, 228))
            # Each side keeps its own visit, though the dataId is the pair's.
            self.assertEqual(stamps.metadata["VISIT"], expected_visit)
            self.assertEqual(stamps.metadata["DET_NAME"], "RXX_S00")

    def testDefocalTypesAreNotSwapped(self) -> None:
        """A swap here silently inverts the wavefront sign."""
        self.assertEqual(
            self.butler.get("donutStampsExtra", dataId=self.data_id).metadata["DFC_TYPE"], "extra"
        )
        self.assertEqual(
            self.butler.get("donutStampsIntra", dataId=self.data_id).metadata["DFC_TYPE"], "intra"
        )

    def testZernikeTableStructure(self) -> None:
        zernikes = self.butler.get("zernikes", dataId=self.data_id)

        # One row per pair plus the average; with a single pair they coincide.
        self.assertEqual(list(zernikes["label"]), ["average", "pair1"])
        self.assertTrue(all(zernikes["used"]))

        # The columns the monolith adds on top of the CalcZernikesTask schema.
        for column in ("cost", "fwhm", "nfev", "fit_success"):
            self.assertIn(column, zernikes.colnames)
        for noll in range(4, 23):
            self.assertIn(f"Z{noll}", zernikes.colnames)

        self.assertEqual(zernikes["Z4"].unit, "nm")
        self.assertEqual(zernikes["fwhm"].unit, "arcsec")

        # No intrinsicZernikes connection for AuxTel: NaN by design.
        row = zernikes[zernikes["label"] == "pair1"][0]
        self.assertTrue(np.isnan(row["Z4_intrinsic"].value))

    def testFitProvenanceMetadata(self) -> None:
        """The metadata that records the two AuxTel workarounds are active."""
        meta = self.butler.get("zernikes", dataId=self.data_id).meta

        self.assertEqual(meta["cam_name"], "LATISS")
        self.assertEqual(meta["optical_model"], "onAxis")
        self.assertEqual(meta["donut_diameter"], 228)
        self.assertEqual(list(meta["noll_indices"]), list(range(4, 23)))

        # Regression #1: the OPD-derived zkRef, not getOffAxisCoeff (43.5x too
        # big for AuxTel). Regression #2: stamps must be peak-normalized.
        self.assertTrue(meta["opd_zk_ref"])
        self.assertTrue(meta["peak_normalized_stamps"])

        # Each side's visit is recorded, so a pairing inversion is detectable.
        self.assertEqual(meta["extra"]["visit"], EXTRA_VISIT)
        self.assertEqual(meta["intra"]["visit"], INTRA_VISIT)

    def testFitReproducesRepoMainValues(self) -> None:
        """The numbers themselves -- what a silent physics regression moves."""
        zernikes = self.butler.get("zernikes", dataId=self.data_id)
        row = zernikes[zernikes["label"] == "pair1"][0]

        self.assertTrue(row["fit_success"])
        self.assertAlmostEqual(row["Z4"].to_value("nm"), EXPECTED_Z4_NM, delta=5.0)
        self.assertAlmostEqual(row["cost"], EXPECTED_COST, delta=50.0)
        self.assertAlmostEqual(row["fwhm"].to_value("arcsec"), EXPECTED_FWHM_ARCSEC, delta=0.1)

        # Sanity floor: a fit that has gone off the rails shows up as absurd
        # low-order terms long before the tolerances above would catch it.
        for noll in range(4, 23):
            self.assertLess(abs(row[f"Z{noll}"].to_value("nm")), 2000.0)


if __name__ == "__main__":
    unittest.main()
