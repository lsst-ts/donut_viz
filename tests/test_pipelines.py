# This file is part of ts_wep.
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

import os
import unittest
from glob import glob
from pathlib import Path

from lsst.daf.butler import Butler
from lsst.pipe.base import Pipeline, PipelineGraph
from lsst.utils import getPackageDir


class TestPipeline(unittest.TestCase):
    """Test the pipeline."""

    @classmethod
    def setUpClass(cls):
        moduleDir = getPackageDir("ts_wep")
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.butler = Butler(cls.repoDir)

    def testRapidAnalysisPipelines(self):
        packageDir = getPackageDir("donut_viz")
        # only test production pipelines
        pipelinePattern = Path(packageDir) / "pipelines" / "production"
        files = glob(pipelinePattern.as_posix() + "/*.yaml")
        for filename in files:
            print(f"Testing pipeline from file: {filename}")
            pipeline = Pipeline.fromFile(filename)
            self.assertIsInstance(pipeline, Pipeline)
            pipeline = pipeline.to_graph(registry=self.butler.registry)
            self.assertIsInstance(pipeline, PipelineGraph)

    def testUSDFPipelines(self):
        packageDir = getPackageDir("donut_viz")
        # only test production pipelines
        pipelinePattern = Path(packageDir) / "pipelines" / "production" / "lsstcam_usdf"
        files = glob(pipelinePattern.as_posix() + "/*.yaml")
        for filename in files:
            if filename.endswith("_InFocus.yaml"):
                continue  # skip InFocus pipelines for now
            print(f"Testing pipeline from file: {filename}")
            pipeline = Pipeline.fromFile(filename)
            self.assertIsInstance(pipeline, Pipeline)
            pipeline = pipeline.to_graph(registry=self.butler.registry)
            self.assertIsInstance(pipeline, PipelineGraph)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
