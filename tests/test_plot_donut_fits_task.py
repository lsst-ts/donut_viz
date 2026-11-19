# This file is part of donut_viz.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Tests for the aiDonut support added to ``PlotDonutFitsTask``.

aiDonut records only the per-donut ``fwhm`` in
``aggregateAOSVisitTableRaw``, so ``PlotDonutFitsTask`` synthesizes the
remaining danish model parameters needed to draw the forward model. These
tests cover ``_fillAiDonutModelMeta`` (the synthesis helper) and the
``aiDonutModelFlux`` config that feeds it.
"""

from unittest import mock

import numpy as np

from lsst.donut.viz.plot_donut_fits_task import (
    PlotDonutFitsTask,
    PlotDonutFitsTaskConfig,
)
from lsst.ts.wep.utils import WfAlgorithmName
from lsst.utils.tests import TestCase

# The danish model parameters getModel requires but aiDonut does not store.
DANISH_MODEL_KEYS = {
    "fit_success",
    "fwhm",
    "model_bkg",
    "model_dx",
    "model_dy",
    "model_flux",
}


def make_task(config: PlotDonutFitsTaskConfig | None = None) -> PlotDonutFitsTask:
    """Build a ``PlotDonutFitsTask`` without contacting the EFD.

    The only network dependency in ``__init__`` is ``makeEfdClient``; the
    instrument and danish factory setup run offline.
    """
    if config is None:
        config = PlotDonutFitsTaskConfig()
    with mock.patch(
        "lsst.donut.viz.plot_donut_fits_task.makeEfdClient",
        return_value=None,
    ):
        return PlotDonutFitsTask(config=config)


def make_ai_meta() -> dict:
    """A per-donut metadata dict shaped like aiDonut's estimatorInfo row.

    aiDonut stores ``fwhm`` as a length-1 array and carries no danish model
    parameters.
    """
    return {
        "algo": WfAlgorithmName.AiDonut.value,
        "fwhm": np.array([1.05], dtype=np.float32),
        "weight": 1.0,
        "blur_clipped": False,
    }


class TestAiDonutModelFluxConfig(TestCase):
    def testDefault(self) -> None:
        # The placeholder flux defaults to a positive value; the drawn model
        # and data are normalized before plotting, so it only sets a scale.
        config = PlotDonutFitsTaskConfig()
        self.assertEqual(config.aiDonutModelFlux, 1.0e6)
        self.assertGreater(config.aiDonutModelFlux, 0.0)


class TestFillAiDonutModelMeta(TestCase):
    def setUp(self) -> None:
        self.task = make_task()

    def testSuppliesAllDanishModelKeys(self) -> None:
        # The synthesized metadata must contain every key getModel needs, so
        # the required-keys gate in run() lets the aiDonut model be drawn.
        filled = self.task._fillAiDonutModelMeta(make_ai_meta())
        self.assertTrue(DANISH_MODEL_KEYS.issubset(filled.keys()))

    def testZeroedOffsetsAndBackground(self) -> None:
        # Centroid offsets and background are zeroed, one value per donut in
        # the [extra, intra] pair.
        filled = self.task._fillAiDonutModelMeta(make_ai_meta())
        np.testing.assert_array_equal(filled["model_dx"], np.zeros(2))
        np.testing.assert_array_equal(filled["model_dy"], np.zeros(2))
        np.testing.assert_array_equal(filled["model_bkg"], np.zeros((2, 1)))
        # One background coefficient per donut -> bkg_order 0 in getModel.
        self.assertEqual(filled["model_bkg"].shape, (2, 1))

    def testFitFlaggedSuccessful(self) -> None:
        # getModel skips drawing when fit_success <= 0, so it must be
        # positive for the synthesized model to be produced.
        filled = self.task._fillAiDonutModelMeta(make_ai_meta())
        self.assertGreater(filled["fit_success"], 0)

    def testFluxFromConfig(self) -> None:
        # The placeholder flux is the configured value, one per donut.
        config = PlotDonutFitsTaskConfig()
        config.aiDonutModelFlux = 2.5e5
        task = make_task(config)
        filled = task._fillAiDonutModelMeta(make_ai_meta())
        np.testing.assert_array_equal(filled["model_flux"], np.full(2, 2.5e5))

    def testFwhmCoercedToScalar(self) -> None:
        # aiDonut stores a length-1 fwhm array, but the danish model expects
        # a scalar.
        filled = self.task._fillAiDonutModelMeta(make_ai_meta())
        self.assertIsInstance(filled["fwhm"], float)
        self.assertAlmostEqual(filled["fwhm"], 1.05, places=5)

    def testFwhmScalarInputPreserved(self) -> None:
        # A danish-style scalar fwhm must also survive as a scalar.
        meta = make_ai_meta()
        meta["fwhm"] = 0.9
        filled = self.task._fillAiDonutModelMeta(meta)
        self.assertIsInstance(filled["fwhm"], float)
        self.assertAlmostEqual(filled["fwhm"], 0.9, places=5)

    def testDoesNotMutateInput(self) -> None:
        # The helper returns a copy; the caller's row metadata is untouched.
        meta = make_ai_meta()
        self.task._fillAiDonutModelMeta(meta)
        self.assertEqual(set(meta.keys()), set(make_ai_meta().keys()))
        self.assertNotIn("model_flux", meta)

    def testPreservesExistingKeys(self) -> None:
        # Non-model metadata (e.g. algo, weight) is carried through.
        filled = self.task._fillAiDonutModelMeta(make_ai_meta())
        self.assertEqual(filled["algo"], WfAlgorithmName.AiDonut.value)
        self.assertEqual(filled["weight"], 1.0)
