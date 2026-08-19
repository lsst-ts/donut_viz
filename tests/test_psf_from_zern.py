"""Tests for the psfPanel plotting helper.

These tests guard against psfPanel crashing when every detector has an
empty psf list - the case where no donut pairs were used anywhere in the
visit (previously "ValueError: zero-size array to reduction operation fmax
which has no identity" from computing the colormap limits). Individual
pair-less detectors were already drawn as blank panels; the all-empty visit
must produce a fully blank panel grid the same way, with the colorbar
omitted since there is nothing to attach it to.
"""

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from lsst.donut.viz.psf_from_zern import psfPanel
from lsst.utils.tests import TestCase

DETECTORS = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]


def get_scatters(fig: Figure) -> list[PathCollection]:
    """Return the scatter collections in the figure (the colorbar's
    LineCollection/QuadMesh artists are not PathCollections)."""
    return [c for ax in fig.axes for c in ax.collections if isinstance(c, PathCollection)]


class TestPsfPanel(TestCase):
    def testAllDetectorsEmptyMakesBlankPanels(self) -> None:
        # The regression case: no psf values on any detector must produce a
        # figure of blank panels rather than crashing on the colormap
        # limits, with nothing scattered and the colorbar omitted.
        empty: list[list[float]] = [[] for _ in DETECTORS]
        fig = psfPanel(empty, empty, empty, DETECTORS, dettype="LSSTCam")
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(get_scatters(fig)), 0)

    def testPartiallyEmptyDetectorsStillPlot(self) -> None:
        # Detectors with data are plotted while empty ones are left blank,
        # as before.
        xs = [[100.0, 200.0], [], [], []]
        ys = [[500.0, 600.0], [], [], []]
        psf = [[0.5, 0.7], [], [], []]
        fig = psfPanel(xs, ys, psf, DETECTORS, dettype="LSSTCam")
        self.assertIsInstance(fig, Figure)
        scatters = get_scatters(fig)
        self.assertEqual(len(scatters), 1)
        # The colormap limits come from the finite psf values.
        np.testing.assert_allclose(scatters[0].get_clim(), (0.5, 0.7))
