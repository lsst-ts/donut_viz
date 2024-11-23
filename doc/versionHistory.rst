.. _lsst.ts.donut.viz-version_history:

##################
Version History
##################

.._lsst.ts.donut.viz-1.2.2

-------------
1.2.2
-------------

* Change tasks to use run methods inside runQuantum to make things easier to test and enable running tasks interactively.

.._lsst.ts.donut.viz-1.2.1

-------------
1.2.1
-------------

* Add PlotPsfZernTask that creates a per-detector scatter plot of the PSF calculated with the convertZernikesToPsfWidth method.

.. _lsst.ts.donut.viz-1.2.0:

-------------
1.2.0
-------------

* Enable sparse zernikes to be used from ts_wep.

.. _lsst.ts.donut.viz-1.1.2:

-------------
1.1.2
-------------

* Enabled Zernike pyramids to work with lower jmax.

.. _lsst.ts.donut.viz-1.1.1:

-------------
1.1.1
-------------

* Add deferQueryConstraint=True to input connections that have no incoming data from an intrafocal DataId.
* Add dummyExposureJoiner to get ``exposure`` dimension into quantum graph generation to relate visit and group.

.. _lsst.ts.donut.viz-1.1.0:

-------------
1.1.0
-------------

* Add tests for full donut_viz pipeline.
* Add changelog github action.
* Add Jenkinsfile.

.. _lsst.ts.donut.viz-1.0.0:

-------------
1.0.0
-------------

* First official release of the donut_viz package
