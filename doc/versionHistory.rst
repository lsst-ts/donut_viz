.. _lsst.ts.donut.viz-version_history:

##################
Version History
##################

.._lsst.ts.donut.viz-1.6.8

-------------
1.6.8
-------------

* Switch to post_isr_image and (in some pipelines) v2 steps

.._lsst.ts.donut.viz-1.6.7

-------------
1.6.7
-------------

* Change Jenkinsfile to point back to develop branch of ts_jenkins_shared_library.

.._lsst.ts.donut.viz-1.6.6

-------------
1.6.6
-------------

* Fix pipelines to use cutOutDonutsCwfsPairTask and reassignCwfsCutoutsPairTask names for new connections.
* Adjust pipelines after testing with initial on-sky commissioning data.
* Fix donut stamp metadata when only a single donutStamp in donutStamps.

.._lsst.ts.donut.viz-1.6.5

-------------
1.6.5
-------------

* Use the managedTempFile from RA's CI to allow it to check plot creation.

.._lsst.ts.donut.viz-1.6.4

-------------
1.6.4
-------------

* Fix aggregateDonutTablesCwfsTask to work when tables are not present for all wavefront detectors.
* Fix tests to work with updated ts_wep test data that only has raw data for one pair of wavefront sensors stored.

.._lsst.ts.donut.viz-1.6.3

-------------
1.6.3
-------------

* Make the Zernike pyramid plots more compact when using CWFS data.

.._lsst.ts.donut.viz-1.6.2

-------------
1.6.2
-------------

* Fix positioning and orientation of stamps in PlotDonutCwfsTask

.._lsst.ts.donut.viz-1.6.1

-------------
1.6.1
-------------

* Move AOS pipelines to donut_viz to avoid import in ts_wep.
* Add LSSTCam pipelines.
* Add test_pipelines.py.

.._lsst.ts.donut.viz-1.6.0

-------------
1.6.0
-------------

* Add S11-only mode for LsstCam Full Array Mode.

.._lsst.ts.donut.viz-1.5.0

-------------
1.5.0
-------------

* Add donut plot for LSSTCam corner sensors.

.._lsst.ts.donut.viz-1.4.0

-------------
1.4.0
-------------

* Add CWFS compatible tasks for donut_viz pipeline.

.._lsst.ts.donut.viz-1.3.0

-------------
1.3.0
-------------

* Fix failures that occur when detectors are missing data.
* Add tests for detectors missing data.
* Fix intra, extra labeling in donut plots.
* Add utilities tests.

.._lsst.ts.donut.viz-1.2.3

-------------
1.2.3
-------------

* Fixed bug when we have different numbers of donuts on different detectors.

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
