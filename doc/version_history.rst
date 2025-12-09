.. _lsst.ts.donut.viz-version_history:

##################
Version History
##################

.. WARNING: DO NOT MANUALLY EDIT THIS FILE.

   Release notes are now managed using towncrier.
   The following comment marks the start of the automatically managed content.
   For help in how to create the "news fragments" see the README page in the
   doc directory.

   Do not remove the following comment line.

.. towncrier release notes start

3.2.1 (2025-12-09)
==================

Other Changes and Additions
---------------------------

- Run with new ruff settings from ts_pre_commit_conf. (`DM-53557 <https://rubinobs.atlassian.net//browse/DM-53557>`_)


3.2.0 (2025-12-02)
==================

Documentation
-------------

- Changed version history to use towncrier. (`DM-53241 <https://rubinobs.atlassian.net//browse/DM-53241>`_)


.._lsst.ts.donut.viz-3.1.2

-------------
3.1.2
-------------

* Change maxFieldDist to 1.725 deg in catalog creation to further avoid vignetted donuts.

.._lsst.ts.donut.viz-3.1.1

-------------
3.1.1
-------------

* Fix metadata indexing for model donuts in plotDonutFitsTask plots.

.._lsst.ts.donut.viz-3.1.0

-------------
3.1.0
-------------

* Added intrinsic and wavefront deviation coefficients to the aggregated Zernike tables.

.._lsst.ts.donut.viz-3.0.1

-------------
3.0.1
-------------

* Updated pipelines now that EstimateZernikesDanishTask is the default for CalcZernikesTask in ts_wep.

.._lsst.ts.donut.viz-3.0.0

-------------
3.0.0
-------------

Breaking Changes
* Updated method signatures for run() implementations to correctly match the superclass API in PlotDonutFitsTask.
  This change enforces proper interface consistency and MyPy type-safety, but breaks external code that previously passed additional arguments (e.g., zk_avg, aggregate_donut_table, or unpaired stamp sets).
  Downstream users must update their calls to these tasks accordingly.

Improvements
* Comprehensive modernization for static type checking (MyPy) across lsst.ts.donut.viz.
* Added missing type annotations and corrected ambiguous signatures, leading to improved safety and maintainability.
* Removed implicit dictionary-based data passing in favor of explicit typed structures (Astropy Row, QTable, etc.).
* Resolved multiple MyPy-inferred errors related to method overriding, attribute access, and indexability.

.._lsst.ts.donut.viz-2.6.3

-------------
2.6.3
-------------

* Add compatibility with new WEP intrinsics, deviation and opd columns.

.._lsst.ts.donut.viz-2.6.2

-------------
2.6.2
-------------

* Fix USDF Unpaired Danish Pipeline.
* Add Unpaired pipeline for RA.
* Add base ISR config for all AOS pipelines.

.._lsst.ts.donut.viz-2.6.1

-------------
2.6.1
-------------

* Turn on doApplyGains for wepDirectDetect, wepRefitWcs, and TARTS pipelines.

.._lsst.ts.donut.viz-2.6.0

-------------
2.6.0
-------------

* Rearrange production pipelines.
* Add LSSTCam USDF processsing pipelines.
* Move more commonalities into the ingredients to make it easier to maintain consistent pipelines.

.._lsst.ts.donut.viz-2.5.1

-------------
2.5.1
-------------

* Add test for PlotDonutFitsUnpairedTask with missing metadata scenario.

.._lsst.ts.donut.viz-2.5.0

-------------
2.5.0
-------------

* Update getModel in plotDonutFits to use new danish parameters saved in butler metadata to create model image instead of reoptimizing.
* Allow getModel in plotDonutFits to be used easily in interactive mode with the butler metadata.
* Change CWFS test to run danish in order to properly test danish outputs.

.._lsst.ts.donut.viz-2.4.2

-------------
2.4.2
-------------
* Expand plotDonutFits to accommodate more donuts.

.._lsst.ts.donut.viz-2.4.1

-------------
2.4.1
-------------

* Add PlotDonutFitsUnpairedTask for visualizing unpaired donut fits.
* Update PlotDonutFitsTask.getModel to support optional thx/thy parameters for unpaired donut visualization.

.._lsst.ts.donut.viz-2.4.0

-------------
2.4.0
-------------

* Add unpaired aggregation tasks.

.._lsst.ts.donut.viz-2.3.3

-------------
2.3.3
-------------

* Ensure correct aggregation when intra-focal detector is missing.

.._lsst.ts.donut.viz-2.3.2

-------------
2.3.2
-------------

* Synchronizes FAM donut pipeline with corner pipeline.
* Adds stricter max field radius to avoid significantly vignetted donuts.

.._lsst.ts.donut.viz-2.3.1

-------------
2.3.1
-------------

* Fix plotDonutFits to work with old aggregated data.

.._lsst.ts.donut.viz-2.3.0

-------------
2.3.0
-------------

* Added AiDonut RA pipeline.

.._lsst.ts.donut.viz-2.2.3

-------------
2.2.3
-------------

* Add band propagation to metadata in AggregateZernikeTablesTask.

.._lsst.ts.donut.viz-2.2.2

-------------
2.2.2
-------------

* Fix steps in refitWCS pipeline for RA.

.._lsst.ts.donut.viz-2.2.1

-------------
2.2.1
-------------

* Update title text in plotDonutFits

.._lsst.ts.donut.viz-2.2.0

-------------
2.2.0
-------------

* Add refitWCS pipeline for Danish and LSSTCam CWFS data.

.._lsst.ts.donut.viz-2.1.1

-------------
2.1.1
-------------

* Fix bug where key is used regardless of whether it exists

.._lsst.ts.donut.viz-2.1.0

-------------
2.1.0
-------------

* Add plotDonutFitsTask.

.._lsst.ts.donut.viz-2.0.4

-------------
2.0.4
-------------

* Fix importing convertDictToVisitInfo after changes in ts_wep.

.._lsst.ts.donut.viz-2.0.3

-------------
2.0.3
-------------

* Fix how we aggregate estimatorInfo metadata in average AggregateZernikes tables.

.._lsst.ts.donut.viz-2.0.2

-------------
2.0.2
-------------

* Add average danish fwhm values into metadata of average table.
* Change science sensor tests to use post_isr_image.

.._lsst.ts.donut.viz-2.0.1

-------------
2.0.1
-------------

* Update pipelines with pairingPlot, turn on Flat for CWFS ISR, turn off maxPowerGrad

.._lsst.ts.donut.viz-2.0.0

-------------
2.0.0
-------------

* Add PlotCwfsPairingTask, update ISR connections to post_isr_image

.._lsst.ts.donut.viz-1.9.0

-------------
1.9.0
-------------

* Add wavefront estimation algorithm metadata into the Aggregate Zernikes Raw and AOS Raw Visit table metadata.

.._lsst.ts.donut.viz-1.8.1

-------------
1.8.1
-------------

* Updated TIE RA pipeline to match Danish.

.._lsst.ts.donut.viz-1.8.0

-------------
1.8.0
-------------

* Simplify aggregate donut stamps metadata to only include visit level info.

.._lsst.ts.donut.viz-1.7.2

-------------
1.7.2
-------------

* Remove calls to plt.Figure to fix memory leak

.._lsst.ts.donut.viz-1.7.1

-------------
1.7.1
-------------

* Fix upload bug for remote uploads to RubinTV.

.._lsst.ts.donut.viz-1.7.0

-------------
1.7.0
-------------

* Fix donut_viz test ISR configs to use default ISR calibration configs.

.._lsst.ts.donut.viz-1.6.9

-------------
1.6.9
-------------

* Change binning to 2x in rapid analysis Danish pipeline
* Selecting only 4 donuts for Zernike estimation in rapid analysis Danish pipeline. This is to take max advantage of parllelization on the summit.

.._lsst.ts.donut.viz-1.6.8

-------------
1.6.8
-------------

* When aggregating donut stamps, only keep the visit level metadata.

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
