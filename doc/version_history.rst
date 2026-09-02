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

v4.8.5 (2026-09-02)
===================

Documentation
-------------

- Updated license text to be compatible with v0.10.2 of ts_pre_commit_conf. (`RSO-903 <https://rubinobs.atlassian.net//browse/RSO-903>`_)


v4.8.4 (2026-08-28)
===================

Other Changes and Additions
---------------------------

- Updated the license headers using the ``insert-license`` hook. (`DM-55979 <https://rubinobs.atlassian.net//browse/DM-55979>`_)


v4.8.3 (2026-08-25)
===================

New Features
------------

- Add a USDF production pipeline for wavefront estimation with the AiDonut algorithm on refit WCS donut stamps:

  * ``lsstCamCwfsUSDF_AiDonutRefitWcs.yaml`` USDF production pipeline

  It imports ``lsstCamCwfsUSDF_AiDonut.yaml`` and overrides only
  ``aggregateDonutTablesCwfsTask.connections.donutTables`` to ``refitWcsDonutTable``, so the
  AiDonut task configuration, model and step layout are inherited rather than duplicated.
  The override is required when the donut stamps being reused were cut out from the refit WCS
  catalog: ``AggregateDonutTablesCwfsTask`` masks the donut table with the quality table
  positionally, and reading the pre-refit ``donutTable`` instead raises ``IndexError`` because
  the two have different lengths. (`RSO-847 <https://rubinobs.atlassian.net//browse/RSO-847>`_)


v4.8.2 (2026-08-24)
===================

Bug Fixes
---------

- Fixed ``AggregateZernikeTablesTask`` writing an unreadable ``aggregateZernikesRaw`` table when no donut pairs were used on any detector in the visit.

  A zero-row table with fixed-shape vector columns writes to ECSV successfully but cannot be read back (astropy fails with "shape mismatch between value and column specifier"), which broke every downstream consumer of the table. Following the convention of ``CalcZernikesTask.empty()``, the task now writes one NaN placeholder row per detector, marked unused, and ECSV roundtrip regression tests have been added.

  The downstream consumers were also made robust to such visits: the ``AggregateAOSVisitTable*`` tasks now leave their NaN donut-value allocations in place when the donut rows cannot be aligned with the zernike rows for a detector (instead of crashing on the row-count mismatch), and allocate all output columns (including the per-side ``_intra``/``_extra`` ones) up front so the output schema no longer depends on the data; ``psfPanel`` draws a fully blank panel grid with the colorbar omitted when no detector has any psf values, matching its existing blank-panel behaviour for individual pair-less detectors; and ``PlotDonutFitsTask`` tolerates empty donut stamp sets, leaving the affected corner panels and the average-Zernike table blank. (`DM-55596 <https://rubinobs.atlassian.net//browse/DM-55596>`_)


v4.8.1 (2026-08-12)
===================

Bug Fixes
---------

- Fixed `NotImplementedError: memoryview: unsupported format 7w` when building the per-detector Zernike summary table in `PlotAOSTask`.

  `np.unique` on an astropy `Column` used to return a `Column`, whose `.data` gives the underlying array. With NumPy >= 2.3 the new hash-based `unique` path re-wraps the result and astropy demotes it to a plain `ndarray`, where `.data` is the raw PEP 3118 buffer instead. Iterating that buffer fails for unicode dtypes. The `.data` access was unnecessary and has been removed. (`DM-55792 <https://rubinobs.atlassian.net//browse/DM-55792>`_)


v4.8.0 (2026-08-10)
===================

New Features
------------

- Add pipelines for unpaired wavefront estimation with refit WCS using the danish algorithm:

  * ``danishCalcZernikesUnpairedBase.yaml`` ingredient
  * ``lsstCamRapidAnalysisPipeline_UnpairedRefitWcsDanish.yaml`` production pipeline
  * ``lsstCamCwfsUSDF_UnpairedRefitWcsDanish.yaml`` USDF production pipeline (`RSO-837 <https://rubinobs.atlassian.net//browse/RSO-837>`_)


v4.7.2 (2026-07-28)
===================

Other Changes and Additions
---------------------------

- Updated AIDonut model paths (`RSO-852 <https://rubinobs.atlassian.net//browse/RSO-852>`_)


v4.7.1 (2026-07-22)
===================

Bug Fixes
---------

- Fix donut fits problems identified.

  1. Make donut fits plot robust to missing EFD schema
  2. Fix errors from RA: galsim.errors.GalSimRangeError: Invalid obscuration. Value nan not in range [0.0, 1.0]
  3. Residual plots are not appearing correctly
  4. Binning should use a value provided by zernikes table metadata. (`RSO-826 <https://rubinobs.atlassian.net//browse/RSO-826>`_)


v4.7.0 (2026-07-15)
===================

New Features
------------

- Add refitWCS pipelines for Science Sensor (FAM). (`RSO-619 <https://rubinobs.atlassian.net//browse/RSO-619>`_)


v4.6.2 (2026-07-11)
===================

Bug Fixes
---------

- Fixed labeling error in plotDonutFits by switching to axis fractional coordinates. (`RSO-827 <https://rubinobs.atlassian.net//browse/RSO-827>`_)


v4.6.1 (2026-07-10)
===================

Bug Fixes
---------

- Fixed binning error in plotDonutFitsTask. (`RSO-824 <https://rubinobs.atlassian.net//browse/RSO-824>`_)


v4.6.0 (2026-07-09)
===================

New Features
------------

- Added danish v1.2 support. Added functionality for plotDonutFits to just use the danish model saved in the zernikes metadata. (`RSO-800 <https://rubinobs.atlassian.net//browse/RSO-800>`_)
- Changed the maxRefObjects for astrometry task in refitWCS pipeline down to 2048 from the default 8192 to speed up crowded fields. (`RSO-822 <https://rubinobs.atlassian.net//browse/RSO-822>`_)


v4.5.0 (2026-06-23)
===================

New Features
------------

- Turned on triangle mode when estimating Zernikes using Danish. (`RSO-655 <https://rubinobs.atlassian.net//browse/RSO-655>`_)
- Turned on cone mode in estimateZernikesDanishTask. (`rso-549 <https://rubinobs.atlassian.net//browse/rso-549>`_)


v4.4.0 (2026-06-22)
===================

Bug Fixes
---------

- Fixed mypy errors in code and tests. (`RSO-529 <https://rubinobs.atlassian.net//browse/RSO-529>`_)


Other Changes and Additions
---------------------------

- Updated the filters used in refitWCS catalog generation to match the better settings used in the baseline WCS catalog generation. (`RSO-617 <https://rubinobs.atlassian.net//browse/RSO-617>`_)


v4.3.0 (2026-05-29)
===================

Other Changes and Additions
---------------------------

- Added the most updated AIDonut model configuration yaml files both binned and unbinned version. (`DM-133 <https://rubinobs.atlassian.net//browse/DM-133>`_)


v4.2.0 (2026-05-26)
===================

New Features
------------

- Set `doDefect` and `doInterpolate` to True in the base ISR pipeline ingredient. (`RSO-554 <https://rubinobs.atlassian.net//browse/RSO-554>`_)


Bug Fixes
---------

- Stopped trying to make the model in plotDonutFitsTask if the original Danish fit was unsuccessful.
  Added logic to make sure that the lists of intra-focal stamps, extra-focal stamps, and donutQualityTables going into AggregateDonutStampsTask are the same and aligned. (`RSO-562 <https://rubinobs.atlassian.net//browse/RSO-562>`_)


v4.1.1 (2026-05-13)
===================

Bug Fixes
---------

- Modified the Unpaired Science Sensor Extra/Intra USDF pipelines to get the noll indices in a consistent manner with the other Unpaired Zernike pipelines. (`rso-539 <https://rubinobs.atlassian.net//browse/rso-539>`_)


v4.1.0 (2026-05-08)
===================

New Features
------------

- Add ability to run pipelines on science sensors in unpaired mode. (`DM-53676 <https://rubinobs.atlassian.net//browse/DM-53676>`_)


Other Changes and Additions
---------------------------

- Set estimateZernikes.nollIndices to align with expectations of weighted average AI donut model, add USDF AI donut pipeline. (`DM-54785 <https://rubinobs.atlassian.net//browse/DM-54785>`_)


v4.0.0 (2026-05-04)
===================

New Features
------------

- Updated to be compatible with danish v1.0.0. (`DM-54280 <https://rubinobs.atlassian.net//browse/DM-54280>`_)
- Added donut id information into aggregate tables. Trimmed aggregate donut stamps plots to only include the donut stamps that appear in the raw table so they match.  Added the donut id information to the donut fits plots. (`RSO-242 <https://rubinobs.atlassian.net//browse/RSO-242>`_)


Other Changes and Additions
---------------------------

- Convert M2/Camera tilt values from EFD (displayed in a table below donutPlot) from degrees to asec. (`DM-54584 <https://rubinobs.atlassian.net//browse/DM-54584>`_)


v3.6.1 (2026-03-31)
===================

Other Changes and Additions
---------------------------

- Changed science sensor Zernike indices calculated to be consistent with the wavefront sensors. (`DM-54424 <https://rubinobs.atlassian.net//browse/DM-54424>`_)


v3.6.0 (2026-03-26)
===================

New Features
------------

- Enable RA to run AOS Parallel across each half chip. (`DM-53212 <https://rubinobs.atlassian.net//browse/DM-53212>`_)
- Update paths to TARTS data files for new TARTS version (TARTS isn't tagged though, so this news is basically meaningless at present) (`DM-54452 <https://rubinobs.atlassian.net//browse/DM-54452>`_)


v3.5.0 (2026-03-16)
===================

Other Changes and Additions
---------------------------

- Add new pipelines for WCS catalog pipeline directly from boresight WCS. (`DM-53904 <https://rubinobs.atlassian.net//browse/DM-53904>`_)
- Add larger maxFieldDist in directDetect catalogs for initial source selection to get more sources in intra-focal detectors for WCS fitting. (`DM-54201 <https://rubinobs.atlassian.net//browse/DM-54201>`_)


v3.4.0 (2026-02-18)
===================

New Features
------------

- Add SNR information to aggregate tables. (`DM-53908 <https://rubinobs.atlassian.net//browse/DM-53908>`_)


v3.3.0 (2026-01-28)
===================

Bug Fixes
---------

- Fix treatment of intrinsic zernikes when creating danish model in plotDonutFitsTask after changes to outputs in zernikes tables from ts_wep. (`DM-53818 <https://rubinobs.atlassian.net//browse/DM-53818>`_)


Documentation
-------------

- Turn up verbosity in scipy.optimize.least_squares inside danish optimization. Move least_squares danish parameters into pipelines/ingredients. (`DM-53756 <https://rubinobs.atlassian.net//browse/DM-53756>`_)


Other Changes and Additions
---------------------------

- Update refitWCS configuration. (`DM-53585 <https://rubinobs.atlassian.net//browse/DM-53585>`_)
- Changed FAM pipelines to include doDefect. (`DM-53827 <https://rubinobs.atlassian.net//browse/DM-53827>`_)
- Update pipelines to have a maximum of 30 function evaluations per donut when running least squares minimization inside danish. (`DM-53906 <https://rubinobs.atlassian.net//browse/DM-53906>`_)


v3.2.3 (2026-01-06)
===================

Bug Fixes
---------

- Save enough donuts for donut fits plot to fully populate. (`DM-53606 <https://rubinobs.atlassian.net//browse/DM-53606>`_)


Other Changes and Additions
---------------------------

- Turn on linearize and crosstalk steps in default ISR configuration in pipelines. (`DM-53613 <https://rubinobs.atlassian.net//browse/DM-53613>`_)


3.2.2 (2025-12-09)
==================

Other Changes and Additions
---------------------------

- Change printed average Zernikes from the mean of zk_CCS per detector to zk_deviation_CCS by default. (`DM-53576 <https://rubinobs.atlassian.net//browse/DM-53576>`_)


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
