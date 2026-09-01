#########
donut_viz
#########

``donut_viz`` visualizes donuts and their derived products for Rubin Observatory.

It provides the pipeline tasks that aggregate the per-detector Zernike and donut
tables produced by the wavefront estimation pipeline (`ts_wep
<https://github.com/lsst-ts/ts_wep>`_) into visit-level products, and that
generate the diagnostic plots (Zernike pyramids, donut fits, PSF-from-Zernike
scatter plots, and corner-sensor donut plots) used to monitor the Active Optics
System.

In automatic operation, these tasks are run as part of the Main Telescope Active
Optics System (`MTAOS <https://ts-mtaos.lsst.io/index.html>`_) and Rapid
Analysis processing.

The production and ingredient pipelines that combine the ``ts_wep`` wavefront
estimation tasks with the aggregation and plotting tasks defined in this package
live in the ``pipelines`` directory.

Documentation is available at https://donut-viz.lsst.io.
