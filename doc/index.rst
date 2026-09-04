.. |developer| replace:: *Bryce Kalmbach <brycek@slac.stanford.edu>* and *Chris Suberlak <suberlak@uw.edu>*
.. |product_owner| replace:: *Sandrine Thomas <sthomas@lsst.org>*

#########
donut_viz
#########

.. image:: https://img.shields.io/badge/GitHub-donut__viz-green.svg
    :target: https://github.com/lsst-ts/donut_viz
.. image:: https://img.shields.io/badge/Jenkins-donut__viz-green.svg
    :target: https://tssw-ci.lsst.org/job/LSST_Telescope-and-Site/job/donut_viz

.. _Overview:

Overview
========

``donut_viz`` visualizes donuts and their derived products for Rubin Observatory.
It provides the pipeline tasks that aggregate the per-detector Zernike and donut tables produced by the wavefront estimation pipeline (`ts_wep <https://ts-wep.lsst.io/index.html>`_) into visit-level products, and that generate the diagnostic plots (Zernike pyramids, donut fits, PSF-from-Zernike scatter plots, and the corner-sensor donut plots) used to monitor the Active Optics System.

In automatic operation, these tasks are run as part of the Main Telescope Active Optics System (`MTAOS <https://ts-mtaos.lsst.io/index.html>`_) and Rapid Analysis processing.

The badges above navigate to the GitHub repository for the ``donut_viz`` code and Jenkins CI jobs.

.. _Pipelines:

Pipelines
=========

``donut_viz`` ships the production and ingredient pipelines that combine the wavefront estimation tasks from ``ts_wep`` with the aggregation and plotting tasks defined in this package.
These YAML pipelines live in the ``pipelines`` directory of the repository and cover corner wavefront sensor (CWFS), full-array-mode (FAM), science sensor, and unpaired processing for both the TIE and Danish wavefront estimation algorithms.

.. _Python_API:

Python API Reference
====================

.. automodapi:: lsst.donut.viz
    :no-main-docstr:
    :no-inheritance-diagram:

.. _Version_History:

Version History
===============

The version history is at the following link.

.. toctree::
    version_history
    :maxdepth: 1

The released version is `here <https://github.com/lsst-ts/donut_viz/releases>`_.

.. _Contact_Personnel:

Contact Personnel
=================

For questions not covered in the documentation, emails should be addressed to the developers: |developer|.
The product owner is |product_owner|.

This page was last modified |today|.
