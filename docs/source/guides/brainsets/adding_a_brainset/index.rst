Adding a New Brainset
=====================

This guide explains how to implement a **Brainset Pipeline** that downloads raw
recordings and writes standardized Session H5 files, then contribute it to
``torch_brain``.

Before starting, complete :doc:`../getting_started/index` so storage paths and
``brainsets prepare`` are familiar.

.. toctree::
   :maxdepth: 1

   setup
   pipeline
   build_session
   save_to_disk
   contribute


If the brainset you are adding is an OpenNeuro EEG or iEEG dataset, see :doc:`../openneuro/index` instead.
