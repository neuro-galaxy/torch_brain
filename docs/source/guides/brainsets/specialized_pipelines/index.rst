Specialized Brainset Pipelines
==============================

Some data sources have dedicated **Brainset Pipeline** base classes that
handle discovery, download, and common processing steps. Subclass the
appropriate base when your source matches; otherwise use
:doc:`../creating_a_brainset_pipeline`.

.. toctree::
   :maxdepth: 1

   openneuro

Available base classes
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Base class
     - Use when
   * - :class:`~torch_brain.pipeline.openneuro.OpenNeuroPipeline`
     - Public EEG or iEEG datasets on `OpenNeuro <https://openneuro.org/>`_

More base classes will be documented here as they are added. For sources
without a specialized base, implement
:class:`~torch_brain.pipeline.BrainsetPipeline` directly.
