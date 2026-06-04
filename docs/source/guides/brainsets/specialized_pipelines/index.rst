Specialized Brainset Pipelines
==============================

Some data sources provide **Brainset Pipeline** base classes that implement
discovery, download, and common processing. Subclass a base when your source
matches; otherwise see :doc:`../creating_a_brainset_pipeline`.

.. toctree::
   :maxdepth: 1

   openneuro

:class:`~torch_brain.pipeline.openneuro.OpenNeuroPipeline` supports public EEG
and iEEG datasets on `OpenNeuro <https://openneuro.org/>`__. See :doc:`openneuro`.

Additional base classes will be documented here as they are added.
