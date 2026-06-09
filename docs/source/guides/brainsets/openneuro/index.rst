Adding an OpenNeuro Brainset
============================

.. py:currentmodule:: torch_brain.pipeline.openneuro
.. |OpenNeuroPipeline| replace:: :class:`OpenNeuroPipeline`

This guide explains how to build a **Brainset Pipeline** for EEG or iEEG
datasets on `OpenNeuro <https://openneuro.org/>`_. :class:`OpenNeuroPipeline`
handles S3 discovery, BIDS download, signal extraction, and Session H5 storage.

Before starting, see :doc:`../adding_a_brainset/index` for the general pipeline
workflow.

Minimal example
---------------

.. code-block:: python

   from torch_brain.pipeline.openneuro import OpenNeuroPipeline


   class Pipeline(OpenNeuroPipeline):
       modality = "eeg"  # or "ieeg"
       brainset_id = "my_sleep_study_ds005555"
       dataset_id = "ds005555"
       origin_version = "1.0.0"
       derived_version = "1.0.0"
       description = "Sleep recordings from OpenNeuro"

Add the pipeline under ``brainsets_pipelines/my_sleep_study_ds005555/`` and run:

.. code-block:: console

   brainsets prepare my_sleep_study_ds005555

Required attributes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``modality``
     - ``"eeg"`` (scalp/wearable) or ``"ieeg"`` (intracranial)
   * - ``dataset_id``
     - OpenNeuro ID in ``ds######`` format (exactly six digits)
   * - ``brainset_id``
     - Unique identifier, e.g. ``klinzing_sleep_ds005555``
   * - ``origin_version``
     - OpenNeuro snapshot version used when building the pipeline (check dataset
       "Snapshots" on openneuro.org)
   * - ``derived_version``
     - Your processing logic version; increment when processing changes

``origin_version`` documents which OpenNeuro snapshot the pipeline was tested
against. At runtime, the latest snapshot is downloaded regardless; a mismatch
triggers the ``--on-version-mismatch`` policy (see
:doc:`../../../cli/commands`).

Optional customization
----------------------

``description``
   Human-readable text stored in brainset metadata.

``CHANNEL_NAME_REMAPPING``
   Map raw channel names to standardized names::

       CHANNEL_NAME_REMAPPING = {"PSG_F3": "F3", "PSG_F4": "F4"}

``TYPE_CHANNELS_REMAPPING``
   Group channels by physiological type::

       TYPE_CHANNELS_REMAPPING = {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

For per-recording mappings, override ``get_channel_name_remapping(recording_id)``
and ``get_type_channels_remapping(recording_id)``. For extra processing beyond
the defaults, override ``process()`` and call ``self.process_common(download_output)``.

Use **UPPERCASE** channel names and types for consistency across EEG/iEEG
brainsets.

Examples
--------

Working OpenNeuro pipelines in the repository:

* `shirazi_hbnr1_ds005505 <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/shirazi_hbnr1_ds005505/pipeline.py>`_ —
  identical channels across recordings
* `klinzing_sleep_ds005555 <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/klinzing_sleep_ds005555/pipeline.py>`_ —
  per-recording channel mappings

See :class:`OpenNeuroPipeline` API documentation for full details.
