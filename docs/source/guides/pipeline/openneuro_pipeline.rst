Creating an OpenNeuroPipeline
=============================

.. py:currentmodule:: brainsets.utils.openneuro
.. |OpenNeuroPipeline| replace:: :class:`OpenNeuroPipeline`

This guide shows you how to build a pipeline that automatically downloads, processes, and standardizes publicly available datasets from `OpenNeuro <https://openneuro.org/>`_. 
The :class:`OpenNeuroPipeline` class currently supports the following data modalities: EEG (electroencephalography) and iEEG (intracranial EEG).

Process brain recordings with minimal boilerplate:

.. code-block:: text

    OpenNeuro S3 → Download → Process → Save to HDF5


What's an OpenNeuro Pipeline?
-----------------------------

An OpenNeuro pipeline is a Python class that extends |OpenNeuroPipeline| and automates the typical data preparation workflow for neuroimaging datasets from OpenNeuro. 
The :class:`OpenNeuroPipeline` class handles the entire workflow, including:

- 🔍 Discovering recordings from OpenNeuro's S3 bucket based on modality
- ⬇️ Downloading BIDS-compliant files
- 🔧 Processing: extracting data (signal) and metadata (channel names and types)
- 💾 Storing: organized HDF5 files with full provenance


Get Started in 3 Minutes
------------------------

Here's a working minimal pipeline:

.. code-block:: python

    from brainsets.utils.openneuro import OpenNeuroPipeline

    class Pipeline(OpenNeuroPipeline):
        modality = "eeg"  # or "ieeg"
        brainset_id = "my_sleep_study_ds005555"
        dataset_id = "ds005555"
        origin_version = "1.0.0"  # Check OpenNeuro for this!
        derived_version = "1.0.0"  # Version of your processing pipeline
        description = "Sleep recordings from OpenNeuro"

**That's it!** The rest is inherited (but can be customized 🛠️).

Add your new pipeline class in its own directory under ``brainsets_pipelines/`` (for example, ``brainsets_pipelines/my_sleep_study_ds005555/pipeline.py``). To run it:

.. code-block:: console

    uv run brainsets prepare my_sleep_study_ds005555

----

Real-World Examples
-------------------

Before diving into details, check out working implementations in the `brainsets_pipelines` directory:

.. list-table::
   :header-rows: 1

   * - Example
     - Use When
     - Complexity
   * - `shirazi_hbnr1_ds005505 <https://github.com/neuro-galaxy/brainsets/blob/main/brainsets_pipelines/shirazi_hbnr1_ds005505/pipeline.py>`_
     - All recordings have identical channels
     - Simple ⭐
   * - `klinzing_sleep_ds005555 <https://github.com/neuro-galaxy/brainsets/blob/main/brainsets_pipelines/klinzing_sleep_ds005555/pipeline.py>`_
     - Different recordings need different channel mappings
     - Complex ⭐⭐⭐


The Five Required Attributes
-----------------------------

Every object extending |OpenNeuroPipeline| **must** have these five attributes:


1. ``modality`` – EEG or iEEG?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``modality`` to ``"eeg"`` for scalp/wearable EEG datasets or ``"ieeg"`` for intracranial datasets. This drives BIDS recording discovery from OpenNeuro S3.

.. code-block:: python

    modality = "eeg"   # scalp or headband EEG
    modality = "ieeg"  # intracranial EEG / ECoG


2. ``dataset_id`` – Which OpenNeuro dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset identifier must use strict OpenNeuro format: ``ds`` followed by exactly 6 digits.

.. code-block:: python

    class Pipeline(OpenNeuroPipeline):
        modality = "eeg"
        dataset_id = "ds005555"      # ✅ Valid
        dataset_id = "5555"          # ❌ Invalid
        dataset_id = "ds5555"        # ❌ Invalid

.. note::

   Dataset ID validation happens internally during pipeline initialization. Invalid IDs raise an error before data discovery.


3. ``brainset_id`` – Your unique name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A descriptive ID for your processed brainset. This should be unique within the ``brainsets_pipelines/`` directory and should be a valid Python identifier. Recommended naming scheme:

.. code-block:: python

    brainset_id = "klinzing_sleep_ds005555"
    #               └─ author's last name
    #                        └─ a short label describing the main task or paradigm in the dataset
    #                              └─ dataset ID


4. ``origin_version`` – The dataset version you used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``origin_version`` to the exact version of the OpenNeuro dataset you used when building and testing your pipeline:

.. code-block:: python

    class Pipeline(OpenNeuroPipeline):
        modality = "eeg"
        dataset_id = "ds005555"
        origin_version = "1.0.0"  # The version available when you created this pipeline

⚠️ **Why hardcode?** OpenNeuro datasets evolve. A newer version might have different subjects, missing files, or structural changes. Hardcoding ensures you document which version your pipeline was originally built and tested with.

**How to find the version:**

1. Visit `https://openneuro.org/ <https://openneuro.org/>`_ and navigate to your dataset
2. Look for "Snapshots" or "Version History"
3. Note the latest version tag (e.g., ``1.0.0``, ``2.1.3``)
4. For details on changes between versions, check the ``CHANGES`` file in the dataset

----

.. important::
   **What happens if versions mismatch during runtime?**

   *This section is more relevant for pipeline users running* ``prepare`` *commands, not pipeline authors.*

   When you run a pipeline, the system automatically:

   1. Fetches the latest snapshot tag from OpenNeuro
   2. Compares it to the ``origin_version`` hardcoded in the pipeline class
   3. If they differ, applies the ``--on-version-mismatch`` policy you specified (or ``prompt`` by default)

   **Controlling mismatch behavior with --on-version-mismatch**

   Users can control how an Open Neuro pipeline responds to version mismatches using the CLI flag ``--on-version-mismatch``:

   .. list-table::
      :header-rows: 1

      * - Option
        - Behavior
        - When to Use
      * - ``prompt`` (default)
        - Interactive session: asks user to confirm. Non-interactive: **raises ValueError immediately** with guidance to use ``continue`` or ``abort``.
        - Local development; lets you decide per run whether to continue (interactive only).
      * - ``continue``
        - Warns and proceeds with latest version.
        - Automation/CI: accept version drift when tests pass.
      * - ``abort``
        - Raises error immediately if mismatch detected.
        - Strict reproducibility: fail fast rather than risk inconsistency.

   **Examples:**

   .. code-block:: console

       # Interactive prompt (default behavior)
       uv run brainsets prepare my_brainset

       # Always continue with a warning
       uv run brainsets prepare my_brainset --on-version-mismatch continue

       # Fail if version mismatch
       uv run brainsets prepare my_brainset --on-version-mismatch abort

       # In CI/automation (non-interactive), must set to continue or abort
       # This will fail with clear error message:
       uv run brainsets prepare my_brainset  # (would error: prompt not allowed in non-interactive mode)
       # Instead, use:
       uv run brainsets prepare my_brainset --on-version-mismatch continue

   **Important:** Regardless of which policy you choose, if you continue, OpenNeuro S3 will serve the latest snapshot for downloads. The version check is informational—it warns you to potential differences but does not download or use the original ``origin_version``.

   **Example warning message:**

   .. code-block:: text

       ⚠️ Dataset version '1.0.0' was used to create the brainset pipeline for dataset 'ds005555',
       but the latest available version on OpenNeuro is '1.2.0'.
       Downloading data or running the pipeline now will use the latest version,
       which may differ from the original version used, potentially causing errors or inconsistencies.
       Check the CHANGES file of the dataset for details about the differences between versions.


   **When versions don't match: What should I do?**

   .. list-table::
      :header-rows: 1

      * - Scenario
        - Action
      * - Changes are minor or unrelated to your pipeline
        - Accept the warning and continue; the latest version is used.
      * - You've tested with the new version
        - Update ``origin_version`` to match, thoroughly test your pipeline, and re-run.
      * - Changes break your pipeline
        - Consider fetching the old version from OpenNeuro archives manually or adjust your pipeline for the new version.

----

5. ``derived_version`` – Your processing pipeline version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``derived_version`` to track the version of **your processing pipeline and its output**:

.. code-block:: python

    class Pipeline(OpenNeuroPipeline):
        modality = "eeg"
        dataset_id = "ds005555"
        origin_version = "1.0.0"
        derived_version = "1.0.0"  # Increment when you change processing logic

While ``origin_version`` tracks the version of the source dataset from OpenNeuro, ``derived_version`` tracks the version of your processing pipeline. This version is stored in the brainset metadata and helps users understand which version of the processing logic was used to generate the data.

**Why maintain a separate version?**

- Allows you to iterate on your processing pipeline independently from dataset updates
- Documents processing pipeline changes in metadata
- Enables reproducibility and traceability of derived results
- Increment this version when you modify channel mappings, filtering, or any processing logic

----


Optional Attributes (with sensible defaults)
--------------------------------------------

Want to customize? These are all optional:


``description``
~~~~~~~~~~~~~~~~

Human-readable description that appears in metadata:

.. code-block:: python

    description = (
        "The Bitbrain Open Access Sleep (BOAS) dataset contains simultaneous "
        "recordings from a clinical PSG system and wearable EEG headband."
    )


``CHANNEL_NAME_REMAPPING``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rename raw channel names from the original dataset to standardized names.

**Dictionary structure:**

- **Keys** are the original/recorded channel names as strings (e.g., those found in the raw data).
- **Values** are the standardized names to which you wish to map them as strings.

.. code-block:: python

    CHANNEL_NAME_REMAPPING = {
        "PSG_F3": "F3",
        "PSG_F4": "F4",
        "PSG_EOG": "EOG",
    }


``TYPE_CHANNELS_REMAPPING``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group channels by physiological type.

**Dictionary structure:**

- **Keys:** Strings representing physiological channel types (e.g., ``"EEG"``, ``"EOG"``, ``"EMG"``).
- **Values:** Lists of string channel names that belong to the given type.  
  These names can be either the original channel names as found in the raw dataset or the standardized names you have mapped them to.

.. code-block:: python

    TYPE_CHANNELS_REMAPPING = {
        "EEG": ["F3", "F4", "C3", "C4"],
        "EOG": ["EOG"],
        "EMG": ["EMG"],
    }

.. note::

   While brainsets permits flexibility in naming schemes to accommodate various datasets, we recommend using **UPPERCASE names** for all channel names and types—both keys and values—wherever possible. This helps to align with widespread EEG and iEEG naming conventions and ensures consistency across datasets, promoting standardization without reducing adaptability.


Advanced: Customize Per Recording
---------------------------------

If you need different channel mappings or groupings for particular recordings (e.g., based on acquisition type, subject, or any other property), override these methods:


``get_channel_name_remapping(recording_id)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return different channel mappings based on the recording:

.. code-block:: python

    def get_channel_name_remapping(self, recording_id):
        if "acq-headband" in recording_id:
            return {
                "HB_1": "AF7",
                "HB_2": "AF8",
                "HB_PULSE": "PULSE",
            }
        return {
            "PSG_F3": "F3",
            "PSG_F4": "F4",
            "PSG_EOG": "EOG",
        }


``get_type_channels_remapping(recording_id)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return different channel groupings based on the recording:

.. code-block:: python

    def get_type_channels_remapping(self, recording_id):
        if "acq-headband" in recording_id:
            return {
                "EEG": ["AF7", "AF8"],
                "PPG": ["PULSE"],
            }
        return {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG"],
        }


``process(download_output)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add custom processing beyond the default:

.. code-block:: python

    def process(self, download_output):
        # Get default processing
        result = self.process_common(download_output)
        if result is None:
            return  # Already processed
        
        data, store_path = result
        
        # Add your custom processing here
        # e.g., apply filters, remove artifacts
        
        # Save the result
        import h5py
        from brainsets import serialize_fn_map
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


What's Next?
------------

1. ✅ Pick a dataset from `OpenNeuro <https://openneuro.org/>`_
2. ✅ Copy the minimal example above
3. ✅ Update ``modality``, ``dataset_id``, ``brainset_id``, ``origin_version``, and ``derived_version``
4. ✅ Add channel name and/or type mappings if needed
5. ✅ Run: ``uv run brainsets prepare <my_brainset_id>``
6. ✅ Done!

For more details, check the base class docstrings in the :mod:`brainsets.utils.openneuro.pipeline` module.

See also: :doc:`Creating a BrainsetPipeline <create_pipeline>` for general pipeline information.
