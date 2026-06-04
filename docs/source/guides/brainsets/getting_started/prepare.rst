Prepare a Brainset
==================

A **brainset** is a standardized collection of **Sessions** from one publication
or data source. Each Session is one H5 file (a single recording period). A
**Brainset Pipeline** downloads raw source data and converts each Session into
that H5 format; pipelines subclass
:class:`~torch_brain.pipeline.BrainsetPipeline` and run through the ``brainsets``
CLI.

The ``brainsets`` CLI orchestrates download and processing. See
:doc:`../../../cli/commands` for the full command reference.

Configure storage
-----------------

First, we set where raw downloads and processed Session files are stored:

.. code-block:: console

   $ brainsets config set

.. code-block:: console

   $ brainsets config show

We can list all available brainsets:
------------------------

.. code-block:: console

   $ brainsets list

Each ``brainset_id`` maps to a pipeline at
``brainsets_pipelines/<brainset_id>/pipeline.py`` in the ``torch_brain``
repository.

We can prepare a brainset using the ``brainsets prepare`` command:
------------------

.. code-block:: console

   $ brainsets prepare <brainset_id> --cores 8

Preparing a brainset involves downloading the raw data and processing it into a standardized Session H5 file.
For example, we prepare the Perich & Miller (2018) spiking brainset using 8 cores, meaning 8 sessions will be processed in parallel.

.. code-block:: console

   $ brainsets prepare perich_miller_population_2018 --cores 8

After preparation, files are laid out as follows:

.. code-block:: text

   <raw_dir>/<brainset_id>/          # downloaded source files
   <processed_dir>/<brainset_id>/    # Session H5 files
       session_1.h5
       session_2.h5
       ...
