Prepare a Brainset
==================

The ``brainsets`` CLI downloads raw data and runs **Brainset Pipelines** to
produce standardized Session H5 files. See :doc:`../../../cli/commands` for the
full command reference.

Configure storage
-----------------

Set where raw downloads and processed Session files are stored:

.. code-block:: console

   $ brainsets config set

.. code-block:: console

   $ brainsets config show

List available brainsets
------------------------

.. code-block:: console

   $ brainsets list

Prepare a brainset
------------------

.. code-block:: console

   $ brainsets prepare <brainset_id> --cores 8

For example, the Perich & Miller (2018) spiking brainset:

.. code-block:: console

   $ brainsets prepare perich_miller_population_2018 --cores 8

Each ``brainset_id`` maps to a **Brainset Pipeline** at
``brainsets_pipelines/<brainset_id>/pipeline.py`` in the ``torch_brain``
repository.

Next, see :doc:`load_dataset` to load the prepared Sessions.
