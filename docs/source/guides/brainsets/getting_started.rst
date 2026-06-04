Getting Started with Brainsets
==============================

The ``brainsets`` CLI downloads raw data and runs **Brainset Pipelines** to
produce standardized Session files. See :doc:`../../cli/commands` for the full
command reference.

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

For example:

.. code-block:: console

   $ brainsets prepare perich_miller_population_2018 --cores 8

Each ``brainset_id`` maps to a **Brainset Pipeline** under
``brainsets_pipelines/<brainset_id>/pipeline.py``.

Load the Dataset
----------------

After preparation, load Sessions through the brainset's **Dataset** class in
:mod:`torch_brain.datasets`:

.. code-block:: python

   from torch_brain.datasets import PerichMillerPopulation2018

   dataset = PerichMillerPopulation2018()

See :doc:`index` for combining brainsets with
:class:`~torch_brain.datasets.NestedDataset` and a full example.

See also
--------

* :doc:`anatomy_of_a_brainset`
* :doc:`../sampling/index`
