Working with Brainsets
======================

A **brainset** is a curated collection of standardized neural and behavioral
recordings from a specific publication or experimental source. Each brainset
consists of **Session** files (one H5 file per session) produced by a
**Brainset Pipeline** and accessed at training time through a PyTorch
**Dataset** class.

This guide covers two common paths:

* **Use an existing brainset**: configure storage, run ``brainsets prepare``,
  load the corresponding **Dataset**.
* **Create a new brainset**: implement a **Brainset Pipeline** that
  downloads raw data and writes standardized Session files.

Choose your path
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - I want to…
     - Start here
   * - Download and use a published brainset
     - :doc:`getting_started`
   * - Build a new Brainset Pipeline from scratch
     - :doc:`creating_a_brainset_pipeline`
   * - Target OpenNeuro (EEG / iEEG) with minimal boilerplate
     - :doc:`specialized_pipelines/index`

The brainset lifecycle
----------------------

Raw source data is downloaded and converted into Session H5 files by a
**Brainset Pipeline**, invoked through the ``brainsets`` CLI. Each brainset
ships with a **Dataset** class in :mod:`torch_brain.datasets` that loads
Sessions and exposes **Sampling Intervals** for training. See the
:doc:`../sampling/index` guide for the next step after loading a Dataset.
You can also see :doc:`anatomy_of_a_brainset` for more details on what a brainset contains on disk.

Brainset Pipelines live under ``brainsets_pipelines/<brainset_id>/`` in the
``torch_brain`` repository. The ``brainsets prepare`` command discovers
pipelines from the installed package.

Example walkthrough: Perich & Miller (2018)
-------------------------------------------

This walkthrough prepares a small spiking brainset, loads it, and inspects
one session.

**1. Configure storage directories**

.. code-block:: console

   $ brainsets config set
   Enter raw data directory: ./data/raw
   Enter processed data directory: ./data/processed

**2. Prepare the brainset**

.. code-block:: console

   $ brainsets prepare perich_miller_population_2018 --cores 8

This runs the
`Perich & Miller Brainset Pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
and writes Session H5 files under your processed directory.

**3. Load the Dataset**

.. code-block:: python

   from torch_brain.datasets import PerichMillerPopulation2018

   dataset = PerichMillerPopulation2018()
   session = dataset.get_recording(dataset.recording_ids[0])
   print(session.spikes.timestamps.shape)

Each brainset has a dedicated **Dataset** subclass. Pass ``recording_ids`` to
load a subset of Sessions, or compose multiple brainsets with
:class:`~torch_brain.datasets.NestedDataset`:

.. code-block:: python

   from torch_brain.datasets import (
       NestedDataset,
       PerichMillerPopulation2018,
       PeiPandarinathNLB2021,
   )

   dataset = NestedDataset([
       PerichMillerPopulation2018(),
       PeiPandarinathNLB2021(),
   ])
   # recording_ids look like "PerichMillerPopulation2018/session_1"

**4. Sample windows for training**

Once you have a **Dataset**, use **Sampling Intervals** and a **Sampler** to
draw training windows. Continue with the :doc:`../sampling/index` guide.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   anatomy_of_a_brainset
   building_sessions
   creating_a_brainset_pipeline
   specialized_pipelines/index
