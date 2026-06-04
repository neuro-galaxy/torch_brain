Working with Brainsets
======================

This guide explains how to use and create **brainsets**: curated collections
of standardized neural and behavioral recordings from a publication or
experimental source. A brainset is stored as **Session** H5 files (one file per
session), produced by a **Brainset Pipeline** and loaded at training time
through a PyTorch **Dataset** class in :mod:`torch_brain.datasets`.

Before starting, see :doc:`anatomy_of_a_brainset` for definitions of Session,
Brainset Pipeline, and Dataset on disk and in code.

To download and use a published brainset, see :doc:`getting_started`. To
implement a new pipeline, see :doc:`creating_a_brainset_pipeline`. For OpenNeuro
EEG or iEEG sources, see :doc:`specialized_pipelines/index`.

Lifecycle
---------

A **Brainset Pipeline** downloads raw source data and writes Session files under
your processed directory. Run it with ``brainsets prepare``; pipeline code lives
in ``brainsets_pipelines/<brainset_id>/`` in the ``torch_brain`` repository.

Each brainset provides a **Dataset** subclass that reads those Sessions and
exposes **Sampling Intervals** for training. After loading a Dataset, see
:doc:`../sampling/index`.

Example: Perich & Miller (2018)
-------------------------------

The following example prepares a spiking brainset, loads one session, and
composes multiple brainsets. Configure storage and run ``brainsets prepare`` as
described in :doc:`getting_started`:

.. code-block:: console

   $ brainsets prepare perich_miller_population_2018 --cores 8

The
`Perich & Miller pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
writes Session H5 files under your processed directory.

Load the Dataset
^^^^^^^^^^^^^^^^

.. code-block:: python

   from torch_brain.datasets import PerichMillerPopulation2018

   dataset = PerichMillerPopulation2018()
   session = dataset.get_recording(dataset.recording_ids[0])
   print(session.spikes.timestamps.shape)

Pass ``recording_ids`` to load a subset of Sessions, or combine brainsets with
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

Sample windows for training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use **Sampling Intervals** and a **Sampler** to draw training windows; see
:doc:`../sampling/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   anatomy_of_a_brainset
   building_sessions
   creating_a_brainset_pipeline
   specialized_pipelines/index
