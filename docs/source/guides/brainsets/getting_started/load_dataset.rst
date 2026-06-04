Load a Dataset
==============

After ``brainsets prepare``, load Sessions through the brainset's **Dataset**
class in :mod:`torch_brain.datasets`.

Example: Perich & Miller (2018)
-------------------------------

.. code-block:: python

   from torch_brain.datasets import PerichMillerPopulation2018

   dataset = PerichMillerPopulation2018()
   session = dataset.get_recording(dataset.recording_ids[0])
   print(session.spikes.timestamps.shape)

The
`Perich & Miller pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
produced the Session H5 files under your processed directory.

Combine brainsets
-----------------

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
---------------------------

Each **Dataset** exposes **Sampling Intervals** for use with **Samplers**. See
:doc:`../../training/sampling/index` to draw training windows.

For standardized evaluation protocols, see :doc:`../../benchmarks/index`.
