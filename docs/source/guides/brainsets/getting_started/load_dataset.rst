Load a Dataset
==============

Each brainset provides a **Dataset** class in :mod:`torch_brain.datasets`: a
PyTorch-compatible loader that reads Session H5 files from your processed
directory. Calling :meth:`~torch_brain.datasets.Dataset.get_recording` returns a
:class:`~torch_brain.data.Data` object with neural signals, behavior, metadata,
and a temporal **Domain**. For more details on the Data object, see :doc:`../../data/index`.

Example: Perich & Miller (2018)
-------------------------------

Let's load the first session of the Perich & Miller (2018) spiking brainset:

.. code-block:: python

   from torch_brain.datasets import PerichMillerPopulation2018

   dataset = PerichMillerPopulation2018()
   session = dataset.get_recording(dataset.recording_ids[0])
   print(session.spikes.timestamps.shape)

The
`Perich & Miller pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
wrote the Session H5 files under ``<processed_dir>/perich_miller_population_2018/``.

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

:class:`~torch_brain.datasets.NestedDataset` namespaces recording IDs as
``"<DatasetClassName>/<session_id>"``.

Sample windows for training
---------------------------

Each **Dataset** exposes **Sampling Intervals** for use with **Samplers**. See
:doc:`../../training/sampling/index` to draw training windows.
