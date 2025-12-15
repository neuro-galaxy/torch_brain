Data Pipeline
=============

This guide explains how data flows from raw recordings to model inputs in **torch_brain**.

.. contents:: Table of Contents
   :local:
   :depth: 2

--------------

Overview
--------

The data pipeline consists of four main components:

1. **Dataset**: Lazily loads recordings from HDF5 files
2. **Sampler**: Generates indices specifying which time windows to sample
3. **Transform**: Applies augmentations and tokenization to each sample
4. **Collate**: Batches variable-length samples together

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                        Data Pipeline                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Dataset   │ -> │   Sampler   │ -> │  Transform  │ -> │   Collate   │
    │  (HDF5 I/O) │    │  (Windows)  │    │ (Tokenize)  │    │  (Batching) │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                    │
                                    ▼
                            Model Inputs

--------------

Preparing Data with Brainsets
-----------------------------

Before using **torch_brain**, neural recordings must be processed into HDF5 format 
using the `brainsets <https://github.com/neuro-galaxy/brainsets>`_ library.

Each processed recording is stored as a single HDF5 file containing:

- Actual data (e.g., spikes, calcium traces)
- Behavioral variables (e.g., cursor position, velocity)
- Metadata (session info, subject info, etc.)
- Split domains (train/valid/test intervals)

--------------

Dataset
-------

The :class:`~torch_brain.data.Dataset` class provides lazy loading of recordings.

Loading a Single Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_brain.data import Dataset

    dataset = Dataset(
        root="./processed",
        recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
        split="train"
    )

Loading Multiple Recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a YAML configuration file to load multiple recordings:

.. code-block:: yaml
    :caption: config.yaml

    - selection:
      - brainset: perich_miller_population_2018
        sessions:
          - c_20131003_center_out_reaching
          - c_20131022_center_out_reaching
      config:
        multitask_readout:
          - readout_id: cursor_velocity_2d
            normalize_mean: [0.0, 0.0]
            normalize_std: [100.0, 100.0]

.. code-block:: python

    dataset = Dataset(
        root="./processed",
        config="config.yaml",
        split="train"
    )

Key Methods
~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - ``get_sampling_intervals()``
     - Returns intervals from which samplers can draw windows
   * - ``get_unit_ids()``
     - Returns all unit IDs across recordings
   * - ``get_session_ids()``
     - Returns all session IDs
   * - ``get(recording_id, start, end)``
     - Retrieves a time slice from a recording

--------------

Samplers
--------

Samplers generate :class:`~torch_brain.data.DatasetIndex` objects that specify 
which recording and time window to load.

Available Samplers
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sampler
     - Use Case
   * - :class:`~torch_brain.data.sampler.RandomFixedWindowSampler`
     - Training: randomly samples fixed-length windows with jitter
   * - :class:`~torch_brain.data.sampler.SequentialFixedWindowSampler`
     - Evaluation: sequentially samples overlapping windows
   * - :class:`~torch_brain.data.sampler.TrialSampler`
     - Trial-based: samples complete trial intervals

Example
~~~~~~~

.. code-block:: python

    from torch_brain.data.sampler import RandomFixedWindowSampler

    sampler = RandomFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=1.0,  # 1 second windows
    )

    for index in sampler:
        # index.recording_id, index.start, index.end
        sample = dataset[index]

See :doc:`sampler` for a detailed guide on sampling strategies.

--------------

Transforms
----------

Transforms modify data samples before they are batched. They are applied 
when accessing ``dataset[index]`` via the ``transform`` parameter.

Compose
~~~~~~~

Use :class:`~torch_brain.transforms.Compose` to chain multiple transforms:

.. code-block:: python

    from torch_brain.transforms import Compose, UnitDropout, RandomTimeScaling

    transform = Compose([
        UnitDropout(min_units=20, mode_units=100),  # Augmentation
        RandomTimeScaling(min_scale=0.9, max_scale=1.1),  # Augmentation
        model.tokenize,  # Tokenization (must be last)
    ])

    dataset = Dataset(
        root="./processed",
        recording_id="...",
        split="train",
        transform=transform,
    )

Available Transforms
~~~~~~~~~~~~~~~~~~~~

**Augmentations:**

- :class:`~torch_brain.transforms.UnitDropout`: Randomly drops units
- :class:`~torch_brain.transforms.RandomTimeScaling`: Scales the time axis
- :class:`~torch_brain.transforms.RandomCrop`: Crops a random time window
- :class:`~torch_brain.transforms.RandomOutputSampler`: Subsamples output tokens

**Filtering:**

- :class:`~torch_brain.transforms.UnitFilter`: Filters units by custom criteria
- :class:`~torch_brain.transforms.UnitFilterById`: Filters units by ID pattern

**Control Flow:**

- :class:`~torch_brain.transforms.RandomChoice`: Randomly selects one transform
- :class:`~torch_brain.transforms.ConditionalChoice`: Conditionally applies transforms

Tokenization
~~~~~~~~~~~~

The model's ``tokenize`` method converts a :obj:`temporaldata.Data` object into 
tensors ready for the model. This should always be the last transform:

.. code-block:: python

    data_dict = model.tokenize(data)
    # Returns:
    # {
    #     "model_inputs": {...},  # Input tensors for model.forward()
    #     "target_values": {...},  # Ground truth targets
    #     "target_weights": {...},  # Sample weights
    #     ...
    # }

--------------

Collate
-------

The :func:`~torch_brain.data.collate.collate` function batches samples of variable 
length. It extends PyTorch's default collate with two strategies:

Padding (``pad``, ``pad8``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pads sequences to the maximum length in the batch:

.. code-block:: python

    from torch_brain.data import pad, pad8, track_mask, track_mask8

    # In tokenize():
    data_dict = {
        "input_timestamps": pad8(timestamps),  # Pad to multiple of 8
        "input_mask": track_mask8(timestamps),  # Track padding mask
    }

Chaining (``chain``)
~~~~~~~~~~~~~~~~~~~~

Concatenates sequences along the first dimension (like PyTorch Geometric):

.. code-block:: python

    from torch_brain.data import chain, track_batch

    data_dict = {
        "target_values": chain(values),  # Concatenate
        "batch_index": track_batch(values),  # Track batch membership
    }

Usage
~~~~~

Pass ``collate`` to the DataLoader:

.. code-block:: python

    from torch.utils.data import DataLoader
    from torch_brain.data import collate

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        collate_fn=collate,
    )

--------------

Complete Example
----------------

.. code-block:: python

    from torch.utils.data import DataLoader
    from torch_brain.data import Dataset, collate
    from torch_brain.data.sampler import RandomFixedWindowSampler
    from torch_brain.transforms import Compose, UnitDropout
    from torch_brain.models import POYOPlus

    # 1. Create model
    model = POYOPlus(sequence_length=1.0, latent_step=0.05, dim=256, depth=4)

    # 2. Create transform pipeline
    transform = Compose([
        UnitDropout(min_units=20, mode_units=100),
        model.tokenize,
    ])

    # 3. Create dataset with transform
    dataset = Dataset(
        root="./processed",
        config="config.yaml",
        split="train",
        transform=transform,
    )

    # 4. Initialize model vocabularies
    model.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model.session_emb.initialize_vocab(dataset.get_session_ids())

    # 5. Create sampler and dataloader
    sampler = RandomFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=1.0,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        collate_fn=collate,
        num_workers=4,
    )

    # 6. Iterate
    for batch in dataloader:
        outputs = model(**batch["model_inputs"])
        # ...

