Getting Started
===============

This guide provides a quick introduction to **torch_brain** after installation.

Verifying Installation
----------------------

After installing the package, verify the installation by importing the library:

.. code-block:: python

    >>> import torch_brain
    >>> print(torch_brain.__name__)
    torch_brain

Loading Data
------------

**torch_brain** provides the :class:`~torch_brain.data.Dataset` class for loading neural 
recordings. The dataset lazily loads data from HDF5 files processed by 
`brainsets <https://github.com/neuro-galaxy/brainsets>`_.

.. code-block:: python

    from torch_brain.data import Dataset

    # Load a single recording
    dataset = Dataset(
        root="./processed",
        recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
        split="train"
    )

    # Or load multiple recordings via a config file
    dataset = Dataset(
        root="./processed",
        config="config.yaml",
        split="train"
    )

Instantiating a Model
---------------------

**torch_brain** provides pre-built neural decoding models. Here's how to instantiate 
the POYOPlus model:

.. code-block:: python

    from torch_brain.models import POYOPlus
    from torch_brain.registry import MODALITY_REGISTRY

    model = POYOPlus(
        sequence_length=1.0,
        readout_specs=MODALITY_REGISTRY,
        latent_step=0.05,
        dim=256,
        depth=4,
    )

    # Initialize unit and session vocabularies from your dataset
    model.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model.session_emb.initialize_vocab(dataset.get_session_ids())

Basic Training Loop
-------------------

A minimal training setup uses samplers to generate windows from the data:

.. code-block:: python

    from torch.utils.data import DataLoader
    from torch_brain.data.sampler import RandomFixedWindowSampler
    from torch_brain.data import collate

    # Create a sampler
    sampler = RandomFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=1.0,
    )

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        collate_fn=collate,
    )

    # Training loop
    for batch in dataloader:
        outputs = model(**batch["model_inputs"])
        # compute loss and backpropagate...

Next Steps
----------

- :doc:`sampler` - Learn about advanced sampling strategies
- :doc:`multitask_readout` - Understand multi-task decoding and modality registration
- :doc:`data_pipeline` - Deep dive into the data loading pipeline
- :doc:`training` - Complete guide to model training with Lightning

