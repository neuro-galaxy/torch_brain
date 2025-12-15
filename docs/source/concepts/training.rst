Model Training
==============

This guide covers training **torch_brain** models using PyTorch Lightning.

.. contents:: Table of Contents
   :local:
   :depth: 2

--------------

Overview
--------

Training a neural decoding model involves:

1. Setting up a Lightning module with loss computation
2. Configuring evaluation callbacks for stitching predictions
3. Handling distributed training considerations

--------------

Lightning Module Setup
----------------------

Create a Lightning module that wraps your model:

.. code-block:: python

    import lightning as L
    import torch
    from torch_brain.models import POYOPlus
    from torch_brain.optim import SparseLamb

    class NeuralDecoder(L.LightningModule):
        def __init__(self, model_config, lr=1e-3):
            super().__init__()
            self.save_hyperparameters()
            self.model = POYOPlus(**model_config)

        def forward(self, **kwargs):
            return self.model(**kwargs)

        def training_step(self, batch, batch_idx):
            output_values = self.model(**batch["model_inputs"], unpack_output=False)
            
            loss = self.compute_loss(
                output_values,
                batch["target_values"],
                batch["target_weights"],
            )
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return SparseLamb(self.parameters(), lr=self.hparams.lr)

--------------

Loss Computation
----------------

For multi-task models, loss is computed per task and aggregated:

.. code-block:: python

    def compute_loss(self, output_values, target_values, target_weights):
        total_loss = 0.0
        batch_size = ...  # Typically from batch metadata

        for task_name in output_values.keys():
            predictions = output_values[task_name]
            targets = target_values[task_name]
            weights = target_weights.get(task_name, None)

            # Get the loss function from the modality spec
            spec = self.model.readout.readout_specs[task_name]
            task_loss = spec.loss_fn(predictions, targets, weights)

            # Weight by number of samples containing this task
            num_samples = torch.any(
                batch["model_inputs"]["output_decoder_index"] == spec.id,
                dim=1
            ).sum()
            
            total_loss += task_loss * num_samples

        return total_loss / batch_size

Sample Weights
~~~~~~~~~~~~~~

Sample weights allow weighting different time periods differently:

.. code-block:: yaml
    :caption: config.yaml

    config:
      multitask_readout:
        - readout_id: cursor_velocity_2d
          weights:
            movement_periods.reach_period: 5.0  # Weight reach 5x more
            movement_periods.hold_period: 0.1   # Down-weight hold

Weights are multiplicative when intervals overlap.

--------------

Evaluation Callbacks
--------------------

Use evaluation callbacks to stitch predictions from overlapping windows before 
computing metrics.

Single-Task Evaluation
~~~~~~~~~~~~~~~~~~~~~~

For single-task models (e.g., POYO), use :class:`~torch_brain.utils.stitcher.DecodingStitchEvaluator`:

.. code-block:: python

    from torch_brain.utils.stitcher import (
        DecodingStitchEvaluator,
        DataForDecodingStitchEvaluator,
    )

    # Create the callback
    evaluator = DecodingStitchEvaluator(
        session_ids=dataset.get_session_ids(),
        modality_spec=readout_spec,
    )

    # In your validation_step, return data for the evaluator
    def validation_step(self, batch, batch_idx):
        output = self.model(**batch["model_inputs"], unpack_output=True)
        
        return DataForDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output,
            targets=batch["target_values"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

Multi-Task Evaluation
~~~~~~~~~~~~~~~~~~~~~

For multi-task models (e.g., POYOPlus), use 
:class:`~torch_brain.utils.stitcher.MultiTaskDecodingStitchEvaluator`:

.. code-block:: python

    from torch_brain.utils.stitcher import (
        MultiTaskDecodingStitchEvaluator,
        DataForMultiTaskDecodingStitchEvaluator,
    )

    # Create the callback with metrics per session and task
    evaluator = MultiTaskDecodingStitchEvaluator(
        metrics=datamodule.get_metrics()
    )

    # In your validation_step
    def validation_step(self, batch, batch_idx):
        output = self.model(**batch["model_inputs"], unpack_output=True)
        
        return DataForMultiTaskDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output,
            targets=batch["target_values"],
            decoder_indices=batch["model_inputs"]["output_decoder_index"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

How Stitching Works
~~~~~~~~~~~~~~~~~~~

The :func:`~torch_brain.utils.stitcher.stitch` function pools predictions by timestamp:

- **Continuous outputs**: Mean pooling
- **Categorical outputs**: Mode pooling (majority vote)

This handles overlapping predictions from sliding window evaluation.

--------------

Distributed Training
--------------------

For multi-GPU training, use the distributed sampler wrapper:

Training Sampler
~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_brain.data.sampler import (
        RandomFixedWindowSampler,
        DistributedEvaluationSamplerWrapper,
    )

    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_dataset.get_sampling_intervals(),
        window_length=sequence_length,
    )

    # Wrap for distributed training (Lightning handles this automatically)

Evaluation Sampler
~~~~~~~~~~~~~~~~~~

For evaluation with stitching, use 
:class:`~torch_brain.data.sampler.DistributedStitchingFixedWindowSampler` to keep 
windows from the same sequence on the same GPU:

.. code-block:: python

    from torch_brain.data.sampler import DistributedStitchingFixedWindowSampler

    val_sampler = DistributedStitchingFixedWindowSampler(
        sampling_intervals=val_dataset.get_sampling_intervals(),
        window_length=sequence_length,
        step=sequence_length / 2,  # 50% overlap
        batch_size=batch_size,
        num_replicas=trainer.world_size,
        rank=trainer.global_rank,
    )

--------------

Checkpointing and Model Loading
-------------------------------

Saving Checkpoints
~~~~~~~~~~~~~~~~~~

Lightning automatically saves checkpoints. Configure the checkpoint callback:

.. code-block:: python

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{average_val_metric:.2f}",
        monitor="average_val_metric",
        mode="max",
        save_top_k=3,
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback, evaluator])

Loading Pretrained Models
~~~~~~~~~~~~~~~~~~~~~~~~~

Load a pretrained POYO model:

.. code-block:: python

    from torch_brain.models import POYO

    model = POYO.load_pretrained(
        checkpoint_path="path/to/checkpoint.ckpt",
        readout_spec=my_readout_spec,
        skip_readout=True,  # Initialize new readout layer
    )

Loading from Lightning Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load the full Lightning module
    module = NeuralDecoder.load_from_checkpoint("checkpoint.ckpt")
    model = module.model

    # Or load just the model weights
    checkpoint = torch.load("checkpoint.ckpt")
    state_dict = {
        k.replace("model.", ""): v 
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

--------------

Complete Training Script
------------------------

.. code-block:: python

    import lightning as L
    from torch.utils.data import DataLoader
    from torch_brain.data import Dataset, collate
    from torch_brain.data.sampler import RandomFixedWindowSampler
    from torch_brain.transforms import Compose, UnitDropout
    from torch_brain.models import POYOPlus
    from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator

    # 1. Setup model and transforms
    model = POYOPlus(sequence_length=1.0, latent_step=0.05, dim=256, depth=4)

    transform = Compose([
        UnitDropout(min_units=20, mode_units=100),
        model.tokenize,
    ])

    # 2. Setup datasets
    train_dataset = Dataset("./processed", config="config.yaml", split="train", transform=transform)
    val_dataset = Dataset("./processed", config="config.yaml", split="valid", transform=transform)

    # 3. Initialize vocabularies
    model.unit_emb.initialize_vocab(train_dataset.get_unit_ids())
    model.session_emb.initialize_vocab(train_dataset.get_session_ids())

    # 4. Setup samplers and dataloaders
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_dataset.get_sampling_intervals(),
        window_length=1.0,
    )

    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=32, collate_fn=collate
    )

    # 5. Create Lightning module and trainer
    module = NeuralDecoder(model)
    evaluator = MultiTaskDecodingStitchEvaluator(metrics=...)

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[evaluator],
        accelerator="gpu",
        devices=1,
    )

    # 6. Train
    trainer.fit(module, train_loader, val_loader)

