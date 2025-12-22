.. _multitask-readout-guide:

Multitask Readout and Modality Registry
=======================================

This guide explains how **torch_brain** supports decoding multiple behavioral or 
stimulus variables simultaneously from neural recordings. The system is built around
two core abstractions:

1. **Modality Registry**: A global registry that defines output modalities (e.g., cursor velocity, visual stimulus class).
2. **MultitaskReadout**: A neural network module that routes model outputs to task-specific projection heads.

These components work together to enable a single model to learn multiple decoding tasks
while sharing neural representations.

--------------

Core Concepts
-------------

What is a Modality?
~~~~~~~~~~~~~~~~~~~

In **torch_brain**, a **modality** represents a specific type of output variable that the model
can predict from neural activity. Each modality is defined by:

- **Name**: A unique string identifier (e.g., ``"cursor_velocity_2d"``)
- **Dimension**: The output dimensionality (e.g., ``2`` for x/y velocity)
- **Data Type**: Whether the variable is continuous, binary, or categorical
- **Loss Function**: The appropriate loss for training (e.g., MSE for continuous, CrossEntropy for categorical)
- **Data Keys**: Where to find timestamps and values in the data object. This assumes that the data object is a :obj:`temporaldata.Data` object.

Examples of modalities include:

- **Continuous variables**: cursor velocity, hand position, running speed, eye position
- **Categorical variables**: stimulus orientation class, image frame ID, trial type, sleep stage, etc.

The Multitask Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~torch_brain.nn.MultitaskReadout` module enables a single transformer encoder
to support multiple output tasks. Here's how it works:

.. code-block:: text

    Neural Input → [Encoder] → Shared Latent Representation
                                          ↓
                               ┌──────────┴──────────┐
                               ↓                     ↓
                       [Linear Head 1]        [Linear Head 2]  ...
                               ↓                     ↓
                       cursor_velocity      grating_orientation
                          (dim=2)               (dim=8)

Each registered modality gets its own linear projection layer, allowing the model to
output predictions of different dimensionalities while sharing the learned neural
representations.

--------------

The Modality Registry
---------------------

Registering a New Modality
~~~~~~~~~~~~~~~~~~~~~~~~~~

To register a custom modality, use the :func:`~torch_brain.register_modality` function:

.. code-block:: python

    import torch_brain
    from torch_brain.registry import DataType

    torch_brain.register_modality(
        "my_custom_modality",
        dim=3,                                          # Output dimension
        type=DataType.CONTINUOUS,                       # Data type
        loss_fn=torch_brain.nn.loss.MSELoss(),          # Loss function
        timestamp_key="behavior.timestamps",            # Where to find timestamps
        value_key="behavior.my_variable",               # Where to find target values
    )

.. important::

    Modalities must be registered **before** creating models that use them. Registration
    typically happens at module import time.

Built-in Modalities
~~~~~~~~~~~~~~~~~~~

**torch_brain** comes with several pre-registered modalities for common neuroscience tasks:

**Motor Decoding (Continuous)**

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Dim
     - Description
   * - ``cursor_velocity_2d``
     - 2
     - Cursor/hand velocity (x, y) for BCI tasks
   * - ``cursor_position_2d``
     - 2
     - Cursor/hand position (x, y)
   * - ``arm_velocity_2d``
     - 2
     - Arm velocity from motor cortex recordings

**Visual Stimulus Classification (Categorical)**

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Dim
     - Description
   * - ``drifting_gratings_orientation``
     - 8
     - Orientation of drifting gratings stimulus
   * - ``drifting_gratings_temporal_frequency``
     - 5
     - Temporal frequency class
   * - ``static_gratings_orientation``
     - 6
     - Orientation of static gratings
   * - ``natural_scenes``
     - 119
     - Natural image classification
   * - ``natural_movie_one_frame``
     - 900
     - Frame prediction for natural movie 1

**Behavioral Variables (Continuous)**

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Dim
     - Description
   * - ``running_speed``
     - 1
     - Animal running speed
   * - ``gaze_pos_2d``
     - 2
     - Eye/gaze position
   * - ``pupil_location``
     - 2
     - Pupil center location
   * - ``pupil_size_2d``
     - 2
     - Pupil size (width, height)

Data Types
~~~~~~~~~~

The :class:`~torch_brain.registry.DataType` enum specifies how the output should be interpreted:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - DataType
     - Target Format
     - Typical Loss
   * - ``CONTINUOUS``
     - Float tensor ``(batch, dim)``
     - ``MSELoss``
   * - ``BINARY``
     - Long tensor ``(batch,)`` with values 0 or 1
     - ``CrossEntropyLoss`` with ``dim=2``
   * - ``MULTINOMIAL``
     - Long tensor ``(batch,)`` with class indices
     - ``CrossEntropyLoss``
   * - ``MULTILABEL``
     - Float tensor ``(batch, num_labels)`` with 0/1 values
     - ``BCEWithLogitsLoss``

The ``dim`` parameter in the modality registration should match the output dimension of
the linear projection (number of classes for classification, or output dimensions for
regression).

The ``ModalitySpec`` Dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each registered modality is stored as a :class:`~torch_brain.registry.ModalitySpec` object:

.. code-block:: python

    from torch_brain.registry import ModalitySpec

    # Access a registered modality
    spec = torch_brain.MODALITY_REGISTRY["cursor_velocity_2d"]

    print(spec.id)             # Unique numeric ID (auto-assigned)
    print(spec.dim)            # Output dimension: 2
    print(spec.type)           # DataType.CONTINUOUS
    print(spec.loss_fn)        # MSELoss instance
    print(spec.timestamp_key)  # "cursor.timestamps"
    print(spec.value_key)      # "cursor.vel"

--------------

MultitaskReadout Module
-----------------------

The :class:`~torch_brain.nn.MultitaskReadout` module creates task-specific linear projection
layers based on the registered modalities.

Creating a MultitaskReadout
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_brain.nn import MultitaskReadout
    from torch_brain.registry import MODALITY_REGISTRY

    # Create readout with all registered modalities
    readout = MultitaskReadout(
        dim=512,                            # Input dimension (from encoder)
        readout_specs=MODALITY_REGISTRY,    # Dictionary of ModalitySpec objects
    )

    # Or with a subset of modalities
    my_tasks = {
        "cursor_velocity_2d": MODALITY_REGISTRY["cursor_velocity_2d"],
        "running_speed": MODALITY_REGISTRY["running_speed"],
    }
    readout = MultitaskReadout(dim=512, readout_specs=my_tasks)

Forward Pass
~~~~~~~~~~~~

The forward pass routes each output token to the appropriate task-specific head:

.. code-block:: python

    import torch

    batch_size, n_outputs, dim = 4, 100, 512

    # Encoder outputs
    output_embs = torch.randn(batch_size, n_outputs, dim)

    # Index specifying which readout head each token uses
    # (matches the .id field of each ModalitySpec)
    output_readout_index = torch.randint(1, 3, (batch_size, n_outputs))

    # Forward pass
    predictions = readout(
        output_embs=output_embs,
        output_readout_index=output_readout_index,
        unpack_output=False,  # Returns dict organized by task
    )

    # predictions is a dict: {"cursor_velocity_2d": Tensor, "running_speed": Tensor, ...}

The ``unpack_output`` Parameter
"""""""""""""""""""""""""""""""

- ``unpack_output=False`` (default): Returns a single dict where predictions for each task
  are **concatenated across all samples** in the batch. Use this during training when
  computing a single loss over the entire batch.

  .. code-block:: python

      # Shape: {"task_name": (total_tokens_for_task, task_dim)}
      predictions = readout(..., unpack_output=False)

- ``unpack_output=True``: Returns a **list of dicts**, one per sample in the batch. Use this
  during validation when you need to track which predictions belong to which sample
  (e.g., for stitching overlapping windows).

  .. code-block:: python

      # Shape: [{"task_name": (tokens_in_sample, task_dim)}, ...]
      predictions = readout(..., unpack_output=True)
      predictions[0]["cursor_velocity_2d"]  # Predictions for first sample

Missing Tasks in a Batch
""""""""""""""""""""""""

If a task has no tokens in a batch (no output indices match that task's ID), the task
is simply skipped and won't appear in the output dict. Your training loop should handle
this by only iterating over the keys present in the output:

.. code-block:: python

    for task_name in output_values.keys():  # Only tasks present in this batch
        # ...

Understanding ``output_readout_index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``output_readout_index`` tensor (called ``output_decoder_index`` in model inputs) tells
the readout module which task each output token belongs to. Each value corresponds to
the ``.id`` attribute of a registered modality:

.. code-block:: python

    # If cursor_velocity_2d has id=1 and running_speed has id=15:
    output_readout_index = torch.tensor([
        [1, 1, 1, 15, 15, 1, ...],  # Sample 1: mix of velocity and speed predictions
        [1, 1, 15, 15, 15, 1, ...],  # Sample 2
        ...
    ])

The readout module uses boolean masking to efficiently route tokens:

1. For each task, create a mask: ``mask = output_readout_index == task_spec.id``
2. Apply the task-specific linear layer: ``task_output = projection[task_name](output_embs[mask])``
3. Collect outputs into the result dictionary

This same index is also used by the model's **task embedding layer** (``self.task_emb``) to
add task-specific information to the output queries before decoding. This allows the
transformer to produce different representations depending on which task is being decoded.

Variable-Length Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~

For memory-efficient processing of variable-length sequences (avoiding padding), use the
``forward_varlen`` method. This expects sequences to be **chained** (concatenated) rather
than padded:

.. code-block:: python

    # Chained sequences: all samples concatenated into one dimension
    output_embs = torch.randn(total_tokens, dim)  # Not (batch, seq_len, dim)
    output_readout_index = torch.tensor([1, 1, 15, 15, 1, ...])  # (total_tokens,)
    output_batch_index = torch.tensor([0, 0, 0, 1, 1, ...])  # Which sample each token belongs to

    predictions = readout.forward_varlen(
        output_embs=output_embs,
        output_readout_index=output_readout_index,
        output_batch_index=output_batch_index,
        unpack_output=True,
    )

--------------

Dataset Configuration
---------------------

The dataset configuration file defines which modalities to decode and how to preprocess
the target values. This is specified in YAML format.

Data Object Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~torch_brain.nn.prepare_for_multitask_readout` function expects your
:obj:`temporaldata.Data` object to have:

1. A ``config`` attribute containing the ``multitask_readout`` list
2. The timestamp and value arrays accessible via the keys specified in the modality
   (or overridden in the config)

Example data structure:

.. code-block:: python

    from temporaldata import Data

    data = Data(
        # Behavioral data
        cursor=Data(
            timestamps=np.array([0.0, 0.01, 0.02, ...]),  # Shape: (n_samples,)
            vel=np.array([[1.0, 2.0], [1.1, 2.1], ...]),  # Shape: (n_samples, 2)
        ),
        # Config specifying what to decode
        config={
            "multitask_readout": [
                {"readout_id": "cursor_velocity_2d", ...}
            ]
        },
    )

    # The keys "cursor.timestamps" and "cursor.vel" are resolved via
    # data.get_nested_attribute("cursor.timestamps")

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - selection:
      - brainset: my_dataset
        sessions:
          - session_001
          - session_002
      config:
        multitask_readout:
          - readout_id: cursor_velocity_2d
            metrics:
              - metric:
                  _target_: torchmetrics.R2Score

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Each entry in ``multitask_readout`` supports these fields:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Required
     - Description
   * - ``readout_id``
     - Yes
     - Name of the registered modality (must exist in ``MODALITY_REGISTRY``)
   * - ``timestamp_key``
     - No
     - Override the modality's default timestamp location
   * - ``value_key``
     - No
     - Override the modality's default value location
   * - ``normalize_mean``
     - No
     - Mean for z-score normalization (scalar or list for per-channel)
   * - ``normalize_std``
     - No
     - Standard deviation for z-score normalization (scalar or list)
   * - ``weights``
     - No
     - Dict mapping interval keys to weight values (see below)
   * - ``metrics``
     - No
     - List of torchmetrics for evaluation (see below)
   * - ``eval_interval``
     - No
     - Key to an ``Interval`` object; only timestamps within this interval are used for metric computation

Normalization
"""""""""""""

Target values are z-score normalized **before** being passed to the loss function. This is 
applied during tokenization:

.. code-block:: python

    normalized_values = (raw_values - normalize_mean) / normalize_std

For multi-dimensional outputs (e.g., x/y coordinates), use lists:

.. code-block:: yaml

    normalize_mean: [0.0, 0.0]      # Per-channel means
    normalize_std: [100.0, 100.0]   # Per-channel stds

.. note::

    Metrics are computed on the normalized predictions and targets. If you need metrics
    in original units, you must denormalize the outputs manually.

Sample Weights
""""""""""""""

The ``weights`` field allows weighting different time periods differently in the loss.
It maps interval keys (from your data object) to weight values:

.. code-block:: yaml

    weights:
      movement_periods.reach_period: 5.0    # Weight reach periods 5x more
      movement_periods.hold_period: 0.1     # Down-weight hold periods
      cursor_outlier_segments: 0.0          # Ignore outlier segments

Weights are multiplicative when intervals overlap. Timestamps outside all specified
intervals get a default weight of 1.0.

Evaluation Interval
"""""""""""""""""""

The ``eval_interval`` field specifies which timestamps to include when computing metrics.
This is useful when you only want to evaluate on specific trial phases:

.. code-block:: yaml

    eval_interval: nlb_eval_intervals  # Key to an Interval in your data object

Timestamps outside this interval are excluded from metric computation but still
contribute to training loss.

Metrics
"""""""

The ``metrics`` field specifies which torchmetrics to compute during validation/test.
These metrics are **not** computed in the training loop—they are handled by the
:class:`~torch_brain.utils.stitcher.MultiTaskDecodingStitchEvaluator` callback.

.. code-block:: yaml

    metrics:
      - metric:
          _target_: torchmetrics.R2Score
      - metric:
          _target_: torchmetrics.MeanSquaredError

For classification tasks, specify the task type and number of classes:

.. code-block:: yaml

    metrics:
      - metric:
          _target_: torchmetrics.Accuracy
          task: multiclass
          num_classes: 8

See :ref:`evaluation-stitching` for how these metrics are computed.


Overriding Default Keys
~~~~~~~~~~~~~~~~~~~~~~~

Sometimes the data keys in your dataset differ from the modality defaults. Override them
in the config:

.. code-block:: yaml

    multitask_readout:
      - readout_id: cursor_velocity_2d
        # Default uses cursor.timestamps and cursor.vel
        # Override for datasets with different naming
        timestamp_key: hand.timestamps
        value_key: hand.vel
        normalize_mean: 0.0
        normalize_std: 100.0

--------------

Data Preparation Pipeline
-------------------------

The :func:`~torch_brain.nn.prepare_for_multitask_readout` function transforms raw data
into the format expected by the model during tokenization.

How It Works
~~~~~~~~~~~~

.. code-block:: python

    from torch_brain.nn import prepare_for_multitask_readout

    # Called during model.tokenize()
    (
        output_timestamps,   # Concatenated timestamps for all tasks
        output_values,       # Dict mapping task names to target values
        output_task_index,   # Task ID for each timestamp
        output_weights,      # Dict mapping task names to sample weights
        output_eval_mask,    # Dict mapping task names to evaluation masks
    ) = prepare_for_multitask_readout(data, readout_registry)

The function:

1. Iterates through each readout config in ``data.config["multitask_readout"]``
2. Validates that the ``readout_id`` exists in the registry
3. Extracts timestamps and values using the specified (or default) keys
4. Applies normalization if specified
5. Resolves sample weights and evaluation masks
6. Chains all timestamps together and creates the ``output_task_index`` tensor

Integration with Models
~~~~~~~~~~~~~~~~~~~~~~~

Models like :class:`~torch_brain.models.POYOPlus` call this function in their ``tokenize()`` method:

.. code-block:: python

    class POYOPlus(nn.Module):
        def tokenize(self, data):
            # ... prepare inputs and latents ...

            # Prepare outputs for all tasks
            (
                output_timestamps,
                output_values,
                output_task_index,
                output_weights,
                output_eval_mask,
            ) = prepare_for_multitask_readout(
                data,
                self.readout_specs,  # The modality registry subset
            )

            # Pack into data_dict for collation
            return {
                "model_inputs": {
                    # ... input tensors ...
                    "output_timestamps": output_timestamps,
                    "output_decoder_index": output_task_index,
                },
                "target_values": output_values,
                "target_weights": output_weights,
                # ...
            }

After tokenization, the data is batched using :func:`~torch_brain.data.collate`. The collate
function pads sequences to equal length within a batch. The ``target_values`` and
``target_weights`` dicts are collated such that predictions and targets can be matched
by task name in the training loop.

--------------

Training Loop Integration
-------------------------

Here's how the components work together during training:

Computing the Loss
~~~~~~~~~~~~~~~~~~

The loss computation has two levels of weighting:

1. **Sample weights** (from the ``weights`` config): Per-timestamp weights within a task
2. **Task balancing**: Tasks are weighted by how many batch samples contain that task

.. code-block:: python

    def training_step(self, batch, batch_idx):
        # Forward pass through the model
        output_values = self.model(**batch["model_inputs"], unpack_output=False)

        # output_values is a dict: {"task_name": predictions_tensor, ...}
        # Note: predictions for each task are concatenated across all batch samples

        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        total_loss = 0
        for task_name in output_values.keys():
            predictions = output_values[task_name]
            targets = target_values[task_name]

            # Sample weights (from config, e.g., down-weighting hold periods)
            # None means uniform weighting
            weights = target_weights.get(task_name, None)

            # Loss function from the modality spec handles the weighting internally
            spec = self.model.readout.readout_specs[task_name]
            task_loss = spec.loss_fn(predictions, targets, weights)

            # Task balancing: weight by number of batch samples containing this task
            # This ensures tasks appearing in fewer samples aren't dominated
            num_samples_with_task = torch.any(
                batch["model_inputs"]["output_decoder_index"] == spec.id,
                dim=1
            ).sum()
            total_loss += task_loss * num_samples_with_task

        return total_loss / batch_size

The built-in loss functions (``MSELoss``, ``CrossEntropyLoss``) handle sample weights
by computing a weighted average: ``(weights * loss).sum() / weights.sum()``.

Filtering by Task
~~~~~~~~~~~~~~~~~

The registry provides a convenient way to get task-specific model outputs:

.. code-block:: python

    from torch_brain.registry import MODALITY_REGISTRY

    # Get the numeric ID for a task
    velocity_id = MODALITY_REGISTRY["cursor_velocity_2d"].id

    # Find which outputs correspond to this task
    decoder_index = batch["model_inputs"]["output_decoder_index"]
    velocity_mask = decoder_index == velocity_id

    # Get predictions for this task only
    velocity_predictions = output_values["cursor_velocity_2d"]

--------------

.. _evaluation-stitching:

Evaluation and Stitching
------------------------

When using overlapping sliding windows for validation/test (which improves metric stability),
the same timestamp may receive predictions from multiple windows. The
:class:`~torch_brain.utils.stitcher.MultiTaskDecodingStitchEvaluator` callback handles
**stitching** these overlapping predictions before computing metrics.

Why Stitching is Needed
~~~~~~~~~~~~~~~~~~~~~~~

Consider a 2-second recording evaluated with 1-second windows and 0.5-second steps:

.. code-block:: text

    Recording:    |-------- 0.0s to 2.0s --------|
    Window 1:     |---- 0.0s to 1.0s ----|
    Window 2:          |---- 0.5s to 1.5s ----|
    Window 3:               |---- 1.0s to 2.0s ----|

Timestamps between 0.5s and 1.5s receive predictions from multiple windows. Stitching
combines these predictions before computing the final metric.

How Stitching Works
~~~~~~~~~~~~~~~~~~~

The :func:`~torch_brain.utils.stitcher.stitch` function pools predictions by timestamp:

- **Continuous outputs** (float): Mean pooling across overlapping predictions
- **Categorical outputs** (long): Mode pooling (majority vote) across overlapping predictions

.. code-block:: python

    from torch_brain.utils.stitcher import stitch

    # Multiple predictions for the same timestamps
    timestamps = torch.tensor([0.5, 0.5, 0.6, 0.6, 0.6])
    predictions = torch.tensor([[1.0, 2.0], [1.2, 1.8], [0.9, 2.1], [1.1, 1.9], [1.0, 2.0]])

    unique_timestamps, stitched = stitch(timestamps, predictions)
    # unique_timestamps: [0.5, 0.6]
    # stitched: mean of predictions at each timestamp

MultiTaskDecodingStitchEvaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This Lightning callback orchestrates the entire evaluation pipeline:

**Setup** (called by the DataModule):

.. code-block:: python

    from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator

    # metrics dict structure: {session_id: {task_name: {metric_name: metric_instance}}}
    evaluator = MultiTaskDecodingStitchEvaluator(metrics=data_module.get_metrics())

**Validation step** (your code returns data for the evaluator):

.. code-block:: python

    from torch_brain.utils.stitcher import DataForMultiTaskDecodingStitchEvaluator

    def validation_step(self, batch, batch_idx):
        # Forward pass with unpack_output=True to get per-sample predictions
        output_values = self.model(**batch["model_inputs"], unpack_output=True)

        # Package data for the evaluator
        return DataForMultiTaskDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,                                    # List of dicts
            targets=batch["target_values"],                         # Dict of tensors
            decoder_indices=batch["model_inputs"]["output_decoder_index"],
            eval_masks=batch["eval_mask"],                          # Dict of masks
            session_ids=batch["session_id"],                        # List of strings
            absolute_starts=batch["absolute_start"],                # For timestamp alignment
        )

**What the evaluator does internally:**

1. **Caches predictions**: Accumulates predictions, targets, and timestamps for each
   recording sequence
2. **Applies eval_mask**: Filters out timestamps outside the evaluation interval
3. **Adds absolute_start**: Converts relative window timestamps to absolute recording time
4. **Stitches when ready**: Once all windows for a sequence are processed, stitches the
   predictions using mean (continuous) or mode (categorical) pooling
5. **Updates metrics**: Passes stitched predictions and targets to the torchmetrics instances
6. **Logs results**: At epoch end, computes and logs metrics per session/task combination

The ``DataForMultiTaskDecodingStitchEvaluator`` Dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - ``timestamps``
     - Output timestamps, shape ``(batch, n_out)``. Relative to window start.
   * - ``preds``
     - List of dicts (one per batch sample) from ``model(..., unpack_output=True)``
   * - ``targets``
     - Dict mapping task names to target tensors (from ``batch["target_values"]``)
   * - ``decoder_indices``
     - Task IDs for each output token, shape ``(batch, n_out)``
   * - ``eval_masks``
     - Dict mapping task names to boolean masks for evaluation filtering
   * - ``session_ids``
     - List of session ID strings (one per batch sample)
   * - ``absolute_starts``
     - Absolute start time of each window, shape ``(batch,)``

Metric Logging Format
~~~~~~~~~~~~~~~~~~~~~

Metrics are logged with the format ``{session_id}/{task_name}/{metric_name}/{prefix}``:

.. code-block:: text

    session_001/cursor_velocity_2d/R2Score/val: 0.85
    session_001/running_speed/MeanSquaredError/val: 0.12
    session_002/cursor_velocity_2d/R2Score/val: 0.82
    ...
    average_val_metric: 0.78

The ``average_val_metric`` (or ``average_test_metric``) is the mean across all
session/task/metric combinations and is used for model checkpointing.

Distributed Evaluation
~~~~~~~~~~~~~~~~~~~~~~

For multi-GPU training, use :class:`~torch_brain.data.sampler.DistributedStitchingFixedWindowSampler`
to ensure windows from the same recording sequence stay on the same GPU. This is required
because stitching happens per-GPU and windows split across GPUs cannot be stitched together.

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

Complete Example
----------------

Here's a complete example showing how to set up multitask decoding:

1. Register Custom Modalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # my_modalities.py
    import torch_brain
    from torch_brain.registry import DataType

    # Register at import time
    torch_brain.register_modality(
        "joystick_position",
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn=torch_brain.nn.loss.MSELoss(),
        timestamp_key="joystick.timestamps",
        value_key="joystick.position",
    )

    torch_brain.register_modality(
        "target_acquired",
        dim=2,  # binary classification
        type=DataType.BINARY,
        loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
        timestamp_key="events.timestamps",
        value_key="events.target_acquired",
    )

2. Create Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # configs/dataset/my_experiment.yaml
    - selection:
      - brainset: my_experiment
        sessions:
          - monkey_J_day1
          - monkey_J_day2
      config:
        multitask_readout:
          - readout_id: joystick_position
            normalize_mean: [0.0, 0.0]
            normalize_std: [50.0, 50.0]
            metrics:
              - metric:
                  _target_: torchmetrics.R2Score

          - readout_id: target_acquired
            metrics:
              - metric:
                  _target_: torchmetrics.Accuracy
                  task: binary

3. Create and Train Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # train.py
    import my_modalities  # Register custom modalities

    from torch_brain.models import POYOPlus
    from torch_brain.registry import MODALITY_REGISTRY

    # Create model with all registered modalities
    model = POYOPlus(
        sequence_length=1.0,
        readout_specs=MODALITY_REGISTRY,
        latent_step=0.05,
        dim=256,
        depth=4,
    )

    # Or filter to only the modalities you need
    my_tasks = {
        k: v for k, v in MODALITY_REGISTRY.items()
        if k in ["joystick_position", "target_acquired"]
    }
    model = POYOPlus(
        sequence_length=1.0,
        readout_specs=my_tasks,
        # ...
    )

Understanding the ID System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each modality gets a unique numeric ID assigned at registration time:

.. code-block:: python

    # IDs are assigned sequentially starting from 1
    id1 = torch_brain.register_modality("task_a", ...)  # id1 = 1
    id2 = torch_brain.register_modality("task_b", ...)  # id2 = 2
    id3 = torch_brain.register_modality("task_c", ...)  # id3 = 3

These IDs are used for:

- Fast tensor-based routing in ``MultitaskReadout.forward()``
- Embedding lookups for task embeddings in the model (``self.task_emb(decoder_index)``)
- Efficient masking operations during training

You can look up a modality name from its ID using:

.. code-block:: python

    modality_name = torch_brain.get_modality_by_id(1)  # Returns "task_a"

.. warning::

    IDs depend on registration order. The built-in modalities in ``torch_brain.registry``
    are registered when the module is imported. If you register custom modalities, do so
    **before** creating models to ensure consistent ID assignment across runs.

--------------

API Reference
-------------

.. seealso::

    **Registry**

    - :class:`torch_brain.registry.ModalitySpec` - Modality specification dataclass
    - :class:`torch_brain.registry.DataType` - Data type enumeration
    - :func:`torch_brain.register_modality` - Registration function

    **Readout**

    - :class:`torch_brain.nn.MultitaskReadout` - Multi-task readout module
    - :func:`torch_brain.nn.prepare_for_multitask_readout` - Data preparation function

    **Evaluation**

    - :class:`torch_brain.utils.stitcher.MultiTaskDecodingStitchEvaluator` - Evaluation callback
    - :class:`torch_brain.utils.stitcher.DataForMultiTaskDecodingStitchEvaluator` - Data container for evaluator
    - :func:`torch_brain.utils.stitcher.stitch` - Timestamp-based prediction pooling

