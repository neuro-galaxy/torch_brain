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

.. contents:: Table of Contents
   :local:
   :depth: 3

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

.. code-block:: python

    from torch_brain.registry import DataType

    DataType.CONTINUOUS   # For regression tasks (uses MSELoss)
    DataType.BINARY       # For binary classification
    DataType.MULTINOMIAL  # For multi-class classification (uses CrossEntropyLoss)
    DataType.MULTILABEL   # For multi-label classification

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

Understanding ``output_readout_index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``output_readout_index`` tensor tells the readout module which task each output
token belongs to. Each value corresponds to the ``.id`` attribute of a registered modality:

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

--------------

Dataset Configuration
---------------------

The dataset configuration file defines which modalities to decode and how to preprocess
the target values. This is specified in YAML format.

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
     - Name of the registered modality
   * - ``timestamp_key``
     - No
     - Override the modality's default timestamp location
   * - ``value_key``
     - No
     - Override the modality's default value location
   * - ``normalize_mean``
     - No
     - Mean for z-score normalization (scalar or list)
   * - ``normalize_std``
     - No
     - Standard deviation for z-score normalization
   * - ``weights``
     - No
     - Configuration for sample weighting
   * - ``metrics``
     - No
     - Evaluation metrics (using torchmetrics)
   * - ``eval_interval``
     - No
     - Key to interval data for evaluation masking


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

--------------

Training Loop Integration
-------------------------

Here's how the components work together during training:

Computing the Loss
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def training_step(self, batch, batch_idx):
        # Forward pass through the model
        output_values = self.model(**batch["model_inputs"], unpack_output=False)

        # output_values is a dict: {"task_name": predictions_tensor, ...}

        # Compute task-wise losses
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        total_loss = 0
        for task_name in output_values.keys():
            predictions = output_values[task_name]
            targets = target_values[task_name]
            weights = target_weights.get(task_name, None)

            # Get the loss function from the modality spec
            spec = self.model.readout.readout_specs[task_name]
            task_loss = spec.loss_fn(predictions, targets, weights)

            # Weight by number of samples with this task
            num_samples = torch.any(
                batch["model_inputs"]["output_decoder_index"] == spec.id,
                dim=1
            ).sum()
            total_loss += task_loss * num_samples

        return total_loss / batch_size

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
- Embedding lookups for task embeddings in the model
- Efficient masking operations during training

.. warning::

    IDs depend on registration order. For reproducibility, ensure modalities are always
    registered in the same order (e.g., by importing modules consistently).

--------------

API Reference
-------------

.. seealso::

    - :class:`torch_brain.registry.ModalitySpec` - Modality specification dataclass
    - :class:`torch_brain.registry.DataType` - Data type enumeration
    - :func:`torch_brain.register_modality` - Registration function
    - :class:`torch_brain.nn.MultitaskReadout` - Multi-task readout module
    - :func:`torch_brain.nn.prepare_for_multitask_readout` - Data preparation function

