import pytest
import torch
import numpy as np
from unittest.mock import Mock
from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict

from torch_brain.data import collate
from torch_brain.registry import DataType, register_modality
from torch_brain.models.poyo_plus import POYOPlus, POYOPlusTokenizer


def setup_module():
    """Register test modalities before running tests"""

    register_modality(
        "custom_gaze_pos_2d",
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="gaze.timestamps",
        value_key="gaze.position",
    )


@pytest.fixture
def task_specs():
    from torch_brain.registry import MODALITIY_REGISTRY

    return MODALITIY_REGISTRY


@pytest.fixture
def model(task_specs):
    model = POYOPlus(
        dim=32,
        dim_head=16,
        num_latents=8,
        depth=2,
        readout_specs=task_specs,
    )

    # initialize unit vocab with 100 units labeled 0-99
    model.unit_emb.initialize_vocab(np.arange(100))
    # initialize session vocab with 10 sessions labeled 0-9
    model.session_emb.initialize_vocab(np.arange(10))

    return model


def test_poyo_plus_forward(model):
    batch_size = 2
    n_in = 10
    n_latent = 8
    n_out = 4

    # Create dummy input data
    inputs = {
        "input_unit_index": torch.randint(0, 100, (batch_size, n_in)),
        "input_timestamps": torch.rand(batch_size, n_in),
        "input_token_type": torch.randint(0, 4, (batch_size, n_in)),
        "input_mask": torch.ones(batch_size, n_in, dtype=torch.bool),
        "latent_index": torch.arange(n_latent).repeat(batch_size, 1),
        "latent_timestamps": torch.linspace(0, 1, n_latent).repeat(batch_size, 1),
        "output_session_index": torch.zeros(batch_size, n_out, dtype=torch.long),
        "output_timestamps": torch.rand(batch_size, n_out),
        "output_decoder_index": torch.zeros(batch_size, n_out, dtype=torch.long),
    }

    # Forward pass
    outputs = model(**inputs)
    assert isinstance(outputs, dict)
    assert outputs["cursor_velocity_2d"].shape == (batch_size * n_out, 2)

    # Try with unpack_output=True
    outputs = model(**inputs, unpack_output=True)
    assert isinstance(outputs, list)
    assert len(outputs) == batch_size
    assert outputs[0]["cursor_velocity_2d"].shape == (n_out, 2)


def test_poyo_plus_tokenizer(task_specs):
    # Create dummy data similar to test_dataset_sim.py
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            unit_index=np.random.randint(0, 3, 1000),
            domain="auto",
        ),
        domain=Interval(0, 1),
        cursor=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            vel=np.random.normal(0, 1, (1000, 2)),
            domain="auto",
        ),
        gaze=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            position=np.random.normal(0, 1, (1000, 2)),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["unit1", "unit2", "unit3"])),
        session="session1",
        # Add config matching the YAML structure
        config={
            "multitask_readout": [
                {
                    "readout_id": "cursor_velocity_2d",
                    "metrics": [
                        {
                            "metric": "r2",
                            "task": "REACHING",
                        }
                    ],
                }
            ]
        },
    )

    # Create mock tokenizers
    unit_tokenizer = lambda x: np.arange(len(x))
    session_tokenizer = lambda x: 0

    # Initialize tokenizer
    tokenizer = POYOPlusTokenizer(
        unit_tokenizer=unit_tokenizer,
        session_tokenizer=session_tokenizer,
        decoder_registry=task_specs,
        latent_step=0.1,
        num_latents_per_step=8,
    )

    # Apply tokenizer
    batch = tokenizer(data)

    # Check that all expected keys are present
    expected_keys = {
        "input_unit_index",
        "input_timestamps",
        "input_token_type",
        "input_mask",
        "latent_index",
        "latent_timestamps",
        "output_session_index",
        "output_timestamps",
        "output_decoder_index",
        "target_values",
        "target_weights",
    }
    assert set(batch.keys()) == expected_keys

    # Check that output values contain the expected tasks
    assert set(batch["target_values"].obj.keys()).issubset(set(task_specs.keys()))

    # Verify latent tokens
    assert batch["latent_index"].shape[0] == len(np.arange(0, 1, 0.1)) * 8
    assert batch["latent_timestamps"].shape[0] == len(np.arange(0, 1, 0.1)) * 8


def test_poyo_plus_tokenizer_to_model(task_specs, model):
    # Create dummy data similar to test_dataset_sim.py
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            unit_index=np.random.randint(0, 3, 1000),
            domain="auto",
        ),
        domain=Interval(0, 1),
        cursor=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            vel=np.random.normal(0, 1, (1000, 2)),
            domain="auto",
        ),
        gaze=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            position=np.random.normal(0, 1, (1000, 2)),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["unit1", "unit2", "unit3"])),
        session="session1",
        config={
            "multitask_readout": [
                {
                    "readout_id": "cursor_velocity_2d",
                    "subtask_weights": {
                        "REACHING.RANDOM": 1.0,
                        "REACHING.HOLD": 0.1,
                        "REACHING.REACH": 5.0,
                        "REACHING.RETURN": 1.0,
                        "REACHING.INVALID": 0.1,
                        "REACHING.OUTLIER": 0.0,
                    },
                    "metrics": [
                        {
                            "metric": "r2",
                            "task": "REACHING",
                            "subtask": "REACHING.REACH",
                        }
                    ],
                }
            ]
        },
    )

    # Create mock tokenizers
    unit_tokenizer = lambda x: np.arange(len(x))
    session_tokenizer = lambda x: 0

    # Initialize tokenizer
    tokenizer = POYOPlusTokenizer(
        unit_tokenizer=unit_tokenizer,
        session_tokenizer=session_tokenizer,
        decoder_registry=task_specs,
        latent_step=0.1,
        num_latents_per_step=8,
    )

    # Apply tokenizer
    batch = tokenizer(data)

    # Create a batch list with a single element (simulating a batch size of 1)
    batch_list = [batch]

    # Use collate to properly batch the inputs
    model_inputs = collate(batch_list)

    model_inputs.pop("target_values")
    model_inputs.pop("target_weights")
    # Forward pass through model
    outputs = model(**model_inputs)

    # Basic checks
    assert isinstance(outputs, dict)
