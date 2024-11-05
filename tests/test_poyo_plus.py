import pytest
import torch
import numpy as np
from unittest.mock import Mock
from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict

from torch_brain.data import collate
from torch_brain.nn import OutputType
from torch_brain.models.poyo_plus import POYOPlus, POYOPlusTokenizer
from torch_brain.nn.multitask_readout import Decoder, DecoderSpec


@pytest.fixture
def task_specs():
    return {
        str(Decoder.CURSORVELOCITY2D): DecoderSpec(
            dim=2,
            type=OutputType.CONTINUOUS,
            loss_fn="mse",
            timestamp_key="cursor.timestamps",
            value_key="cursor.vel",
        ),
        "GAZE_POS_2D": DecoderSpec(
            dim=2,
            type=OutputType.CONTINUOUS,
            loss_fn="mse",
            timestamp_key="gaze.timestamps",
            value_key="gaze.position",
        ),
    }


@pytest.fixture
def model(task_specs):
    model = POYOPlus(
        dim=32,
        dim_head=16,
        num_latents=8,
        depth=2,
        task_specs=task_specs,
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
        "target_values": [
            {
                str(Decoder.CURSORVELOCITY2D): torch.randn(n_out, 2),
                "GAZE_POS_2D": torch.randn(n_out, 2),
            }
            for _ in range(batch_size)
        ],
        "target_weights": [
            {
                str(Decoder.CURSORVELOCITY2D): torch.ones(n_out),
                "GAZE_POS_2D": torch.ones(n_out),
            }
            for _ in range(batch_size)
        ],
    }

    # Forward pass
    outputs, loss, losses_taskwise = model(**inputs)

    # Basic shape checks
    assert isinstance(outputs, list)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(losses_taskwise, dict)
    assert loss.shape == torch.Size([])


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
                    "decoder_id": "CURSORVELOCITY2D",
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
                    "decoder_id": "CURSORVELOCITY2D",
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

    # Forward pass through model
    outputs, loss, losses_taskwise = model(**model_inputs)

    # Basic checks
    assert isinstance(outputs, list)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(losses_taskwise, dict)
    assert loss.shape == torch.Size([])

    # Check outputs structure matches task_specs
    assert len(outputs) == 1  # batch size of 1
    assert set(outputs[0].keys()).issubset(set(task_specs.keys()))
