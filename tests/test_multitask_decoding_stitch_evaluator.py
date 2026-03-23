import pytest
import torch
import lightning as L
import torchmetrics
from collections import defaultdict

import torch_brain
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.utils.callbacks import (
    MultiTaskDecodingStitchEvaluator,
    DataForMultiTaskDecodingStitchEvaluator,
)


class MockDataModule:
    def __init__(self, val_sequence_index, test_sequence_index):
        self.val_sequence_index = val_sequence_index
        self.test_sequence_index = test_sequence_index


class MockTrainer:
    def __init__(self, val_sequence_index, test_sequence_index):
        self.datamodule = MockDataModule(val_sequence_index, test_sequence_index)
        self.loggers = []
        self.is_global_zero = True


@pytest.fixture
def mock_metrics():
    return {
        "session1": {
            "task1": {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=3),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=3),
            },
            "task2": {"mse": torchmetrics.MeanSquaredError()},
        }
    }


@pytest.fixture
def evaluator(mock_metrics):
    return MultiTaskDecodingStitchEvaluator(metrics=mock_metrics)


def test_initialization(evaluator):
    assert evaluator.metrics is not None
    assert "session1" in evaluator.metrics
    assert "task1" in evaluator.metrics["session1"]
    assert "task2" in evaluator.metrics["session1"]


def test_on_validation_epoch_start(evaluator):
    val_sequence_index = torch.tensor([0, 0, 1, 1, 1])
    test_sequence_index = torch.tensor([0, 1, 1, 2, 2])  # different sequence
    trainer = MockTrainer(val_sequence_index, test_sequence_index)

    evaluator.on_validation_epoch_start(trainer, None)

    assert evaluator.sample_ptr == 0
    assert len(evaluator.cache) == 2  # max val_sequence_index + 1
    assert evaluator.counter == [0, 0]
    assert torch.equal(evaluator.cache_flush_threshold, torch.tensor([2, 3]))


def test_on_test_epoch_start(evaluator):
    val_sequence_index = torch.tensor([0, 0, 1, 1, 1])
    test_sequence_index = torch.tensor([0, 1, 1, 2, 2])  # different sequence
    trainer = MockTrainer(val_sequence_index, test_sequence_index)

    evaluator.on_test_epoch_start(trainer, None)

    assert evaluator.sample_ptr == 0
    assert len(evaluator.cache) == 3  # max test_sequence_index + 1
    assert evaluator.counter == [0, 0, 0]  # three counters for three sequences
    assert torch.equal(evaluator.cache_flush_threshold, torch.tensor([1, 2, 2]))


def test_cache_assignment():
    vel = "cursor_velocity_2d"
    pos = "cursor_position_2d"
    arm = "arm_velocity_2d"
    vel_id = MODALITY_REGISTRY[vel].id
    pos_id = MODALITY_REGISTRY[pos].id
    arm_id = MODALITY_REGISTRY[arm].id

    # 6 samples, T_max=3. Padding (0) where tokens are fewer than 3.
    #   sample 0 (sess_A): vel vel vel     — 3 tokens
    #   sample 1 (sess_A): vel pos  0      — 2 tokens
    #   sample 2 (sess_B): pos  0   0      — 1 token
    #   sample 3 (sess_B): pos vel pos     — 3 tokens
    #   sample 4 (sess_C): arm arm  0      — 2 tokens
    #   sample 5 (sess_C): arm  0   0      — 1 token
    decoder_indices = torch.tensor(
        [
            [vel_id, vel_id, vel_id],
            [vel_id, pos_id, 0],
            [pos_id, 0, 0],
            [pos_id, vel_id, pos_id],
            [arm_id, arm_id, 0],
            [arm_id, 0, 0],
        ]
    )
    timestamps = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.0, 0.1, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    preds = [
        {vel: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        {vel: torch.tensor([[7.0, 8.0]]), pos: torch.tensor([[9.0, 10.0]])},
        {pos: torch.tensor([[11.0, 12.0]])},
        {
            pos: torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
            vel: torch.tensor([[17.0, 18.0]]),
        },
        {arm: torch.tensor([[19.0, 20.0], [21.0, 22.0]])},
        {arm: torch.tensor([[23.0, 24.0]])},
    ]

    # targets/eval_masks are concatenated across the batch per readout type,
    # ordered as torch.where(decoder_indices == readout_id) would yield
    targets = {
        vel: torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [17.0, 18.0]]
        ),
        pos: torch.tensor([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]),
        arm: torch.tensor([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
    }
    eval_masks = {
        vel: torch.ones(5, dtype=torch.bool),
        pos: torch.ones(4, dtype=torch.bool),
        arm: torch.ones(3, dtype=torch.bool),
    }

    data = DataForMultiTaskDecodingStitchEvaluator(
        timestamps=timestamps,
        preds=preds,
        targets=targets,
        decoder_indices=decoder_indices,
        eval_masks=eval_masks,
        session_ids=["sess_A", "sess_A", "sess_B", "sess_B", "sess_C", "sess_C"],
        absolute_starts=torch.zeros(6),
    )

    mse = lambda: torchmetrics.MeanSquaredError()
    metrics = {
        "sess_A": {vel: {"mse": mse()}, pos: {"mse": mse()}},
        "sess_B": {vel: {"mse": mse()}, pos: {"mse": mse()}},
        "sess_C": {arm: {"mse": mse()}},
    }
    evaluator = MultiTaskDecodingStitchEvaluator(metrics=metrics)

    # Simulates a val set of 9 samples (3 per session/sequence) so the cache
    # flush threshold (3) isn't reached by this single batch of 6.
    sequence_index = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2])
    trainer = MockTrainer(sequence_index, sequence_index)
    evaluator.on_validation_epoch_start(trainer, None)
    evaluator.on_validation_batch_end(trainer, None, data)

    # sess_A (seq 0): vel from samples 0,1; pos from sample 1 only
    assert len(evaluator.cache[0]["pred"][vel]) == 2
    assert len(evaluator.cache[0]["pred"][pos]) == 1
    assert arm not in evaluator.cache[0]["pred"]
    assert torch.equal(
        torch.cat(evaluator.cache[0]["pred"][vel]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )
    assert torch.equal(
        torch.cat(evaluator.cache[0]["pred"][pos]),
        torch.tensor([[9.0, 10.0]]),
    )

    # sess_B (seq 1): pos from samples 2,3; vel from sample 3 only
    assert len(evaluator.cache[1]["pred"][pos]) == 2
    assert len(evaluator.cache[1]["pred"][vel]) == 1
    assert arm not in evaluator.cache[1]["pred"]
    assert torch.equal(
        torch.cat(evaluator.cache[1]["pred"][pos]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]),
    )
    assert torch.equal(
        torch.cat(evaluator.cache[1]["pred"][vel]),
        torch.tensor([[17.0, 18.0]]),
    )

    # sess_C (seq 2): arm from samples 4,5 only
    assert len(evaluator.cache[2]["pred"][arm]) == 2
    assert vel not in evaluator.cache[2]["pred"]
    assert pos not in evaluator.cache[2]["pred"]
    assert torch.equal(
        torch.cat(evaluator.cache[2]["pred"][arm]),
        torch.tensor([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
    )
