import pytest
import torch
import lightning as L
import torchmetrics
from collections import defaultdict

from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator


class MockDataModule:
    def __init__(self, sequence_index):
        self.val_sequence_index = sequence_index
        self.test_sequence_index = sequence_index


class MockTrainer:
    def __init__(self, sequence_index):
        self.datamodule = MockDataModule(sequence_index)
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
    sequence_index = torch.tensor([0, 0, 1, 1, 1])
    trainer = MockTrainer(sequence_index)

    evaluator._on_epoch_start_helper(trainer, mode="val")

    assert evaluator.sample_ptr == 0
    assert len(evaluator.cache) == 2  # max sequence_index + 1
    assert evaluator.counter == [0, 0]
    assert torch.equal(evaluator.cache_flush_threshold, torch.tensor([2, 3]))


def test_on_test_epoch_start(evaluator):
    sequence_index = torch.tensor([0, 0, 1, 1, 1])
    trainer = MockTrainer(sequence_index)

    evaluator._on_epoch_start_helper(trainer, mode="test")

    assert evaluator.sample_ptr == 0
    assert len(evaluator.cache) == 2
    assert evaluator.counter == [0, 0]
    assert torch.equal(evaluator.cache_flush_threshold, torch.tensor([2, 3]))
