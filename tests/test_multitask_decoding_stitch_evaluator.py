import pytest
import torch
import torchmetrics

from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator


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


def test_initialization(mock_metrics):
    sequence_index = torch.tensor([0, 0, 1, 1, 1, 2])
    evaluator = MultiTaskDecodingStitchEvaluator(
        metrics=mock_metrics,
        sequence_index=sequence_index,
    )

    assert evaluator.metrics is not None
    assert "session1" in evaluator.metrics
    assert "task1" in evaluator.metrics["session1"]
    assert "task2" in evaluator.metrics["session1"]

    assert evaluator._sample_ptr == 0
    assert len(evaluator._cache) == 3
    assert evaluator._counter == [0, 0, 0]
    assert torch.equal(evaluator._cache_flush_threshold, torch.tensor([2, 3, 1]))
