import pytest
import torch
import torchmetrics

from torch_brain.metrics import MetricGroupWithStitcher


@pytest.fixture
def mock_metrics():
    return {
        "session1": {
            "task1": [
                torchmetrics.Accuracy(task="multiclass", num_classes=3),
                torchmetrics.F1Score(task="multiclass", num_classes=3),
            ],
            "task2": [torchmetrics.MeanSquaredError()],
        }
    }


def test_initialization(mock_metrics):
    sequence_index = torch.tensor([0, 0, 1, 1, 1, 2])
    metric_group = MetricGroupWithStitcher(metrics=mock_metrics)
    metric_group._sequence_index = sequence_index
    metric_group._smart_flushing = True
    metric_group._init_cache()

    assert metric_group.metrics is not None
    assert "session1" in metric_group.metrics
    assert "task1" in metric_group.metrics["session1"]
    assert "task2" in metric_group.metrics["session1"]

    assert metric_group._sample_ptr == 0
    assert len(metric_group._cache) == 3
    assert metric_group._counter == [0, 0, 0]
    assert torch.equal(metric_group._cache_flush_threshold, torch.tensor([2, 3, 1]))
