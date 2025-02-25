import pytest
import torch
import torchmetrics
import numpy as np
from temporaldata import Interval

from torch_brain.metrics.metric_group import MetricGroupWithStitcher
from torch_brain.data.sampler import SequentialFixedWindowSampler


def test_metric_group_single_task():
    # Create metrics dict with a single task
    metrics = {
        "recording1": torchmetrics.Accuracy(task="binary"),
        "recording2": [
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Precision(task="binary"),
        ],
    }

    metric_group = MetricGroupWithStitcher(metrics)
    assert metric_group._depth == 1
    assert metric_group._smart_flushing == False
    assert metric_group._sequence_index is None

    # Test update and compute
    preds = torch.tensor([1, 0, 1, 1])
    targets = torch.tensor([1, 1, 1, 0])
    timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0])

    metric_group.update(preds, targets, timestamps, "recording1")
    metric_group.update(preds, targets, timestamps, "recording2")

    results = metric_group.compute()
    assert "recording1/BinaryAccuracy()" in results
    assert "recording2/BinaryAccuracy()" in results
    assert "recording2/BinaryPrecision()" in results
    metric_group.reset()


def test_metric_group_multi_task():
    # Create metrics dict with depth 2
    metrics = {
        "recording1": {
            "task1": torchmetrics.Accuracy(task="binary"),
            "task2": [
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Precision(task="binary"),
            ],
        }
    }

    metric_group = MetricGroupWithStitcher(metrics)
    assert metric_group._depth == 2
    assert metric_group._smart_flushing == False
    assert metric_group._sequence_index is None

    # Test update and compute
    preds = {"task1": torch.tensor([1, 0, 1, 1]), "task2": torch.tensor([1, 1, 0, 0])}
    targets = {"task1": torch.tensor([1, 1, 1, 0]), "task2": torch.tensor([1, 1, 0, 1])}
    timestamps = {
        "task1": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        "task2": torch.tensor([0.0, 1.0, 2.0, 3.0]),
    }

    metric_group.update(preds, targets, timestamps, "recording1")

    results = metric_group.compute()
    assert "recording1/task1/BinaryAccuracy()" in results
    assert "recording1/task2/BinaryAccuracy()" in results
    assert "recording1/task2/BinaryPrecision()" in results
    metric_group.reset()


def test_stitcher_sampler_conversion():
    metrics = {
        "recording1": torchmetrics.Accuracy(task="binary"),
    }
    metric_group = MetricGroupWithStitcher(metrics)

    # Create a sampler
    intervals = {"recording1": Interval(start=np.array([0.0]), end=np.array([10.0]))}
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=intervals, window_length=2.0, step=1.0
    )
    assert len(sampler) == 9
    # Convert to stitcher sampler
    stitcher_sampler = metric_group.convert_to_stitcher_sampler(sampler)

    assert metric_group._smart_flushing == True
    assert metric_group._sequence_index is not None


def test_invalid_metrics():
    # Test invalid metrics type
    with pytest.raises(ValueError):
        MetricGroupWithStitcher([torchmetrics.Accuracy(task="binary")])

    # Test invalid depth
    with pytest.raises(ValueError):
        MetricGroupWithStitcher(
            {"rec1": {"task1": {"subtask1": torchmetrics.Accuracy(task="binary")}}}
        )

    # Test invalid metric type
    with pytest.raises(ValueError):
        MetricGroupWithStitcher({"rec1": "not_a_metric"})


def test_smart_flushing():
    metrics = {"recording1": torchmetrics.Accuracy(task="binary")}
    metric_group = MetricGroupWithStitcher(metrics)

    # Setup smart flushing
    intervals = {"recording1": Interval(start=np.array([0.0]), end=np.array([10.0]))}
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=intervals, window_length=2.0, step=1.0
    )
    stitcher_sampler = metric_group.convert_to_stitcher_sampler(sampler)

    # Test update with overlapping windows
    preds = torch.tensor([1, 0])
    targets = torch.tensor([1, 1])
    timestamps = torch.tensor([0.0, 1.0])

    metric_group.update(preds, targets, timestamps, "recording1")

    # Verify cache is managed correctly
    assert metric_group._sample_ptr == 1
    metric_group.reset()
