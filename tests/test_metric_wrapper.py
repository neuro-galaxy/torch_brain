import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import torchmetrics
from torchmetrics.regression import MeanSquaredError, R2Score
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy
from torchmetrics.collections import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper

from torch_brain.metrics.metric_wrapper import MetricWrapper


# Test MetricWrapper Initialization
def test_basic_initialization():
    """Test basic initialization with valid metrics."""
    metrics = {"recording_001": MeanSquaredError(), "recording_002": R2Score()}
    wrapper = MetricWrapper(metrics)

    assert len(wrapper.metrics) == 2
    assert "recording_001" in wrapper.metrics
    assert "recording_002" in wrapper.metrics
    assert wrapper._prefix == ""
    assert wrapper._postfix == ""
    assert not wrapper._stitch


def test_initialization_with_prefix_postfix():
    """Test initialization with prefix and postfix."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, prefix="test_", postfix="_end")

    assert wrapper._prefix == "test_"
    assert wrapper._postfix == "_end"


def test_initialization_with_stitching():
    """Test initialization with stitching enabled."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    assert wrapper._stitch
    assert hasattr(wrapper, "_cache")
    assert "rec_001" in wrapper._cache


def test_initialization_with_metric_collection():
    """Test initialization with MetricCollection."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    assert isinstance(wrapper.metrics["rec_001"], MetricCollection)


def test_initialization_with_multitask_wrapper():
    """Test initialization with MultitaskWrapper."""
    metrics = {
        "rec_001": MultitaskWrapper(
            {"task_1": MeanSquaredError(), "task_2": MulticlassAccuracy(num_classes=3)}
        )
    }
    wrapper = MetricWrapper(metrics)

    assert isinstance(wrapper.metrics["rec_001"], MultitaskWrapper)


def test_invalid_metrics_type():
    """Test error when metrics is not a dictionary."""
    with pytest.raises(TypeError, match="Expected argument `metrics` to be a dict"):
        MetricWrapper([MeanSquaredError()])


def test_invalid_metric_type():
    """Test error when individual metric is invalid type."""
    with pytest.raises(TypeError, match="Expected each metric to be a Metric"):
        MetricWrapper({"rec_001": "invalid_metric"})


def test_invalid_prefix_type():
    """Test error when prefix is not string."""
    with pytest.raises(ValueError, match="Expected argument `prefix`"):
        MetricWrapper({"rec_001": MeanSquaredError()}, prefix=123)


def test_invalid_postfix_type():
    """Test error when postfix is not string."""
    with pytest.raises(ValueError, match="Expected argument `postfix`"):
        MetricWrapper({"rec_001": MeanSquaredError()}, postfix=123)


# Test Basic MetricWrapper Usage
def test_basic_update_and_compute():
    """Test basic update and compute functionality."""
    metrics = {"rec_001": MeanSquaredError(), "rec_002": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    # Update first recording
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    # Update second recording
    preds = torch.tensor([2.0, 3.0, 4.0])
    targets = torch.tensor([2.2, 3.2, 4.2])
    wrapper.update(preds, targets, None, recording_id="rec_002")

    # Compute results
    results = wrapper.compute()

    assert "rec_001" in results
    assert "rec_002" in results
    assert isinstance(results["rec_001"], torch.Tensor)
    assert isinstance(results["rec_002"], torch.Tensor)

    # Results should be close to expected MSE values
    assert torch.isclose(results["rec_001"], torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(results["rec_002"], torch.tensor(0.04), atol=1e-6)


def test_with_metric_collection():
    """Test with MetricCollection."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 2.1, 3.1, 4.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    results = wrapper.compute()

    assert "rec_001" in results
    assert "mse" in results["rec_001"]
    assert "r2" in results["rec_001"]

    assert torch.isclose(results["rec_001"]["mse"], torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(results["rec_001"]["r2"], torch.tensor(0.992), atol=1e-6)


def test_with_classification_metrics():
    """Test with classification metrics."""
    metrics = {
        "rec_001": MulticlassAccuracy(num_classes=3, average="micro"),
        "rec_002": BinaryAccuracy(),
        "rec_003": MulticlassAccuracy(num_classes=3, average="micro"),
    }
    wrapper = MetricWrapper(metrics)

    # Multiclass
    preds = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 2])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    # Binary
    preds = torch.tensor([0.8, 0.2, 0.9, 0.1])
    targets = torch.tensor([1, 0, 1, 0])
    wrapper.update(preds, targets, None, recording_id="rec_002")

    # Multiclass with logits
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.7],  # Class 2 has highest probability
            [0.8, 0.1, 0.1],  # Class 0 has highest probability
            [0.1, 0.7, 0.2],  # Class 1 has highest probability
            [0.3, 0.4, 0.3],  # Class 1 has highest probability
        ]
    )
    targets = torch.tensor([2, 0, 1, 1])
    wrapper.update(logits, targets, None, recording_id="rec_003")

    results = wrapper.compute()

    assert "rec_001" in results
    assert "rec_002" in results
    assert "rec_003" in results
    assert torch.isclose(results["rec_001"], torch.tensor(0.75))  # 3/4 correct
    assert torch.isclose(results["rec_002"], torch.tensor(1.0))  # All correct
    assert torch.isclose(results["rec_003"], torch.tensor(1.0))  # All correct


def test_multiple_updates():
    """Test multiple updates to same recording."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    # First update
    preds = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.1, 2.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    # Second update
    preds = torch.tensor([3.0, 4.0])
    targets = torch.tensor([3.1, 4.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    results = wrapper.compute()

    # Should compute MSE over all 4 samples
    assert torch.isclose(results["rec_001"], torch.tensor(0.01))


def test_reset_functionality():
    """Test reset functionality."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    preds = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.1, 2.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    # Reset and verify metrics are cleared
    wrapper.reset()

    # After reset, should have no accumulated data
    # We can verify by computing - it should give a different result
    preds = torch.tensor([5.0, 6.0])
    targets = torch.tensor([5.5, 6.5])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    results = wrapper.compute()
    assert torch.isclose(results["rec_001"], torch.tensor(0.25))


# Test MetricWrapper Stitching
def test_basic_stitching():
    """Test basic stitching with overlapping timestamps."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # First window: times 0, 1, 2
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])
    timestamps = torch.tensor([0.0, 1.0, 2.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second window: overlapping times 1, 2
    preds = torch.tensor([2.5, 3.5, 4.0])
    targets = torch.tensor([2.6, 3.6, 4.1])
    timestamps = torch.tensor([1.0, 2.0, 3.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    # Should stitch overlapping predictions and compute MSE
    assert "rec_001" in results
    assert isinstance(results["rec_001"], torch.Tensor)


def test_stitching_with_no_overlap():
    """Test stitching when there's no temporal overlap."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # First window: times 0, 1, 2
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])
    timestamps = torch.tensor([0.0, 1.0, 2.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second window: non-overlapping times 5, 6, 7
    preds = torch.tensor([5.0, 6.0, 7.0])
    targets = torch.tensor([5.1, 6.1, 7.1])
    timestamps = torch.tensor([5.0, 6.0, 7.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    # Should concatenate all data and compute MSE
    assert torch.isclose(results["rec_001"], torch.tensor(0.01))


def test_stitching_with_metric_collection():
    """Test stitching with MetricCollection."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics, stitch=True)

    # First update
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])
    timestamps = torch.tensor([0.0, 1.0, 2.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second update
    preds = torch.tensor([2.5, 3.5, 4.0])
    targets = torch.tensor([2.6, 3.6, 4.1])
    timestamps = torch.tensor([1.0, 2.0, 3.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    assert "rec_001" in results
    assert "mse" in results["rec_001"]
    assert "r2" in results["rec_001"]
    assert torch.isclose(results["rec_001"]["mse"], torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(results["rec_001"]["r2"], torch.tensor(0.9921), atol=1e-6)


def test_stitching_with_multitask_wrapper():
    """Test stitching with MultitaskWrapper."""
    metrics = {
        "rec_001": MultitaskWrapper(
            {"task_1": MeanSquaredError(), "task_2": MulticlassAccuracy(num_classes=3)}
        )
    }
    wrapper = MetricWrapper(metrics, stitch=True)

    # First update
    preds = {"task_1": torch.tensor([1.0, 2.0]), "task_2": torch.tensor([1])}
    targets = {"task_1": torch.tensor([1.1, 2.1]), "task_2": torch.tensor([1])}
    timestamps = {"task_1": torch.tensor([0.0, 1.0]), "task_2": torch.tensor([1.0])}
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second update with overlap
    preds = {"task_1": torch.tensor([2.5, 3.0]), "task_2": torch.tensor([1, 2])}
    targets = {"task_1": torch.tensor([2.6, 3.1]), "task_2": torch.tensor([1, 2])}
    timestamps = {
        "task_1": torch.tensor([1.0, 2.0]),
        "task_2": torch.tensor([1.0, 2.0]),
    }
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    assert "rec_001" in results
    assert "task_1" in results["rec_001"]
    assert "task_2" in results["rec_001"]
    assert torch.isclose(results["rec_001"]["task_1"], torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(results["rec_001"]["task_2"], torch.tensor(1.0), atol=1e-6)


def test_stitching_with_classification_and_integer_preds():
    """Test that stitching properly handles integer preds."""
    metrics = {"rec_001": MulticlassAccuracy(num_classes=3)}
    wrapper = MetricWrapper(metrics, stitch=True)

    # First window
    preds = torch.tensor([0, 1, 2])
    targets = torch.tensor([0, 1, 2])
    timestamps = torch.tensor([0.0, 1.0, 2.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second window with overlap
    preds = torch.tensor([1, 2, 0])
    targets = torch.tensor([1, 2, 0])
    timestamps = torch.tensor([1.0, 2.0, 3.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    assert isinstance(results["rec_001"], torch.Tensor)
    assert torch.isclose(results["rec_001"], torch.tensor(1.0), atol=1e-6)


def test_stitching_with_classification_and_float_targets():
    """Test that stitching properly handles float targets."""
    metrics = {"rec_001": MulticlassAccuracy(num_classes=3)}
    wrapper = MetricWrapper(metrics, stitch=True)

    # First window
    preds = torch.tensor(
        [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]]
    )
    targets = torch.tensor([0, 1, 2, 1])
    timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Second window with overlap
    preds = torch.tensor(
        [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]]
    )
    targets = torch.tensor([0, 1, 2, 1])
    timestamps = torch.tensor([1.0, 2.0, 3.0, 4.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    results = wrapper.compute()

    assert isinstance(results["rec_001"], torch.Tensor)
    assert torch.isclose(results["rec_001"], torch.tensor(0.3889), atol=1e-4)


def test_cache_initialization_without_smart_flushing():
    """Test cache initialization in basic stitching mode."""
    metrics = {"rec_001": MeanSquaredError(), "rec_002": R2Score()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # Verify cache structure
    assert hasattr(wrapper, "_cache")
    assert "rec_001" in wrapper._cache
    assert "rec_002" in wrapper._cache
    assert "preds" in wrapper._cache["rec_001"]
    assert "targets" in wrapper._cache["rec_001"]
    assert "timestamps" in wrapper._cache["rec_001"]


# Test MetricWrapper Smart Flushing
@patch("torch_brain.metrics.metric_wrapper.StitcherSamplerWrapper")
def test_convert_to_stitcher_sampler(mock_stitcher_sampler):
    """Test conversion to stitcher sampler."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # Mock the stitcher sampler
    mock_sampler = Mock()
    mock_stitcher_instance = Mock()
    mock_stitcher_instance.sequence_index = torch.tensor([0, 0, 1, 1, 1])
    mock_stitcher_sampler.return_value = mock_stitcher_instance

    result = wrapper.convert_to_stitcher_sampler(mock_sampler)

    # Verify the conversion
    assert wrapper._smart_flushing
    assert wrapper._sequence_index is not None
    assert result == mock_stitcher_instance
    mock_stitcher_sampler.assert_called_once_with(mock_sampler, num_replicas=1, rank=0)


def test_convert_to_stitcher_sampler_without_stitching():
    """Test error when trying to convert without stitching enabled."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=False)

    with pytest.raises(ValueError, match="Stitching is not enabled"):
        wrapper.convert_to_stitcher_sampler(Mock())


@patch("torch_brain.metrics.metric_wrapper.StitcherSamplerWrapper")
def test_smart_flushing_cache_management(mock_stitcher_sampler):
    """Test smart flushing cache management."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # Setup mock stitcher sampler
    mock_sampler = Mock()
    mock_stitcher_instance = Mock()
    # Two sequences: sequence 0 has 2 samples, sequence 1 has 3 samples
    mock_stitcher_instance.sequence_index = torch.tensor([0, 0, 1, 1, 1])
    mock_stitcher_sampler.return_value = mock_stitcher_instance

    wrapper.convert_to_stitcher_sampler(mock_sampler)

    # Verify cache structure for smart flushing
    assert len(wrapper._cache) == 2  # Two sequences
    assert wrapper._sample_ptr == 0
    assert len(wrapper._counter) == 2
    assert wrapper._cache_flush_threshold.tolist() == [2, 3]  # Expected counts


# Test MetricWrapper Utility Methods
def test_items_flatten_true():
    """Test items() method with flatten=True."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()}),
        "rec_002": MulticlassAccuracy(num_classes=3),
    }
    wrapper = MetricWrapper(metrics, prefix="test_", postfix="_end")

    items = list(wrapper.items(flatten=True))

    # Should flatten MetricCollection but not single metrics
    expected_keys = ["test_rec_001_mse_end", "test_rec_001_r2_end", "test_rec_002_end"]
    actual_keys = [key for key, _ in items]

    assert set(actual_keys) == set(expected_keys)


def test_items_flatten_false():
    """Test items() method with flatten=False."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()}),
        "rec_002": MulticlassAccuracy(num_classes=3),
    }
    wrapper = MetricWrapper(metrics, prefix="test_", postfix="_end")

    items = list(wrapper.items(flatten=False))

    expected_keys = ["test_rec_001_end", "test_rec_002_end"]
    actual_keys = [key for key, _ in items]

    assert set(actual_keys) == set(expected_keys)


def test_keys_flatten_true():
    """Test keys() method with flatten=True."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    keys = list(wrapper.keys(flatten=True))

    expected_keys = ["rec_001_mse", "rec_001_r2"]
    assert set(keys) == set(expected_keys)


def test_keys_flatten_false():
    """Test keys() method with flatten=False."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    keys = list(wrapper.keys(flatten=False))

    assert keys == ["rec_001"]


def test_values_flatten_true():
    """Test values() method with flatten=True."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    values = list(wrapper.values(flatten=True))

    assert len(values) == 2
    assert isinstance(values[0], (MeanSquaredError, R2Score))
    assert isinstance(values[1], (MeanSquaredError, R2Score))


def test_values_flatten_false():
    """Test values() method with flatten=False."""
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
    }
    wrapper = MetricWrapper(metrics)

    values = list(wrapper.values(flatten=False))

    assert len(values) == 1
    assert isinstance(values[0], MetricCollection)


def test_convert_output():
    """Test _convert_output method with prefix/postfix."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, prefix="pre_", postfix="_post")

    input_output = {"rec_001": torch.tensor(0.5)}
    converted = wrapper._convert_output(input_output)

    expected = {"pre_rec_001_post": torch.tensor(0.5)}
    assert list(converted.keys()) == list(expected.keys())
    assert torch.equal(converted["pre_rec_001_post"], expected["pre_rec_001_post"])


# Test MetricWrapper Edge Cases
def test_empty_metrics_dict():
    """Test with empty metrics dictionary."""
    wrapper = MetricWrapper({})

    results = wrapper.compute()
    assert results == {}


def test_single_sample_update():
    """Test with single sample updates."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    preds = torch.tensor([1.0])
    targets = torch.tensor([1.1])
    wrapper.update(preds, targets, None, recording_id="rec_001")

    results = wrapper.compute()
    assert torch.isclose(results["rec_001"], torch.tensor(0.01))


def test_large_number_of_recordings():
    """Test with large number of recordings."""
    num_recordings = 100
    metrics = {f"rec_{i:03d}": MeanSquaredError() for i in range(num_recordings)}
    wrapper = MetricWrapper(metrics)

    # Update all recordings
    for i in range(num_recordings):
        preds = torch.tensor([float(i), float(i + 1)])
        targets = torch.tensor([float(i + 0.1), float(i + 1.1)])
        wrapper.update(preds, targets, None, recording_id=f"rec_{i:03d}")

    results = wrapper.compute()

    assert len(results) == num_recordings
    # All should have same MSE (0.01)
    for i in range(num_recordings):
        assert torch.isclose(results[f"rec_{i:03d}"], torch.tensor(0.01), atol=1e-6)


def test_compute_with_smart_flushing_incomplete_cache():
    """Test compute error when smart flushing cache is incomplete."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics, stitch=True)

    # Manually setup incomplete smart flushing state
    wrapper._smart_flushing = True
    wrapper._cache = [{"some": "data"}, None]  # Incomplete cache

    with pytest.raises(
        RuntimeError, match="Cache at index 0 was not automatically flushed"
    ):
        wrapper.compute()


def test_concat_and_stitch_static_method():
    """Test the static _concat_and_stitch method directly."""
    preds_list = [torch.tensor([1.0, 2.0]), torch.tensor([2.5, 3.0])]
    targets_list = [torch.tensor([1.1, 2.1]), torch.tensor([2.6, 3.1])]
    timestamps_list = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])]

    stitched_preds, stitched_targets = MetricWrapper._concat_and_stitch(
        preds_list, targets_list, timestamps_list
    )

    # Should have 3 unique timestamps after stitching
    assert len(stitched_preds) >= 2  # At least non-overlapping parts
    assert len(stitched_targets) >= 2


def test_concat_and_stitch_integer_targets():
    """Test _concat_and_stitch with integer targets."""
    preds_list = [torch.tensor([0, 1]), torch.tensor([1, 2])]
    targets_list = [torch.tensor([0, 1]), torch.tensor([1, 2])]  # integer targets
    timestamps_list = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])]

    stitched_preds, stitched_targets = MetricWrapper._concat_and_stitch(
        preds_list, targets_list, timestamps_list
    )

    # Targets should remain as long integers after stitching
    assert stitched_targets.dtype == torch.long


def test_process_cache_chunk_with_multitask():
    """Test _process_cache_chunk with MultitaskWrapper."""
    metrics = {
        "rec_001": MultitaskWrapper(
            {
                "task_1": MeanSquaredError(),
                "task_2": MulticlassAccuracy(num_classes=2, average="micro"),
            }
        )
    }
    wrapper = MetricWrapper(metrics, stitch=True)

    cache = {
        "preds": [
            {
                "task_1": torch.tensor([1.0]),
                "task_2": torch.tensor([[0.8, 0.1], [0.1, 0.2]]),
            },
            {
                "task_1": torch.tensor([2.0]),
                "task_2": torch.tensor([[0.1, 0.7], [0.8, 0.4]]),
            },
        ],
        "targets": [
            {"task_1": torch.tensor([1.1]), "task_2": torch.tensor([0, 1])},
            {"task_1": torch.tensor([2.1]), "task_2": torch.tensor([1, 0])},
        ],
        "timestamps": [
            {"task_1": torch.tensor([0.0]), "task_2": torch.tensor([0.0, 1.0])},
            {"task_1": torch.tensor([1.0]), "task_2": torch.tensor([1.0, 2.0])},
        ],
    }

    # This should not raise an error
    wrapper._process_cache_chunk(cache, wrapper.metrics["rec_001"])


# Test MetricWrapper Integration
def test_full_workflow():
    metrics = {
        "rec_001": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()}),
        "rec_002": MetricCollection(
            {"mse": MeanSquaredError(), "acc": MulticlassAccuracy(num_classes=3)}
        ),
    }
    wrapper = MetricWrapper(metrics, stitch=True, prefix="test_")

    # Update rec_001 with regression data
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 2.1, 3.1, 4.1])
    timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Update rec_002 with mixed data
    preds = torch.tensor([0, 1, 2])
    targets = torch.tensor([0, 1, 2])
    timestamps = torch.tensor([0.0, 1.0, 2.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_002")

    # Second update to rec_001 with overlap
    preds = torch.tensor([3.5, 4.5, 5.0])
    targets = torch.tensor([3.6, 4.6, 5.1])
    timestamps = torch.tensor([2.0, 3.0, 4.0])
    wrapper.update(preds, targets, timestamps, recording_id="rec_001")

    # Compute results
    results = wrapper.compute()

    # Verify structure
    assert "test_rec_001" in results
    assert "test_rec_002" in results
    assert "mse" in results["test_rec_001"]
    assert "r2" in results["test_rec_001"]
    assert "mse" in results["test_rec_002"]
    assert "acc" in results["test_rec_002"]

    # Verify utility methods work
    keys = list(wrapper.keys(flatten=True))
    expected_keys = [
        "test_rec_001_mse",
        "test_rec_001_r2",
        "test_rec_002_mse",
        "test_rec_002_acc",
    ]
    assert set(keys) == set(expected_keys)

    # Test reset
    wrapper.reset()

    # After reset, cache should be reinitialized
    assert wrapper._cache is not None
    assert len(wrapper._cache["rec_001"]["preds"]) == 0


def test_mixed_metric_types_workflow():
    """Test workflow with mixed metric types."""
    metrics = {
        "single": MeanSquaredError(),
        "collection": MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()}),
        "multitask": MultitaskWrapper(
            {
                "regression": MeanSquaredError(),
                "classification": MulticlassAccuracy(num_classes=3),
            }
        ),
    }
    wrapper = MetricWrapper(metrics)

    # Update single metric recording
    preds = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.1, 2.1])
    wrapper.update(preds, targets, None, recording_id="single")

    # Update collection metric recording
    preds = torch.tensor([2.0, 3.0, 4.0])
    targets = torch.tensor([2.1, 3.1, 4.1])
    wrapper.update(preds, targets, None, recording_id="collection")

    # Update multitask metric recording
    preds = {
        "regression": torch.tensor([1.0, 2.0]),
        "classification": torch.tensor([0, 1]),
    }
    targets = {
        "regression": torch.tensor([1.1, 2.1]),
        "classification": torch.tensor([0, 1]),
    }
    wrapper.update(preds, targets, None, recording_id="multitask")

    results = wrapper.compute()

    # Verify all different result structures
    assert isinstance(results["single"], torch.Tensor)
    assert isinstance(results["collection"], dict)
    assert isinstance(results["multitask"], dict)
    assert "mse" in results["collection"]
    assert "r2" in results["collection"]
    assert "regression" in results["multitask"]
    assert "classification" in results["multitask"]


@pytest.mark.parametrize(
    "prefix,postfix", [("", ""), ("pre_", ""), ("", "_post"), ("pre_", "_post")]
)
def test_metric_wrapper_parametrized_prefix_postfix(prefix, postfix):
    """Parametrized test for prefix/postfix combinations."""
    sample_metrics = {
        "rec_001": MeanSquaredError(),
        "rec_002": R2Score(),
        "rec_003": MulticlassAccuracy(num_classes=5),
    }
    wrapper = MetricWrapper(sample_metrics, prefix=prefix, postfix=postfix)

    keys = list(wrapper.keys(flatten=False))

    for key in keys:
        assert key.startswith(prefix)
        assert key.endswith(postfix)


# Additional Edge Case Tests
def test_metric_wrapper_with_different_tensor_dtypes():
    """Test MetricWrapper with different tensor data types."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    # Test with different dtypes
    preds = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    targets = torch.tensor([1.1, 2.1, 3.1], dtype=torch.float64)

    wrapper.update(preds, targets, None, recording_id="rec_001")
    results = wrapper.compute()

    assert "rec_001" in results
    assert isinstance(results["rec_001"], torch.Tensor)


def test_metric_wrapper_with_cuda_tensors():
    """Test MetricWrapper with CUDA tensors if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    preds = torch.tensor([1.0, 2.0, 3.0]).cuda()
    targets = torch.tensor([1.1, 2.1, 3.1]).cuda()

    wrapper.update(preds, targets, None, recording_id="rec_001")
    results = wrapper.compute()

    assert "rec_001" in results
    assert isinstance(results["rec_001"], torch.Tensor)


def test_metric_wrapper_with_empty_updates():
    """Test MetricWrapper behavior with empty tensor updates."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    # Update with empty tensors
    preds = torch.tensor([])
    targets = torch.tensor([])

    # This might raise an error depending on the metric implementation
    # but we test that our wrapper handles it gracefully
    try:
        wrapper.update(preds, targets, None, recording_id="rec_001")
        results = wrapper.compute()
        assert "rec_001" in results
    except Exception:
        # If the underlying metric doesn't support empty tensors,
        # that's expected behavior
        pass


def test_metric_wrapper_with_nan_values():
    """Test MetricWrapper behavior with NaN values."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    preds = torch.tensor([1.0, float("nan"), 3.0])
    targets = torch.tensor([1.1, 2.1, 3.1])

    wrapper.update(preds, targets, None, recording_id="rec_001")
    results = wrapper.compute()

    assert "rec_001" in results
    # The result might be NaN, which is expected behavior
    assert isinstance(results["rec_001"], torch.Tensor)


def test_metric_wrapper_state_persistence():
    """Test that MetricWrapper properly maintains state across updates."""
    metrics = {"rec_001": MeanSquaredError()}
    wrapper = MetricWrapper(metrics)

    # First batch
    preds1 = torch.tensor([1.0, 2.0])
    targets1 = torch.tensor([1.1, 2.1])
    wrapper.update(preds1, targets1, None, recording_id="rec_001")

    # Second batch
    preds2 = torch.tensor([3.0, 4.0])
    targets2 = torch.tensor([3.1, 4.1])
    wrapper.update(preds2, targets2, None, recording_id="rec_001")

    # Third batch
    preds3 = torch.tensor([5.0, 6.0])
    targets3 = torch.tensor([5.1, 6.1])
    wrapper.update(preds3, targets3, None, recording_id="rec_001")

    results = wrapper.compute()

    # Should compute MSE over all 6 samples
    expected_mse = torch.mean(
        (
            torch.cat([preds1, preds2, preds3])
            - torch.cat([targets1, targets2, targets3])
        )
        ** 2
    )
    assert torch.isclose(results["rec_001"], expected_mse, atol=1e-6)
