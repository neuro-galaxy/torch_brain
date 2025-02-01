import numpy as np
import pytest
from temporaldata import Data, Interval

from torch_brain.utils.weights import resolve_weights_based_on_interval_membership


def MockData():
    data = Data(
        movement_periods=Data(
            reach_period=Interval(start=np.array([2.0]), end=np.array([4.0])),
            hold_period=Interval(start=np.array([3.0]), end=np.array([5.0])),
            domain="auto",
        ),
        cursor_outlier_segments=Interval(start=np.array([3.5]), end=np.array([4.5])),
        empty_interval=Interval(start=np.array([]), end=np.array([])),
        out_of_bounds_interval=Interval(start=np.array([6.0]), end=np.array([7.0])),
        invalid_attribute="not_an_interval",
        domain=Interval(start=np.array([0.0]), end=np.array([10.0])),
    )

    return data


def test_default_weights():
    """Test that default weights (no config) are all ones."""
    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = MockData()

    weights = resolve_weights_based_on_interval_membership(timestamps, data)
    assert np.all(weights == 1.0)
    assert weights.shape == timestamps.shape


def test_single_interval_weights():
    """Test weights with a single interval."""
    timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = MockData()
    config = {"movement_periods.reach_period": 5.0}

    weights = resolve_weights_based_on_interval_membership(timestamps, data, config)
    expected = np.array([1.0, 5.0, 5.0, 1.0, 1.0])
    np.testing.assert_array_equal(weights, expected)


def test_overlapping_intervals():
    """Test weights with overlapping intervals."""
    timestamps = np.array([1.0, 2.0, 3.0, 3.75, 4.75])
    data = MockData()
    config = {
        "movement_periods.reach_period": 2.0,
        "movement_periods.hold_period": 3.0,
        "cursor_outlier_segments": 0.0,
    }

    weights = resolve_weights_based_on_interval_membership(timestamps, data, config)
    # At t=3.75, all three intervals overlap: 2.0 * 3.0 * 0.0 = 0.0
    expected = np.array([1.0, 2.0, 6.0, 0.0, 3.0])
    np.testing.assert_array_equal(weights, expected)


def test_empty_interval():
    """Test weights with an empty interval."""
    timestamps = np.array([1.0, 2.0, 3.0])
    data = MockData()
    config = {"empty_interval": 2.0}

    weights = resolve_weights_based_on_interval_membership(timestamps, data, config)
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(weights, expected)


def test_out_of_bounds_interval():
    """Test weights with an out of bounds interval."""
    timestamps = np.array([1.0, 2.0, 3.0])
    data = MockData()
    config = {"out_of_bounds_interval": 2.0}

    weights = resolve_weights_based_on_interval_membership(timestamps, data, config)
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(weights, expected)


def test_invalid_interval_type():
    """Test that an error is raised for invalid interval type."""
    timestamps = np.array([1.0, 2.0, 3.0])
    data = MockData()
    config = {"invalid_attribute": 2.0}

    with pytest.raises(ValueError, match="Expected an Interval object"):
        resolve_weights_based_on_interval_membership(timestamps, data, config)
