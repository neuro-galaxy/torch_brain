import logging

import numpy as np
import pytest
from scipy.signal import decimate, resample
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

from torch_brain.transforms import Resampler

logger = logging.getLogger(__name__)


@pytest.fixture
def base_sampling_rate():
    return 1.0


@pytest.fixture
def downsampling_rate():
    return 0.5


@pytest.fixture
def upsampling_rate():
    return 2.0


@pytest.fixture
def center_window():
    return Interval(40.0, 60.0)


@pytest.fixture
def left_window():
    return Interval(0.0, 30.0)


@pytest.fixture
def right_window():
    return Interval(90.0, 100.0)


@pytest.fixture
def unaligned_window():
    return Interval(60.5, 70.0)


# TODO if domain="auto for IrregularTimeSeries" causes issues
@pytest.fixture
def data(base_sampling_rate):
    data = Data(
        irregular_att=IrregularTimeSeries(
            timestamps=np.arange(100, dtype=np.float64),
            values=np.arange(100, 200, dtype=np.float64),
            timekeys=["timestamps", "values"],
            domain=Interval(0.0, 100.0),
        ),
        regular_att=RegularTimeSeries(
            sampling_rate=base_sampling_rate,
            domain=Interval(0.0, 100.0),
            values=np.arange(100, 200, dtype=np.float64),
        ),
        domain="auto",
    )
    return data


def dflt_resample_fn(x, downsample_factor):
    return decimate(x, downsample_factor, axis=0, ftype="iir")


def test_downsampling_centered_regular(
    data, base_sampling_rate, downsampling_rate, center_window
):
    # Test on a centered window for regular data
    pre_slice_transform = Resampler(
        target_sampling_rate=downsampling_rate, target_keys=["regular_att"]
    )
    transformed_data = pre_slice_transform(data, center_window)

    timestamps = transformed_data.regular_att.timestamps

    expected_timestamps = np.arange(
        center_window.start[0],
        center_window.end[0],
        step=1 / downsampling_rate,
    )
    assert np.array_equal(timestamps, expected_timestamps)

    values = transformed_data.regular_att.values

    buffer = pre_slice_transform._anti_aliasing_buffer
    window_start = int(center_window.start[0] - buffer)
    window_end = int(center_window.end[0] + buffer)
    x_in = data.regular_att.values[window_start:window_end]
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = dflt_resample_fn(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_values = x_out[resampled_buffer:-resampled_buffer]

    assert np.array_equal(values, expected_values)


def test_downsampling_left_irregular(
    data, base_sampling_rate, downsampling_rate, left_window
):
    # Test for left window for irregular data
    pre_slice_transform = Resampler(
        target_sampling_rate=downsampling_rate, target_keys=["irregular_att"]
    )
    transformed_data = pre_slice_transform(data, left_window)

    timestamps = transformed_data.irregular_att.timestamps

    expected_timestamps = np.arange(
        left_window.start[0],
        left_window.end[0],
        step=1 / downsampling_rate,
    )
    assert np.array_equal(timestamps, expected_timestamps)

    values = transformed_data.irregular_att.values

    buffer = pre_slice_transform._anti_aliasing_buffer
    window_start = int(left_window.start[0])
    window_end = int(left_window.end[0] + buffer)
    x_in = data.regular_att.values[window_start:window_end]
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = dflt_resample_fn(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_values = x_out[:-resampled_buffer]

    assert np.array_equal(values, expected_values)


def test_downsampling_right_regular_and_irregular(
    data, base_sampling_rate, downsampling_rate, right_window
):
    # Test or tight window for regular and irregular data
    pre_slice_transform = Resampler(
        target_sampling_rate=downsampling_rate,
        target_keys=["regular_att", "irregular_att"],
    )
    transformed_data = pre_slice_transform(data, right_window)

    regular_timestamps = transformed_data.regular_att.timestamps
    irregular_timestamps = transformed_data.irregular_att.timestamps

    expected_timestamps = np.arange(
        right_window.start[0],
        right_window.end[0],
        step=1 / downsampling_rate,
    )

    assert np.array_equal(regular_timestamps, expected_timestamps)
    assert np.array_equal(irregular_timestamps, expected_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform._anti_aliasing_buffer
    window_start = int(right_window.start[0] - buffer)
    window_end = int(right_window.end[0])
    x_in = data.regular_att.values[window_start:window_end]
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = dflt_resample_fn(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_values = x_out[resampled_buffer:]

    assert np.array_equal(regular_values, expected_values)
    assert np.array_equal(irregular_values, expected_values)


def test_downsampling_unaliged(
    data, base_sampling_rate, downsampling_rate, unaligned_window
):
    # Test for unaligned window for irregular data
    pre_slice_transform = Resampler(
        target_sampling_rate=downsampling_rate,
        target_keys=["irregular_att", "regular_att"],
    )
    transformed_data = pre_slice_transform(data, unaligned_window)

    regular_timestamps = transformed_data.regular_att.timestamps
    irregular_timestamps = transformed_data.irregular_att.timestamps

    expected_timestamps = np.arange(
        unaligned_window.start[0],
        unaligned_window.end[0],
        step=1 / downsampling_rate,
    )
    offset = unaligned_window.start[0] % (1 / downsampling_rate)
    expected_timestamps -= offset

    assert np.array_equal(regular_timestamps, expected_timestamps)
    assert np.array_equal(irregular_timestamps, expected_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform._anti_aliasing_buffer
    window_start = int(unaligned_window.start[0] - buffer)
    window_end = int(unaligned_window.end[0] + buffer)
    x_in = data.regular_att.values[window_start:window_end]
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = dflt_resample_fn(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_values = x_out[resampled_buffer:-resampled_buffer]

    assert np.array_equal(regular_values, expected_values)
    assert np.array_equal(irregular_values, expected_values)
