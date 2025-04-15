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


def test_downsampling(
    data,
    base_sampling_rate,
    downsampling_rate,
    center_window,
    left_window,
    right_window,
):
    # 1 case: centered window on regular data
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

    # 2 case: left window on irregular data
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

    # 3 case: left window on regular and irregular data
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


# def test_upsampling_irregular(
#     irregular_data,
#     base_frequency,
#     upsampling_frequency,
#     center_window,
#     right_window,
#     left_window,
# ):
#     # 1 case: centered window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = center_window["start"]
#     end = center_window["end"]
#     data = pre_slice_transform(irregular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length / 2)
#     window_end = int(end + extra_window_length / 2)
#     x_in = irregular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = irregular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency / 2)
#     assert np.array_equal(data.att1.timestamps, t_out[N:-N])
#     assert np.array_equal(data.att1.values, x_out[N:-N])

#     # 2 case: right window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = right_window["start"]
#     end = right_window["end"]
#     data = pre_slice_transform(irregular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start)
#     window_end = int(end + extra_window_length)
#     x_in = irregular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = irregular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[:-N])
#     assert np.array_equal(data.att1.values, x_out[:-N])

#     # 3 case: left window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = left_window["start"]
#     end = left_window["end"]
#     data = pre_slice_transform(irregular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length)
#     window_end = int(end)
#     x_in = irregular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = irregular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[N:])
#     assert np.array_equal(data.att1.values, x_out[N:])


# def test_downsampling_regular(
#     regular_data,
#     base_frequency,
#     downsampling_frequency,
#     center_window,
#     right_window,
#     left_window,
# ):
#     # 1 case: centered window
#     pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
#     start = center_window["start"]
#     end = center_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length / 2)
#     window_end = int(end + extra_window_length / 2)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * downsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * downsampling_frequency / 2)
#     assert np.array_equal(data.att1.timestamps, t_out[N:-N])
#     assert np.array_equal(data.att1.values, x_out[N:-N])

#     # 2 case: right window
#     pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
#     start = right_window["start"]
#     end = right_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start)
#     window_end = int(end + extra_window_length)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * downsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * downsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[:-N])
#     assert np.array_equal(data.att1.values, x_out[:-N])

#     # 3 case: left window
#     pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
#     start = left_window["start"]
#     end = left_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length)
#     window_end = int(end)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * downsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * downsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[N:])
#     assert np.array_equal(data.att1.values, x_out[N:])


# def test_upsampling_regular(
#     regular_data,
#     base_frequency,
#     upsampling_frequency,
#     center_window,
#     right_window,
#     left_window,
# ):
#     # 1 case: centered window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = center_window["start"]
#     end = center_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length / 2)
#     window_end = int(end + extra_window_length / 2)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency / 2)
#     assert np.array_equal(data.att1.timestamps, t_out[N:-N])
#     assert np.array_equal(data.att1.values, x_out[N:-N])

#     # 2 case: right window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = right_window["start"]
#     end = right_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start)
#     window_end = int(end + extra_window_length)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[:-N])
#     assert np.array_equal(data.att1.values, x_out[:-N])

#     # 3 case: left window
#     pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
#     start = left_window["start"]
#     end = left_window["end"]
#     data = pre_slice_transform(regular_data, start, end)

#     extra_window_length = pre_slice_transform.extra_window_length
#     window_start = int(start - extra_window_length)
#     window_end = int(end)
#     x_in = regular_data.att1.values[window_start:window_end]
#     num = int((window_end - window_start) * upsampling_frequency)
#     t_in = regular_data.att1.timestamps[window_start:window_end]
#     x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

#     N = int(extra_window_length * upsampling_frequency)
#     assert np.array_equal(data.att1.timestamps, t_out[N:])
#     assert np.array_equal(data.att1.values, x_out[N:])
