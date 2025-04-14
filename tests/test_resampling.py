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
def right_window():
    return Interval(0.0, 10.0)


@pytest.fixture
def left_window():
    return Interval(90.0, 100.0)


@pytest.fixture
def data(base_sampling_rate):
    data = Data(
        irregular_att=IrregularTimeSeries(
            timestamps=np.arange(100, dtype=np.float64),
            values=np.arange(100, 200, dtype=np.float64),
            timekeys=["timestamps", "values"],
            domain="auto",
        ),
        regular_att=RegularTimeSeries(
            sampling_rate=base_sampling_rate,
            domain=Interval(0.0, 100.0),
            values=np.arange(100, 200, dtype=np.float64),
        ),
        domain="auto",
    )
    return data


def test_downsampling(
    data,
    downsampling_rate,
    center_window,
    right_window,
    left_window,
):
    # 1 case: centered window
    pre_slice_transform = Resampler(
        target_sampling_rate=downsampling_rate, target_keys=["regular_att"]
    )
    N = pre_slice_transform._anti_aliasing_buffer
    window_start = int(center_window.start[0] - N)
    window_end = int(center_window.end[0] + N)

    # t_out = down_sampling_rate

    x_in = data.regular_att.values[window_start:window_end]
    downsample_factor = int(data.regular_att.sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor, axis=0, ftype="iir")

    # assert np.array_equal(data.att1.timestamps, t_out[N:-N])
    transformed_data = pre_slice_transform(data, center_window)
    assert np.array_equal(transformed_data.regular_att.values, x_out[10:-10])

    # # 2 case: right window
    # pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    # start = right_window["start"]
    # end = right_window["end"]
    # data = pre_slice_transform(irregular_data, start, end)

    # extra_window_length = pre_slice_transform.extra_window_length
    # window_start = int(start)
    # window_end = int(end + extra_window_length)
    # x_in = irregular_data.att1.values[window_start:window_end]
    # num = int((window_end - window_start) * downsampling_frequency)
    # t_in = irregular_data.att1.timestamps[window_start:window_end]
    # x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    # N = int(extra_window_length * downsampling_frequency)
    # assert np.array_equal(data.att1.timestamps, t_out[:-N])
    # assert np.array_equal(data.att1.values, x_out[:-N])

    # # 3 case: left window
    # pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    # start = left_window["start"]
    # end = left_window["end"]
    # data = pre_slice_transform(irregular_data, start, end)

    # extra_window_length = pre_slice_transform.extra_window_length
    # window_start = int(start - extra_window_length)
    # window_end = int(end)
    # x_in = irregular_data.att1.values[window_start:window_end]
    # num = int((window_end - window_start) * downsampling_frequency)
    # t_in = irregular_data.att1.timestamps[window_start:window_end]
    # x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    # N = int(extra_window_length * downsampling_frequency)
    # assert np.array_equal(data.att1.timestamps, t_out[N:])
    # assert np.array_equal(data.att1.values, x_out[N:])


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
