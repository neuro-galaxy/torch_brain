import logging

import numpy as np
import pytest
from scipy.signal import resample
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
def base_frequency():
    return 1.0


@pytest.fixture
def downsampling_frequency():
    return 0.5


@pytest.fixture
def upsampling_frequency():
    return 2.0


@pytest.fixture
def center_window():
    return {"start": 20.0, "end": 30.0}


@pytest.fixture
def right_window():
    return {"start": 0.0, "end": 10.0}


@pytest.fixture
def left_window():
    return {"start": 40.0, "end": 50.0}


@pytest.fixture
def irregular_data():
    data = Data(
        att1=IrregularTimeSeries(
            timestamps=np.arange(50, dtype=np.float64),
            values=np.arange(100, 150, dtype=np.float64),
            timekeys=["timestamps", "values"],
            domain="auto",
        ),
        domain="auto",
    )
    return data


@pytest.fixture
def regular_data(base_frequency):
    data = Data(
        att1=RegularTimeSeries(
            sampling_rate=base_frequency,
            domain=Interval(0.0, 50.0),
            values=np.arange(100, 150, dtype=np.float64),
        ),
        domain="auto",
    )
    return data


def test_downsampling_irregular(
    irregular_data,
    base_frequency,
    downsampling_frequency,
    center_window,
    right_window,
    left_window,
):
    # 1 case: centered window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = center_window["start"]
    end = center_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length / 2)
    window_end = int(end + extra_window_length / 2)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency / 2)
    assert np.array_equal(data.att1.timestamps, t_out[N:-N])
    assert np.array_equal(data.att1.values, x_out[N:-N])

    # 2 case: right window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = right_window["start"]
    end = right_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start)
    window_end = int(end + extra_window_length)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[:-N])
    assert np.array_equal(data.att1.values, x_out[:-N])

    # 3 case: left window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = left_window["start"]
    end = left_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length)
    window_end = int(end)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[N:])
    assert np.array_equal(data.att1.values, x_out[N:])


def test_upsampling_irregular(
    irregular_data,
    base_frequency,
    upsampling_frequency,
    center_window,
    right_window,
    left_window,
):
    # 1 case: centered window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = center_window["start"]
    end = center_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length / 2)
    window_end = int(end + extra_window_length / 2)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency / 2)
    assert np.array_equal(data.att1.timestamps, t_out[N:-N])
    assert np.array_equal(data.att1.values, x_out[N:-N])

    # 2 case: right window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = right_window["start"]
    end = right_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start)
    window_end = int(end + extra_window_length)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[:-N])
    assert np.array_equal(data.att1.values, x_out[:-N])

    # 3 case: left window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = left_window["start"]
    end = left_window["end"]
    data = pre_slice_transform(irregular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length)
    window_end = int(end)
    x_in = irregular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = irregular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[N:])
    assert np.array_equal(data.att1.values, x_out[N:])


def test_downsampling_regular(
    regular_data,
    base_frequency,
    downsampling_frequency,
    center_window,
    right_window,
    left_window,
):
    # 1 case: centered window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = center_window["start"]
    end = center_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length / 2)
    window_end = int(end + extra_window_length / 2)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency / 2)
    assert np.array_equal(data.att1.timestamps, t_out[N:-N])
    assert np.array_equal(data.att1.values, x_out[N:-N])

    # 2 case: right window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = right_window["start"]
    end = right_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start)
    window_end = int(end + extra_window_length)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[:-N])
    assert np.array_equal(data.att1.values, x_out[:-N])

    # 3 case: left window
    pre_slice_transform = Resampler(base_frequency, downsampling_frequency)
    start = left_window["start"]
    end = left_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length)
    window_end = int(end)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * downsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * downsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[N:])
    assert np.array_equal(data.att1.values, x_out[N:])


def test_upsampling_regular(
    regular_data,
    base_frequency,
    upsampling_frequency,
    center_window,
    right_window,
    left_window,
):
    # 1 case: centered window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = center_window["start"]
    end = center_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length / 2)
    window_end = int(end + extra_window_length / 2)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency / 2)
    assert np.array_equal(data.att1.timestamps, t_out[N:-N])
    assert np.array_equal(data.att1.values, x_out[N:-N])

    # 2 case: right window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = right_window["start"]
    end = right_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start)
    window_end = int(end + extra_window_length)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[:-N])
    assert np.array_equal(data.att1.values, x_out[:-N])

    # 3 case: left window
    pre_slice_transform = Resampler(base_frequency, upsampling_frequency)
    start = left_window["start"]
    end = left_window["end"]
    data = pre_slice_transform(regular_data, start, end)

    extra_window_length = pre_slice_transform.extra_window_length
    window_start = int(start - extra_window_length)
    window_end = int(end)
    x_in = regular_data.att1.values[window_start:window_end]
    num = int((window_end - window_start) * upsampling_frequency)
    t_in = regular_data.att1.timestamps[window_start:window_end]
    x_out, t_out = resample(x_in, num, t=t_in, window="hamming")

    N = int(extra_window_length * upsampling_frequency)
    assert np.array_equal(data.att1.timestamps, t_out[N:])
    assert np.array_equal(data.att1.values, x_out[N:])
