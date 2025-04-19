import copy
import logging

import numpy as np
import pytest
from scipy.signal import decimate
from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries

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
    return Interval(15.0, 30.0)


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
            values=np.arange(100, 200, dtype=np.float64),
            domain=Interval(0.0, 100.0),
        ),
        domain="auto",
    )
    return data


@pytest.fixture
def resampling_factor():
    return 2


@pytest.fixture
def data_2(base_sampling_rate, resampling_factor):
    base_frequency = 1.0 / base_sampling_rate
    reduced_frequency = base_sampling_rate / resampling_factor

    data = Data(
        irregular_att=IrregularTimeSeries(
            timestamps=np.arange(0, 100, base_frequency, dtype=np.float64),
            values=np.arange(100, 200, base_frequency, dtype=np.float64),
            timekeys=["timestamps", "values"],
            domain=Interval(0.0, 100.0),
        ),
        regular_att=RegularTimeSeries(
            sampling_rate=base_sampling_rate,
            values=np.arange(100, 200, base_frequency, dtype=np.float64),
            domain=Interval(0.0, 100.0),
        ),
        irregular_att_2=IrregularTimeSeries(
            timestamps=np.arange(0, 100, reduced_frequency, dtype=np.float64),
            values=np.arange(1000, 1100, reduced_frequency, dtype=np.float64),
            timekeys=["timestamps", "values"],
            domain=Interval(0.0, 100.0),
        ),
        regular_att_2=RegularTimeSeries(
            sampling_rate=base_sampling_rate * resampling_factor,
            values=np.arange(1000, 1100, reduced_frequency, dtype=np.float64),
            domain=Interval(0.0, 100.0),
        ),
        domain="auto",
    )
    return data


def test_downsampling_centered(
    data, base_sampling_rate, downsampling_rate, center_window
):
    # Test on a centered window for resampling all the data object at the same sampling_rate
    start, end = center_window.start[0], center_window.end[0]
    pre_slice_transform = Resampler(
        args=[{"target_key": "irregular_att"}, {"target_key": "regular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), center_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start - buffer, end + buffer, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:-resampled_buffer]

    assert np.array_equal(regular_values, expected_resampled_values)
    assert np.array_equal(irregular_values, expected_resampled_values)


def test_downsampling_left(data, base_sampling_rate, downsampling_rate, left_window):
    # Test on a left window for resampling all the data object at the same sampling_rate
    start, end = left_window.start[0], left_window.end[0]
    pre_slice_transform = Resampler(
        args=[{"target_key": "irregular_att"}, {"target_key": "regular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), left_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start, end + buffer, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[:-resampled_buffer]

    assert np.array_equal(regular_values, expected_resampled_values)
    assert np.array_equal(irregular_values, expected_resampled_values)


def test_downsampling_right(data, base_sampling_rate, downsampling_rate, right_window):
    # Test on a right window for resampling all the data object at the same sampling_rate
    start, end = right_window.start[0], right_window.end[0]
    pre_slice_transform = Resampler(
        args=[{"target_key": "irregular_att"}, {"target_key": "regular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), right_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start - buffer, end, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:]

    assert np.array_equal(regular_values, expected_resampled_values)
    assert np.array_equal(irregular_values, expected_resampled_values)


def test_downsampling_one_attribute_targeted(
    data, base_sampling_rate, downsampling_rate, center_window
):
    # Test on a centered window for resampling one attribute in the data object
    start, end = center_window.start[0], center_window.end[0]

    original_timestamps = data.regular_att.timestamps
    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    pre_slice_transform = Resampler(
        args=[{"target_key": "irregular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), center_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, original_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    expected_original_values = data.regular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start - buffer, end + buffer, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:-resampled_buffer]

    assert np.array_equal(regular_values, expected_original_values)
    assert np.array_equal(irregular_values, expected_resampled_values)

    pre_slice_transform = Resampler(
        args=[{"target_key": "regular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), center_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    assert np.array_equal(irregular_timestamps, original_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    expected_original_values = data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start - buffer, end + buffer, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:-resampled_buffer]

    assert np.array_equal(irregular_values, expected_original_values)
    assert np.array_equal(regular_values, expected_resampled_values)


def test_downsampling_diff_base_sampling_rate(
    data_2, base_sampling_rate, downsampling_rate, center_window, resampling_factor
):
    # Test on a centered window for resampling all the data object at the same sampling_rate
    # while they have different base sampling rates
    start, end = center_window.start[0], center_window.end[0]
    pre_slice_transform = Resampler(
        args=[
            {"target_key": "irregular_att"},
            {"target_key": "regular_att"},
            {"target_key": "irregular_att_2"},
            {"target_key": "regular_att_2"},
        ],
        target_sampling_rate=downsampling_rate,
    )

    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data_2), center_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps
    irregular_timestamps_2 = transformed_data.irregular_att_2.timestamps
    regular_timestamps_2 = transformed_data.regular_att_2.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)
    assert np.array_equal(irregular_timestamps_2, downsampled_timestamps)
    assert np.array_equal(regular_timestamps_2, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values
    regular_values_2 = transformed_data.regular_att_2.values
    irregular_values_2 = transformed_data.irregular_att_2.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data_2.slice(start - buffer, end + buffer, reset_origin=False)

    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:-resampled_buffer]

    x_in_2 = new_data.regular_att_2.values
    downsample_factor_2 = int(
        resampling_factor * base_sampling_rate / downsampling_rate
    )
    x_out_2 = decimate(x_in_2, downsample_factor_2)
    resampled_buffer_2 = int(buffer * downsampling_rate)
    expected_resampled_values_2 = x_out_2[resampled_buffer_2:-resampled_buffer_2]

    assert np.array_equal(regular_values, expected_resampled_values)
    assert np.array_equal(irregular_values, expected_resampled_values)
    assert np.array_equal(regular_values_2, expected_resampled_values_2)
    assert np.array_equal(irregular_values_2, expected_resampled_values_2)


def test_downsampling_diff_target_sampling_rate(
    data, base_sampling_rate, downsampling_rate, center_window, resampling_factor
):
    # Test on a centered window for resampling all the data object at different target sampling rates
    # while they have the same base sampling rates
    start, end = center_window.start[0], center_window.end[0]

    reduced_downsampling_rate = base_sampling_rate / resampling_factor
    pre_slice_transform = Resampler(
        args=[
            {
                "target_key": "irregular_att",
                "target_sampling_rate": downsampling_rate,
            },
            {
                "target_key": "regular_att",
                "target_sampling_rate": reduced_downsampling_rate,
            },
        ]
    )

    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), center_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)
    reduced_downsampled_timestamps = np.arange(
        start, end, step=1 / reduced_downsampling_rate
    )

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, reduced_downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    new_data = data.slice(start - buffer, end + buffer, reset_origin=False)

    x_in = new_data.irregular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    expected_resampled_values = x_out[resampled_buffer:-resampled_buffer]

    reduced_x_in = new_data.regular_att.values
    reduced_downsample_factor = int(base_sampling_rate / reduced_downsampling_rate)
    reduced_x_out = decimate(reduced_x_in, reduced_downsample_factor)
    reduced_resampled_buffer = int(buffer * reduced_downsampling_rate)
    expected_reduced_resampled_values = reduced_x_out[
        reduced_resampled_buffer:-reduced_resampled_buffer
    ]

    assert np.array_equal(irregular_values, expected_resampled_values)
    assert np.array_equal(regular_values, expected_reduced_resampled_values)


def test_downsampling_unaliged(
    data, base_sampling_rate, downsampling_rate, unaligned_window
):
    # Test on a centered window for resampling all the data object at the same sampling_rate
    start, end = unaligned_window.start[0], unaligned_window.end[0]
    pre_slice_transform = Resampler(
        args=[{"target_key": "regular_att"}, {"target_key": "irregular_att"}],
        target_sampling_rate=downsampling_rate,
    )
    # TODO be careful the object is overide
    transformed_data = pre_slice_transform(copy.deepcopy(data), unaligned_window)

    irregular_timestamps = transformed_data.irregular_att.timestamps
    regular_timestamps = transformed_data.regular_att.timestamps

    downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

    assert np.array_equal(irregular_timestamps, downsampled_timestamps)
    assert np.array_equal(regular_timestamps, downsampled_timestamps)

    regular_values = transformed_data.regular_att.values
    irregular_values = transformed_data.irregular_att.values

    buffer = pre_slice_transform.args[0]["anti_aliasing_buffer"]
    offset = start % (1.0 / downsampling_rate)

    new_data = data.slice(offset, end + buffer, reset_origin=False)
    x_in = new_data.regular_att.values
    downsample_factor = int(base_sampling_rate / downsampling_rate)
    x_out = decimate(x_in, downsample_factor)
    resampled_buffer = int(buffer * downsampling_rate)
    resampled_offfset_buffer = int((start - offset) * downsampling_rate)
    expected_resampled_values = x_out[resampled_offfset_buffer:-resampled_buffer]

    assert np.array_equal(regular_values, expected_resampled_values)
    assert np.array_equal(irregular_values, expected_resampled_values)
