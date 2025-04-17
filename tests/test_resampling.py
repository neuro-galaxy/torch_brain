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

    # def __init__(
    #     self,
    #     args: List[Dict[str, Any]],
    #     *,
    #     target_sampling_rate: float,
    #     method: Optional[str] = "decimate",
    # ):


def test_downsampling_centered_regular(
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


# def test_downsampling_left_irregular(
#     data, base_sampling_rate, downsampling_rate, left_window
# ):
#     # Test on a left window for resampling irregular data
#     start, end = left_window.start[0], left_window.end[0]
#     pre_slice_transform = Resampler(
#         target_sampling_rate=downsampling_rate, target_keys=["irregular_att"]
#     )
#     transformed_data = pre_slice_transform(data, left_window)

#     regular_timestamps = transformed_data.regular_att.timestamps
#     irregular_timestamps = transformed_data.irregular_att.timestamps

#     original_timestamps = np.arange(start, end, step=1 / base_sampling_rate)
#     downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

#     assert np.array_equal(irregular_timestamps, downsampled_timestamps)
#     assert np.array_equal(regular_timestamps, original_timestamps)

#     values = transformed_data.irregular_att.values

#     buffer = pre_slice_transform._anti_aliasing_buffer
#     window_start, window_end = int(start), int(end + buffer)
#     x_in = data.regular_att.values[window_start:window_end]
#     downsample_factor = int(base_sampling_rate / downsampling_rate)
#     x_out = pre_slice_transform.dflt_resample_fn(x_in, downsample_factor)
#     resampled_buffer = int(buffer * downsampling_rate)
#     expected_values = x_out[:-resampled_buffer]

#     assert np.array_equal(values, expected_values)


# def test_downsampling_right_regular_and_irregular(
#     data, base_sampling_rate, downsampling_rate, right_window
# ):
#     # Test on a right window for resampling irregular and irregular data
#     start, end = right_window.start[0], right_window.end[0]
#     pre_slice_transform = Resampler(
#         target_sampling_rate=downsampling_rate,
#         target_keys=["regular_att", "irregular_att"],
#     )
#     transformed_data = pre_slice_transform(data, right_window)

#     regular_timestamps = transformed_data.regular_att.timestamps
#     irregular_timestamps = transformed_data.regular_att.timestamps

#     downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)

#     assert np.array_equal(irregular_timestamps, downsampled_timestamps)
#     assert np.array_equal(regular_timestamps, downsampled_timestamps)

#     regular_values = transformed_data.regular_att.values
#     irregular_values = transformed_data.irregular_att.values

#     buffer = pre_slice_transform._anti_aliasing_buffer
#     window_start = int(right_window.start[0] - buffer)
#     window_end = int(right_window.end[0])
#     x_in = data.regular_att.values[window_start:window_end]
#     downsample_factor = int(base_sampling_rate / downsampling_rate)
#     x_out = pre_slice_transform.dflt_resample_fn(x_in, downsample_factor)
#     resampled_buffer = int(buffer * downsampling_rate)
#     expected_values = x_out[resampled_buffer:]

#     assert np.array_equal(regular_values, expected_values)
#     assert np.array_equal(irregular_values, expected_values)


# def test_downsampling_unaliged(
#     data, base_sampling_rate, downsampling_rate, unaligned_window
# ):
#     # Test for window that need to be aligned
#     start, end = unaligned_window.start[0], unaligned_window.end[0]
#     pre_slice_transform = Resampler(
#         target_sampling_rate=downsampling_rate,
#         target_keys=["irregular_att", "regular_att"],
#         shouldAlign=True,
#     )
#     transformed_data = pre_slice_transform(data, unaligned_window)

#     regular_timestamps = transformed_data.regular_att.timestamps
#     irregular_timestamps = transformed_data.irregular_att.timestamps

#     downsampled_timestamps = np.arange(start, end, step=1 / downsampling_rate)
#     offset = start % (1 / downsampling_rate)
#     aligned_downsampled_timestamps = downsampled_timestamps - offset

#     assert np.array_equal(regular_timestamps, aligned_downsampled_timestamps)
#     assert np.array_equal(irregular_timestamps, aligned_downsampled_timestamps)

#     regular_values = transformed_data.regular_att.values
#     irregular_values = transformed_data.irregular_att.values

#     buffer = pre_slice_transform._anti_aliasing_buffer
#     window_start, window_end = int(start - buffer), int(end + buffer)
#     x_in = data.regular_att.values[window_start:window_end]
#     downsample_factor = int(base_sampling_rate / downsampling_rate)
#     x_out = pre_slice_transform.dflt_resample_fn(x_in, downsample_factor)
#     resampled_buffer = int(buffer * downsampling_rate)
#     expected_values = x_out[resampled_buffer:-resampled_buffer]

#     assert np.array_equal(regular_values, expected_values)
#     assert np.array_equal(irregular_values, expected_values)
