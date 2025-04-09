import logging

import numpy as np
import pytest
from scipy.signal import resample
from temporaldata import ArrayDict, Data, IrregularTimeSeries

from torch_brain.transforms import Resampler

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_data_1():
    timestamps = np.arange(200, dtype=np.float64)
    values = np.arange(200, 400, dtype=np.float64)

    data = Data(
        att1=IrregularTimeSeries(
            timestamps=timestamps,
            values=values,
            timekeys=["timestamps", "values"],
            domain="auto",
        ),
        domain="auto",
    )
    return data


@pytest.fixture
def base_frequency_1():
    return 1.0


@pytest.fixture
def resample_frequency_1():
    return 0.5


def test_downsampling(mock_data_1, base_frequency_1, resample_frequency_1):
    pre_slice_transform = Resampler(base_frequency_1, resample_frequency_1)
    data = pre_slice_transform(mock_data_1, 0.0, 10.0)
    print(data.att1.timestamps)

    expected_timestamps = np.arange(
        0.0, 10.0 + pre_slice_transform.extra_window_length, 1 / resample_frequency_1
    )
    print(expected_timestamps)

    assert np.array_equal(data.att1.timestamps, expected_timestamps)

    expected_values = resample(mock_data_1.att1.values[:15], len(expected_timestamps))
    assert np.array_equal(data.att1.values, expected_values)
