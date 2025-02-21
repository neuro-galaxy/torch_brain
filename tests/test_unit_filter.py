import logging

import numpy as np
import pytest

from temporaldata import ArrayDict, Data, IrregularTimeSeries
from torch_brain.transforms.unit_filter import UnitFilter, UnitFilterById

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_data():
    timestamps = np.arange(200)
    unit_index = [0] * 10 + [1] * 20 + [2] * 70 + [3] * 10 + [4] * 20 + [5] * 70
    unit_index = np.array(unit_index)
    types = np.zeros(200)
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(
                [
                    "sorted_a",
                    "unsorted_a",
                    "sorted_b",
                    "unsorted_b",
                    "sorted_c",
                    "unsorted_c",
                ]
            ),
        ),
        domain="auto",
    )
    return data


def test_unit_filter_w_keyword(mock_data):
    transform = UnitFilterById(
        "unsorted", field="spikes", reset_index=True, keep_matches=True
    )
    data_t = transform(mock_data)

    expected_unit_ids = ["unsorted_a", "unsorted_b", "unsorted_c"]
    assert np.array_equal(data_t.units.id, expected_unit_ids)

    expected_unit_index = [0] * 20 + [1] * 10 + [2] * 70
    assert np.array_equal(data_t.spikes.unit_index, expected_unit_index)

    expected_timestamps = np.concatenate(
        [np.arange(10, 30), np.arange(100, 110), np.arange(130, 200)]
    )
    assert np.array_equal(data_t.spikes.timestamps, expected_timestamps)


def test_unit_filter_w_regex(mock_data):
    transform = UnitFilterById(
        r"^sorted_.*", field="spikes", reset_index=True, keep_matches=True
    )
    data_t = transform(mock_data)

    expected_unit_ids = ["sorted_a", "sorted_b", "sorted_c"]
    assert np.array_equal(data_t.units.id, expected_unit_ids)

    expected_unit_index = [0] * 10 + [1] * 70 + [2] * 20
    assert np.array_equal(data_t.spikes.unit_index, expected_unit_index)

    expected_timestamps = np.concatenate(
        [np.arange(10), np.arange(30, 100), np.arange(110, 130)]
    )
    assert np.array_equal(data_t.spikes.timestamps, expected_timestamps)
