import logging

import numpy as np
import pytest
from temporaldata import ArrayDict, Data, IrregularTimeSeries

from torch_brain.transforms.unit_filter import UnitFilter, UnitFilterByAttr

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
            brain_region=np.array(
                [
                    "M1",
                    "S1",
                    "M1",
                    "V1",
                    "S1",
                    "M1",
                ]
            ),
        ),
        domain="auto",
    )
    return data


@pytest.fixture
def index_mock_data():
    return Data(
        spikes=IrregularTimeSeries(
            timestamps=np.arange(8),
            unit_index=np.array([4, 1, 4, 1, 4, 1, 4, 1]),
            types=np.zeros(8),
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["u0", "u1", "u2", "u3", "u4", "u5"]),
            brain_region=np.array(["A", "M1", "B", "C", "M1", "D"]),
        ),
        domain="auto",
    )


def test_unit_filter_w_keyword(mock_data: Data):
    # keyword pattern keeps units whose ids contain "unsorted".
    transform = UnitFilterByAttr(
        target_attr="spikes", pattern="unsorted", reset_index=True, keep_matches=True
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
    # regex pattern keeps only units whose ids match the sorted prefix.
    transform = UnitFilterByAttr(
        target_attr="spikes", pattern=r"^sorted_.*", reset_index=True, keep_matches=True
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


def test_unit_filter_by_brain_region(mock_data: Data):
    # filtering by brain region keeps only matching units and aligned spikes.
    transform = UnitFilterByAttr(
        target_attr="spikes",
        pattern="M1",
        filter_attr="brain_region",
        reset_index=True,
        keep_matches=True,
    )
    data_t = transform(mock_data)

    expected_unit_ids = ["sorted_a", "sorted_b", "unsorted_c"]
    assert np.array_equal(data_t.units.id, expected_unit_ids)

    expected_brain_regions = ["M1", "M1", "M1"]
    assert np.array_equal(data_t.units.brain_region, expected_brain_regions)

    expected_unit_index = [0] * 10 + [1] * 70 + [2] * 70
    assert np.array_equal(data_t.spikes.unit_index, expected_unit_index)

    expected_timestamps = np.concatenate(
        [np.arange(10), np.arange(30, 100), np.arange(130, 200)]
    )
    assert np.array_equal(data_t.spikes.timestamps, expected_timestamps)


def test_unit_filter_reset_index_true_deterministic_remap(index_mock_data: Data):
    # reset-index remapping should be stable across independent sliced views.
    transform = UnitFilterByAttr(
        target_attr="spikes",
        pattern="M1",
        filter_attr="brain_region",
        reset_index=True,
        keep_matches=True,
    )
    data_view_1 = index_mock_data.slice(0, 4, reset_origin=False)
    data_view_2 = index_mock_data.slice(4, 8, reset_origin=False)

    data_t_1 = transform(data_view_1)
    data_t_2 = transform(
        data_view_2,
    )

    # Kept unit order follows original unit index order: [1, 4].
    assert np.array_equal(data_t_1.units.id, ["u1", "u4"])
    assert np.array_equal(data_t_2.units.id, ["u1", "u4"])
    assert np.array_equal(data_t_1.spikes.unit_index, [1, 0, 1, 0])
    assert np.array_equal(data_t_2.spikes.unit_index, [1, 0, 1, 0])
    assert np.array_equal(data_t_1.spikes.timestamps, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(data_t_2.spikes.timestamps, [4.0, 5.0, 6.0, 7.0])


def test_unit_filter_reset_index_false_keeps_original_unit_index(index_mock_data: Data):
    # without reset-index, spike indices stay in the original unit-id space.
    transform = UnitFilterByAttr(
        target_attr="spikes",
        pattern="M1",
        filter_attr="brain_region",
        reset_index=False,
        keep_matches=True,
    )
    data_t = transform(index_mock_data)

    # Without reset, units table stays unchanged and spike indices are not remapped.
    assert np.array_equal(data_t.units.id, ["u0", "u1", "u2", "u3", "u4", "u5"])
    assert np.array_equal(data_t.spikes.unit_index, [4, 1, 4, 1, 4, 1, 4, 1])
    assert np.array_equal(
        data_t.spikes.timestamps, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )
