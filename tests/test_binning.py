import numpy as np
import pytest
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from torch_brain.transforms.binning import BinSpikes
from torch_brain.utils.binning import bin_spikes


def test_bin_data():
    spikes = IrregularTimeSeries(
        timestamps=np.array([0, 0, 1, 1, 2, 3, 4, 4.5, 6, 7, 8, 9]),
        unit_index=np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        domain=Interval(0, 10),
    )
    binned_data = bin_spikes(spikes, num_units=2, bin_size=1.0, right=True)

    expected = np.array(
        [[1, 1, 1, 1, 2, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)
    assert binned_data.dtype == np.float32

    # test with np.int32
    binned_data = bin_spikes(
        spikes, num_units=2, bin_size=1.0, right=True, dtype=np.int32
    )
    assert binned_data.dtype == np.int32

    # larger bin size
    spikes = IrregularTimeSeries(
        timestamps=np.array([0, 0, 0.34, 2, 2.1, 3, 4, 4.1, 6, 7, 8, 9]),
        unit_index=np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        domain=Interval(0, 10),
    )
    binned_data = bin_spikes(spikes, num_units=2, bin_size=3.0, right=True)

    expected = np.array([[2.0, 3.0, 3.0], [1.0, 0.0, 0.0]])
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)

    # bin size 2.5
    binned_data = bin_spikes(spikes, num_units=2, bin_size=2.5, right=True)

    expected = np.array([[3.0, 3.0, 2.0, 2.0], [2.0, 0.0, 0.0, 0.0]])
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)

    # align to the left
    binned_data = bin_spikes(spikes, num_units=2, bin_size=3.0, right=False)
    expected = np.array([[3, 3, 3], [2, 0, 0]])

    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)

    # multiple units
    spikes = IrregularTimeSeries(
        timestamps=np.array(
            [0.01, 0.015, 0.1, 0.2, 0.3, 0.35, 0.5, 0.61, 0.7, 0.83, 0.91]
        ),
        unit_index=np.array([0, 1, 2, 3, 2, 2, 1, 0, 0, 1, 2]),
        domain=Interval(0.0, 1.0),
    )
    binned_data = bin_spikes(spikes, num_units=4, bin_size=0.1, right=True)

    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)

    # test max_spikes
    spikes = IrregularTimeSeries(
        timestamps=np.array(
            [0, 0, 1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 4.5, 6, 7, 8, 9]
        ),
        unit_index=np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        domain=Interval(0, 10),
    )
    binned_data = bin_spikes(
        spikes, num_units=2, bin_size=1.0, right=True, max_spikes=3
    )

    expected = np.array(
        [[1, 1, 1, 1, 2, 0, 1, 1, 1, 1], [1, 3, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)
    assert binned_data.dtype == np.float32

    # fix numerical instability
    # Duration is intended to be exactly 1.0, but represented with
    # floating-point error.
    for base in [0.0, 1e3, 1e6]:
        ts = base + np.array(
            [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9999999]
        )
        spikes = IrregularTimeSeries(
            timestamps=ts, unit_index=np.zeros(10, dtype=int), domain="auto"
        )

        binned_data = bin_spikes(spikes, num_units=1, bin_size=0.1)

        expected = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        assert binned_data.shape == expected.shape
        assert np.allclose(binned_data, expected)


@pytest.fixture
def simple_spikes_data():
    """Creates a simple 2-unit dataset for binning verification."""
    timestamps = np.array([0.5, 1.5, 2.5, 0.5, 0.6])
    unit_index = np.array([0, 0, 0, 1, 1])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain=Interval(0, 3),
        ),
        units=ArrayDict(
            id=np.array(["unit_a", "unit_b"]),
        ),
        domain=Interval(0, 3),
    )
    return data


def test_binning_transform_basic(simple_spikes_data):
    bin_size = 1.0
    transform = BinSpikes(bin_size=bin_size)

    data_t = transform(simple_spikes_data)

    # Check if the new attribute was created
    assert hasattr(data_t, "spikes_binned")

    # Verify the spikes_binned created
    expected_binned = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    assert np.array_equal(data_t.spikes_binned, expected_binned)


def test_binning_transform_custom_attr_names(simple_spikes_data):
    # Test that it respects different attribute names
    # (e.g., if spikes are under 'lfp_spikes' instead of 'spikes')
    simple_spikes_data.lfp_spikes = simple_spikes_data.spikes

    transform = BinSpikes(
        spikes_attribute="lfp_spikes", units_attribute="units", bin_size=1.0
    )

    data_t = transform(simple_spikes_data)

    assert hasattr(data_t, "lfp_spikes_binned")
    assert data_t.lfp_spikes_binned.shape == (2, 3)
