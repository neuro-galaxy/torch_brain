import numpy as np
import pytest
from temporaldata import Interval, IrregularTimeSeries

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
    print(binned_data)
    assert binned_data.shape == expected.shape
    assert np.allclose(binned_data, expected)

    # fix numerical instability
    # Duration is intended to be exactly 1.0, but represented with
    # floating-point error.
    # for base in [0.0, 1e3, 1e6]:
    for base in [0.0]:
        ts = base + np.array(
            [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9999999]
        )
        spikes = IrregularTimeSeries(
            timestamps=ts, unit_index=np.zeros(10, dtype=int), domain="auto"
        )

        binned_data = bin_spikes(
            spikes,
            num_units=1,
            bin_size=0.1,
            right=True,
        )

        expected = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        assert binned_data.shape == expected.shape
        assert np.allclose(binned_data, expected)
