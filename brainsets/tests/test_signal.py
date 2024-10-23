import numpy as np
import pytest
from brainsets.processing.signal import (
    extract_bands,
    cube_to_long,
)


@pytest.mark.skipif(pytest.importorskip("mne") is None, reason="mne not installed")
def test_extract_bands():
    bands, ts, band_names = extract_bands(
        np.random.randn(1000, 10), np.arange(1000), Fs=1000, notch=60
    )
    assert bands.ndim == 3
    assert ts.size == bands.shape[0]
    assert bands.shape[2] == len(band_names)


@pytest.mark.skipif(pytest.importorskip("mne") is None, reason="mne not installed")
def test_cube_to_long():
    ts = np.arange(10).astype(float)
    cube = np.zeros((1, 10, 2), dtype=int)
    cube[0, 6, 0] = 2
    cube[0, 5, 1] = 3

    trials, units = cube_to_long(ts, cube)

    assert len(units.id) == 2
    assert len(trials) == 1
    assert len(trials[0].timestamps) == 5
    assert np.allclose(np.array([1, 1, 1, 0, 0]), trials[0].unit_index)
    assert np.allclose(np.array([5, 5, 5, 6, 6]), trials[0].timestamps)
