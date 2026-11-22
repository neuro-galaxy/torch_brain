import copy

import h5py
import numpy as np
import pytest

from torch_brain.data.utils import DeferredH5Dataset, _size_repr


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "deferred.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(12, dtype=np.float32).reshape(4, 3))
    return path


def test_dataset_is_opened_on_first_use(h5_path):
    with h5py.File(h5_path, "r") as f:
        d = DeferredH5Dataset(f, "x")
        assert d._dataset is None

        assert d.shape == (4, 3)
        assert isinstance(d._dataset, h5py.Dataset)
        # the opened dataset is cached
        assert d.dataset is d.dataset


def test_mirrors_h5py_dataset_api(h5_path):
    with h5py.File(h5_path, "r") as f:
        d = DeferredH5Dataset(f, "x")
        expected = f["x"][:]

        assert len(d) == 4
        assert d.shape == (4, 3)
        assert d.dtype == np.float32
        np.testing.assert_array_equal(d[:], expected)
        np.testing.assert_array_equal(d[1:3], expected[1:3])
        mask = np.array([True, False, True, False])
        np.testing.assert_array_equal(d[mask], expected[mask])


def test_copies_share_the_reference(h5_path):
    with h5py.File(h5_path, "r") as f:
        d = DeferredH5Dataset(f, "x")
        assert copy.copy(d) is d
        assert copy.deepcopy(d) is d


def test_valid_and_repr_track_file_state(h5_path):
    f = h5py.File(h5_path, "r")
    d = DeferredH5Dataset(f, "x")
    assert d.valid
    assert repr(d) == '<DeferredH5Dataset "x" (not opened)>'
    # repr must not open the dataset
    assert d._dataset is None

    _ = d.shape  # opens the dataset
    assert repr(d).startswith("<DeferredH5Dataset of <HDF5 dataset")

    f.close()
    assert not d.valid
    assert repr(d) == '<DeferredH5Dataset "x" (file closed)>'


def test_size_repr_shows_lazy_array(h5_path):
    with h5py.File(h5_path, "r") as f:
        d = DeferredH5Dataset(f, "x")
        assert "lazy array, shape=(4, 3), dtype=float32" in _size_repr("x", d)
        assert "shape=(2, 3)" in _size_repr("x", d, first_dim=2)
