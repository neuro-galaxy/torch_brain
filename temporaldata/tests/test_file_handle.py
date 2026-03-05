import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import (
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    Data,
)


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


@pytest.fixture
def saved_data(test_filepath):
    """Create and save a Data object, return the filepath."""
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.0, 1.0, 2.0]),
            x=np.array([1.0, 2.0, 3.0]),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.random.random((100, 4)),
            sampling_rate=10,
            domain=Interval(0.0, 10.0),
        ),
        trials=Interval(
            start=np.array([0.0, 1.0, 2.0]),
            end=np.array([1.0, 2.0, 3.0]),
        ),
        session_id="test_session",
        domain=Interval(0.0, 3.0),
    )
    data.save(test_filepath)
    return test_filepath


@pytest.fixture
def nested_saved_data(test_filepath):
    """Create and save a nested Data object, return the filepath."""
    inner = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.0, 0.5]),
            x=np.array([1.0, 2.0]),
            domain="auto",
        ),
        domain=Interval(0.0, 1.0),
    )
    outer = Data(
        inner=inner,
        y=np.array([3, 4]),
        domain=Interval(0.0, 2.0),
    )
    outer.save(test_filepath)
    return test_filepath


class TestFileProperty:
    def test_load_lazy_sets_file(self, saved_data):
        data = Data.load(saved_data)
        assert data.file is not None
        assert isinstance(data.file, h5py.File)
        assert data.file.id.valid
        data.close()

    def test_load_non_lazy_file_is_none(self, saved_data):
        data = Data.load(saved_data, lazy=False)
        assert data.file is None
        assert np.all(data.spikes.x == np.array([1.0, 2.0, 3.0]))

    def test_file_none_by_default(self):
        data = Data(domain=Interval(0.0, 1.0))
        assert data.file is None

    def test_from_hdf5_lazy_sets_file(self, saved_data):
        f = h5py.File(saved_data, "r")
        data = Data.from_hdf5(f, lazy=True)
        assert data.file is f
        f.close()

    def test_from_hdf5_non_lazy_file_is_none(self, saved_data):
        with h5py.File(saved_data, "r") as f:
            data = Data.from_hdf5(f, lazy=False)
            assert data.file is None

    def test_file_not_in_keys(self, saved_data):
        data = Data.load(saved_data)
        assert "_file" not in data.keys()
        assert "file" not in data.keys()
        data.close()

    def test_nested_data_file_is_none(self, nested_saved_data):
        data = Data.load(nested_saved_data, lazy=True)
        assert data.file is not None
        assert data.inner.file is None
        data.close()


class TestClose:
    def test_close_closes_file(self, saved_data):
        data = Data.load(saved_data)
        assert data.file is not None
        data.close()
        assert data.file is None

    def test_close_without_file_is_silent(self):
        data = Data(domain=Interval(0.0, 1.0))
        data.close()  # should not raise

    def test_close_strict_raises_without_file(self):
        data = Data(domain=Interval(0.0, 1.0))
        with pytest.raises(RuntimeError, match="No file handle is open"):
            data.close(strict=True)

    def test_close_strict_does_not_raise_with_file(self, saved_data):
        data = Data.load(saved_data)
        data.close(strict=True)  # should not raise
        assert data.file is None

    def test_double_close_is_silent(self, saved_data):
        data = Data.load(saved_data)
        data.close()
        data.close()  # should not raise

    def test_data_accessible_after_materialize_and_close(self, saved_data):
        data = Data.load(saved_data)
        data.materialize()
        data.close()
        # data should still be accessible after materializing and closing
        assert np.all(data.spikes.timestamps == np.array([0.0, 1.0, 2.0]))
        assert np.all(data.spikes.x == np.array([1.0, 2.0, 3.0]))


class TestContextManager:
    def test_context_manager_closes_file(self, saved_data):
        with Data.load(saved_data) as data:
            assert data.file is not None
            assert np.all(data.spikes.timestamps == np.array([0.0, 1.0, 2.0]))
        assert data.file is None

    def test_context_manager_returns_data(self, saved_data):
        with Data.load(saved_data) as data:
            assert isinstance(data, Data)
            assert "spikes" in data.keys()

    def test_context_manager_closes_on_exception(self, saved_data):
        with pytest.raises(ValueError):
            with Data.load(saved_data) as data:
                file_ref = data.file
                raise ValueError("test error")
        assert not file_ref.id.valid

    def test_context_manager_non_lazy(self, saved_data):
        with Data.load(saved_data, lazy=False) as data:
            assert data.file is None
            assert np.all(data.spikes.timestamps == np.array([0.0, 1.0, 2.0]))

    def test_context_manager_without_file(self):
        data = Data(domain=Interval(0.0, 1.0))
        with data as d:
            assert d is data
        # should not raise


class TestLoadFileLifecycle:
    def test_lazy_access_works_before_close(self, saved_data):
        data = Data.load(saved_data)
        # access lazy attributes
        assert np.all(data.spikes.timestamps == np.array([0.0, 1.0, 2.0]))
        assert np.all(data.trials.start == np.array([0.0, 1.0, 2.0]))
        data.close()

    def test_load_non_lazy_file(self, saved_data):
        data = Data.load(saved_data, lazy=False)
        assert np.all(data.spikes.x == np.array([1.0, 2.0, 3.0]))

    def test_load_closes_file_on_error(self, test_filepath):
        from unittest.mock import patch

        # write a corrupt file (missing required 'object' attr on a group)
        with h5py.File(test_filepath, "w") as f:
            f.attrs["object"] = "Data"
            f.attrs["absolute_start"] = 0.0
            f.create_group("bad_child")
            # no 'object' attr on the group => from_hdf5 will raise

        # capture the file handle created inside load()
        opened_file = None
        original_init = h5py.File.__init__

        def tracking_init(self, *args, **kwargs):
            nonlocal opened_file
            original_init(self, *args, **kwargs)
            opened_file = self

        with patch.object(h5py.File, "__init__", tracking_init):
            with pytest.raises(KeyError):
                Data.load(test_filepath)

        assert opened_file is not None
        assert not opened_file.id.valid

    def test_slice_on_lazy_loaded_data(self, saved_data):
        with Data.load(saved_data) as data:
            sliced = data.slice(0.5, 2.0)
            # slice [0.5, 2.0): timestamps 1.0 is included, reset to 0.5
            assert np.all(sliced.spikes.timestamps == np.array([0.5]))
            assert np.all(sliced.spikes.x == np.array([2.0]))
            # trials [0,1) and [1,2) overlap with [0.5, 2.0), reset by -0.5
            assert np.allclose(sliced.trials.start, np.array([-0.5, 0.5]))
            assert np.allclose(sliced.trials.end, np.array([0.5, 1.5]))
            # sliced object should not hold a file handle
            assert sliced.file is None

    def test_select_by_interval_on_lazy_loaded_data(self, saved_data):
        with Data.load(saved_data) as data:
            interval = Interval(
                start=np.array([0.0, 1.5]),
                end=np.array([0.5, 2.5]),
            )
            selected = data.select_by_interval(interval)
            # timestamps 0.0 is in [0.0, 0.5), 2.0 is in [1.5, 2.5)
            assert np.all(selected.spikes.timestamps == np.array([0.0, 2.0]))
            assert np.all(selected.spikes.x == np.array([1.0, 3.0]))
            assert selected.file is None

    def test_to_dict_excludes_file(self, saved_data):
        with Data.load(saved_data) as data:
            d = data.to_dict()
            assert "_file" not in d
            assert "file" not in d
            assert "spikes" in d
            assert "session_id" in d


class TestDeepCopy:
    def test_deepcopy_preserves_file_ref(self, saved_data):
        import copy

        data = Data.load(saved_data)
        data_copy = copy.deepcopy(data)
        # both should share the same h5py.File (not deepcopied)
        assert data_copy.file is data.file
        data.close()
