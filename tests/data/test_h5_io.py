import os
import tempfile

import h5py
import numpy as np
import pytest

from torch_brain.data import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    RegularTimeSeries,
)


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        # clean up the temporary file after the test
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def test_save_to_hdf5(test_filepath):
    a = IrregularTimeSeries(
        timestamps=np.array([0.0, 1.0, 2.0]), x=np.array([1, 2, 3]), domain="auto"
    )

    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    b = Interval(start=np.array([0.0, 1.0, 2.0]), end=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    c = RegularTimeSeries(x=np.random.random((100, 48)), sampling_rate=10)

    with h5py.File(test_filepath, "w") as file:
        c.to_hdf5(file)

    d = Data(
        a_timeseries=a,
        b_intervals=b,
        c_timeseries=c,
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as file:
        d.to_hdf5(file)


def test_load_from_h5(test_filepath):
    # create a file and save it
    a = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )
    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    del a

    # load it again
    with h5py.File(test_filepath, "r") as file:
        a = IrregularTimeSeries.from_hdf5(file)

        np.testing.assert_array_equal(a.timestamps[:], np.array([0, 1, 2]))
        np.testing.assert_array_equal(a.x[:], np.array([1, 2, 3]))

    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    del b

    with h5py.File(test_filepath, "r") as file:
        b = Interval.from_hdf5(file)

        np.testing.assert_array_equal(b.start[:], np.array([0, 1, 2]))
        np.testing.assert_array_equal(b.end[:], np.array([1, 2, 3]))

    a = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )
    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))
    c = RegularTimeSeries(x=np.random.random((100, 48)), sampling_rate=10)
    d = Data(
        a_timeseries=a,
        b_intervals=b,
        c_timeseries=c,
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as file:
        d.to_hdf5(file)

    del d

    with h5py.File(test_filepath, "r") as file:
        d = Data.from_hdf5(file)

        np.testing.assert_array_equal(d.a_timeseries.timestamps[:], np.array([0, 1, 2]))
        np.testing.assert_array_equal(d.a_timeseries.x[:], np.array([1, 2, 3]))
        np.testing.assert_array_equal(d.b_intervals.start[:], np.array([0, 1, 2]))
        np.testing.assert_array_equal(d.b_intervals.end[:], np.array([1, 2, 3]))
        np.testing.assert_array_equal(d.x[:], np.array([0, 1, 2]))
        np.testing.assert_array_equal(d.y[:], np.array([1, 2, 3]))
        np.testing.assert_array_equal(d.z[:], np.array([2, 3, 4]))


def test_load_classmethod(test_filepath):

    a = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )
    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))
    c = RegularTimeSeries(x=np.random.random((100, 48)), sampling_rate=10)
    d = Data(
        a_timeseries=a,
        b_intervals=b,
        c_timeseries=c,
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as file:
        d.to_hdf5(file)

    del d

    d = Data.load(test_filepath)
    np.testing.assert_array_equal(d.a_timeseries.timestamps[:], np.array([0, 1, 2]))
    np.testing.assert_array_equal(d.a_timeseries.x[:], np.array([1, 2, 3]))
    np.testing.assert_array_equal(d.b_intervals.start[:], np.array([0, 1, 2]))
    np.testing.assert_array_equal(d.b_intervals.end[:], np.array([1, 2, 3]))
    np.testing.assert_array_equal(d.x[:], np.array([0, 1, 2]))
    np.testing.assert_array_equal(d.y[:], np.array([1, 2, 3]))
    np.testing.assert_array_equal(d.z[:], np.array([2, 3, 4]))


def test_lazy_load_h5_with_missing_keys(test_filepath):

    a = Interval(start=np.array([0.0, 1.0, 2.0]), end=np.array([1.0, 2.0, 3.0]))

    # save
    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    # load and break
    with h5py.File(test_filepath, "r+") as file:
        del file["start"]

    # load again
    with h5py.File(test_filepath, "r") as file:
        with pytest.raises(KeyError, match="missing keys required"):
            LazyInterval.from_hdf5(file)

    b = RegularTimeSeries(x=np.random.random((100, 48)), sampling_rate=10)

    # save
    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    del b

    # load and break
    with h5py.File(test_filepath, "r+") as file:
        del file["domain"]

    # load again
    with h5py.File(test_filepath, "r") as file:
        with pytest.raises(KeyError, match="missing keys required"):
            LazyRegularTimeSeries.from_hdf5(file)

    c = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )

    # save
    with h5py.File(test_filepath, "w") as file:
        c.to_hdf5(file)

    del c

    # load and break
    with h5py.File(test_filepath, "r+") as file:
        del file["timestamp_indices_1s"]

    # load again
    with h5py.File(test_filepath, "r") as file:
        with pytest.raises(KeyError, match="missing keys required"):
            LazyIrregularTimeSeries.from_hdf5(file)


def test_lazy_load_h5_with_invalid_values(test_filepath):

    a = RegularTimeSeries(x=np.random.random((100, 48)), sampling_rate=10)
    a._sampling_rate = 0  # invalid value

    # save
    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    # load
    with h5py.File(test_filepath, "r") as file:
        with pytest.raises(ValueError, match="strictly positive"):
            LazyRegularTimeSeries.from_hdf5(file)


class TestNestedDataLazyPropagation:
    """Test suite for lazy parameter propagation in nested Data objects."""

    @pytest.fixture
    def nested_data_filepath(self, test_filepath):
        """Create a nested Data structure and save to HDF5."""
        innermost_interval = Interval(
            start=np.array([0.0, 0.5]), end=np.array([0.5, 1.0])
        )
        innermost_data = Data(
            interval=innermost_interval, z=np.array([10, 20]), domain=Interval(0.0, 1.0)
        )

        inner_interval = Interval(start=np.array([0.0, 1.0]), end=np.array([1.0, 2.0]))
        inner_data = Data(
            interval=inner_interval,
            x=np.array([1, 2]),
            nested_level2=innermost_data,
            domain=Interval(0.0, 2.0),
        )

        outer_data = Data(
            nested=inner_data, y=np.array([3, 4]), domain=Interval(0.0, 3.0)
        )

        with h5py.File(test_filepath, "w") as f:
            outer_data.to_hdf5(f)

        return test_filepath

    def test_lazy_true_interval(self, nested_data_filepath):
        """Test that lazy=True loads intervals as LazyInterval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=True)
            assert isinstance(loaded.nested.interval, LazyInterval), (
                "Level 1: interval should be LazyInterval when lazy=True"
            )
            assert isinstance(loaded.nested.nested_level2.interval, LazyInterval), (
                "Level 2: interval should be LazyInterval when lazy=True"
            )

    def test_lazy_true_domain(self, nested_data_filepath):
        """Test that lazy=True loads domains as LazyInterval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=True)
            assert isinstance(loaded.nested.domain, LazyInterval), (
                "Level 1: domain should be LazyInterval when lazy=True"
            )
            assert isinstance(loaded.nested.nested_level2.domain, LazyInterval), (
                "Level 2: domain should be LazyInterval when lazy=True"
            )

    def test_lazy_false_interval(self, nested_data_filepath):
        """Test that lazy=False loads intervals as regular Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=False)
            assert type(loaded.nested.interval) is Interval, (
                f"Level 1: interval should be Interval when lazy=False, got {type(loaded.nested.interval)}"
            )
            assert not isinstance(loaded.nested.interval, LazyInterval), (
                "Level 1: interval should NOT be LazyInterval when lazy=False"
            )
            assert type(loaded.nested.nested_level2.interval) is Interval, (
                f"Level 2: interval should be Interval when lazy=False, got {type(loaded.nested.nested_level2.interval)}"
            )
            assert not isinstance(loaded.nested.nested_level2.interval, LazyInterval), (
                "Level 2: interval should NOT be LazyInterval when lazy=False"
            )

    def test_lazy_false_domain(self, nested_data_filepath):
        """Test that lazy=False loads domains as regular Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=False)
            assert type(loaded.nested.domain) is Interval, (
                f"Level 1: domain should be Interval when lazy=False, got {type(loaded.nested.domain)}"
            )
            assert not isinstance(loaded.nested.domain, LazyInterval), (
                "Level 1: domain should NOT be LazyInterval when lazy=False"
            )
            assert type(loaded.nested.nested_level2.domain) is Interval, (
                f"Level 2: domain should be Interval when lazy=False, got {type(loaded.nested.nested_level2.domain)}"
            )
            assert not isinstance(loaded.nested.nested_level2.domain, LazyInterval), (
                "Level 2: domain should NOT be LazyInterval when lazy=False"
            )

    def test_materialize_converts_lazy(self, nested_data_filepath):
        """Test that materialize() converts LazyInterval to Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=True)
            loaded.materialize()
            assert type(loaded.nested.interval) is Interval, (
                "Level 1: interval should be Interval after materialize()"
            )
            assert not isinstance(loaded.nested.interval, LazyInterval), (
                "Level 1: interval should NOT be LazyInterval after materialize()"
            )
            assert type(loaded.nested.domain) is Interval, (
                "Level 1: domain should be Interval after materialize()"
            )
            assert not isinstance(loaded.nested.domain, LazyInterval), (
                "Level 1: domain should NOT be LazyInterval after materialize()"
            )


@pytest.mark.parametrize(
    "loader",
    [
        ArrayDict,
        LazyArrayDict,
        Interval,
        LazyInterval,
        IrregularTimeSeries,
        LazyIrregularTimeSeries,
        RegularTimeSeries,
        LazyRegularTimeSeries,
    ],
)
def test_from_hdf5_wrong_object_type(test_filepath, loader):
    # a group saved as one class cannot be loaded as another
    other = (
        Interval(start=np.array([0.0]), end=np.array([1.0]))
        if not issubclass(loader, Interval)
        else ArrayDict(x=np.array([1, 2, 3]))
    )
    with h5py.File(test_filepath, "w") as file:
        other.to_hdf5(file.create_group("group"))

    with h5py.File(test_filepath, "r") as file:
        with pytest.raises(ValueError, match="expected"):
            loader.from_hdf5(file["group"])


@pytest.mark.parametrize("lazy", [True, False])
def test_data_from_hdf5_reads_group_attrs_once(test_filepath, lazy, monkeypatch):
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.0, 1.0, 2.0]),
            unit_index=np.array([0, 1, 0]),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((10, 2)), sampling_rate=10.0, domain_start=0.0
        ),
        trials=Interval(start=np.array([0.0]), end=np.array([1.0])),
        units=ArrayDict(id=np.array(["a", "b"])),
        session=Data(label=np.array([1, 2]), domain="auto"),
        domain=Interval(start=np.array([0.0]), end=np.array([2.0])),
    )
    with h5py.File(test_filepath, "w") as file:
        data.to_hdf5(file)

    # count attribute reads per object (group or file)
    reads = {}
    original_getitem = h5py.AttributeManager.__getitem__

    def counting_getitem(self, name):
        key = (self._id.id, name)
        reads[key] = reads.get(key, 0) + 1
        return original_getitem(self, name)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", counting_getitem)

    with h5py.File(test_filepath, "r") as file:
        loaded = Data.from_hdf5(file, lazy=lazy)
        assert set(loaded.keys()) >= {"spikes", "lfp", "trials", "units", "session"}

    duplicates = {key: n for key, n in reads.items() if n > 1}
    assert not duplicates, f"attributes read more than once: {duplicates}"
