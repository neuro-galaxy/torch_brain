import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import (
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    Data,
    LazyInterval,
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

    c = RegularTimeSeries(
        x=np.random.random((100, 48)), sampling_rate=10, domain=Interval(0.0, 10.0)
    )

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

        assert np.all(a.timestamps[:] == np.array([0, 1, 2]))
        assert np.all(a.x[:] == np.array([1, 2, 3]))

    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    del b

    with h5py.File(test_filepath, "r") as file:
        b = Interval.from_hdf5(file)

        assert np.all(b.start[:] == np.array([0, 1, 2]))
        assert np.all(b.end[:] == np.array([1, 2, 3]))

    a = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )
    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))
    c = RegularTimeSeries(
        x=np.random.random((100, 48)), sampling_rate=10, domain=Interval(0.0, 10.0)
    )
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

        assert np.all(d.a_timeseries.timestamps[:] == np.array([0, 1, 2]))
        assert np.all(d.a_timeseries.x[:] == np.array([1, 2, 3]))
        assert np.all(d.b_intervals.start[:] == np.array([0, 1, 2]))
        assert np.all(d.b_intervals.end[:] == np.array([1, 2, 3]))
        assert np.all(d.x[:] == np.array([0, 1, 2]))
        assert np.all(d.y[:] == np.array([1, 2, 3]))
        assert np.all(d.z[:] == np.array([2, 3, 4]))


def test_load_classmethod(test_filepath):

    a = IrregularTimeSeries(
        np.array([0.0, 1.0, 2.0]), x=np.array([1.0, 2.0, 3.0]), domain="auto"
    )
    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))
    c = RegularTimeSeries(
        x=np.random.random((100, 48)), sampling_rate=10, domain=Interval(0.0, 10.0)
    )
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
    assert np.all(d.a_timeseries.timestamps[:] == np.array([0, 1, 2]))
    assert np.all(d.a_timeseries.x[:] == np.array([1, 2, 3]))
    assert np.all(d.b_intervals.start[:] == np.array([0, 1, 2]))
    assert np.all(d.b_intervals.end[:] == np.array([1, 2, 3]))
    assert np.all(d.x[:] == np.array([0, 1, 2]))
    assert np.all(d.y[:] == np.array([1, 2, 3]))
    assert np.all(d.z[:] == np.array([2, 3, 4]))


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
            assert isinstance(
                loaded.nested.interval, LazyInterval
            ), "Level 1: interval should be LazyInterval when lazy=True"
            assert isinstance(
                loaded.nested.nested_level2.interval, LazyInterval
            ), "Level 2: interval should be LazyInterval when lazy=True"

    def test_lazy_true_domain(self, nested_data_filepath):
        """Test that lazy=True loads domains as LazyInterval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=True)
            assert isinstance(
                loaded.nested.domain, LazyInterval
            ), "Level 1: domain should be LazyInterval when lazy=True"
            assert isinstance(
                loaded.nested.nested_level2.domain, LazyInterval
            ), "Level 2: domain should be LazyInterval when lazy=True"

    def test_lazy_false_interval(self, nested_data_filepath):
        """Test that lazy=False loads intervals as regular Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=False)
            assert (
                type(loaded.nested.interval) == Interval
            ), f"Level 1: interval should be Interval when lazy=False, got {type(loaded.nested.interval)}"
            assert not isinstance(
                loaded.nested.interval, LazyInterval
            ), "Level 1: interval should NOT be LazyInterval when lazy=False"
            assert (
                type(loaded.nested.nested_level2.interval) == Interval
            ), f"Level 2: interval should be Interval when lazy=False, got {type(loaded.nested.nested_level2.interval)}"
            assert not isinstance(
                loaded.nested.nested_level2.interval, LazyInterval
            ), "Level 2: interval should NOT be LazyInterval when lazy=False"

    def test_lazy_false_domain(self, nested_data_filepath):
        """Test that lazy=False loads domains as regular Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=False)
            assert (
                type(loaded.nested.domain) == Interval
            ), f"Level 1: domain should be Interval when lazy=False, got {type(loaded.nested.domain)}"
            assert not isinstance(
                loaded.nested.domain, LazyInterval
            ), "Level 1: domain should NOT be LazyInterval when lazy=False"
            assert (
                type(loaded.nested.nested_level2.domain) == Interval
            ), f"Level 2: domain should be Interval when lazy=False, got {type(loaded.nested.nested_level2.domain)}"
            assert not isinstance(
                loaded.nested.nested_level2.domain, LazyInterval
            ), "Level 2: domain should NOT be LazyInterval when lazy=False"

    def test_materialize_converts_lazy(self, nested_data_filepath):
        """Test that materialize() converts LazyInterval to Interval in nested Data."""
        with h5py.File(nested_data_filepath, "r") as f:
            loaded = Data.from_hdf5(f, lazy=True)
            loaded.materialize()
            assert (
                type(loaded.nested.interval) == Interval
            ), "Level 1: interval should be Interval after materialize()"
            assert not isinstance(
                loaded.nested.interval, LazyInterval
            ), "Level 1: interval should NOT be LazyInterval after materialize()"
            assert (
                type(loaded.nested.domain) == Interval
            ), "Level 1: domain should be Interval after materialize()"
            assert not isinstance(
                loaded.nested.domain, LazyInterval
            ), "Level 1: domain should NOT be LazyInterval after materialize()"
