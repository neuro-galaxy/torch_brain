import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import RegularTimeSeries, IrregularTimeSeries, Interval, Data


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
