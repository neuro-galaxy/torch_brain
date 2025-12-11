import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import RegularTimeSeries, LazyRegularTimeSeries


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


def test_regulartimeseries(test_filepath):
    def _test_regulartimeseries(data):
        assert len(data) == 100

        assert data.domain.start[0] == 0.0
        assert data.domain.end[0] == 9.9

        data_slice = data.slice(2.0, 8.0, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])
        assert data_slice.domain.start[0] == 2.0
        assert data_slice.domain.end[0] == 7.9
        assert np.allclose(data_slice.timestamps, np.arange(2.0, 8.0, 0.1))

        data_slice = data.slice(2.0, 8.0, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])
        assert data_slice.domain.start[0] == 0.0
        assert data_slice.domain.end[0] == 5.9
        assert np.allclose(data_slice.timestamps, np.arange(0.0, 6.0, 0.1))

        # try slicing with skewed start and end
        # the sampling frequency is 10
        data_slice = data.slice(2.03, 8.09, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[21:80])
        assert np.allclose(data_slice.domain.start, np.array([0.07]))
        assert np.allclose(data_slice.domain.end, np.array([5.87]))
        assert np.allclose(data_slice.timestamps, np.arange(0.07, 5.88, 0.1))

        data_slice = data.slice(4.051, 12.0, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[41:])
        assert np.allclose(data_slice.domain.start, np.array([0.049]))
        assert np.allclose(data_slice.domain.end, np.array([5.849]))
        assert np.allclose(data_slice.timestamps, np.arange(0.049, 5.88, 0.1))

        data_slice = data.slice(4.051, 12.0, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp[41:])
        assert np.allclose(data_slice.domain.start, np.array([4.1]))
        assert np.allclose(data_slice.domain.end, np.array([9.9]))
        assert np.allclose(data_slice.timestamps, np.arange(4.1, 10.0, 0.1))

        data_slice = data.slice(-10, 20, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp)
        assert np.allclose(data_slice.domain.start, data.domain.start)
        assert np.allclose(data_slice.domain.end, data.domain.end)
        assert np.allclose(data_slice.timestamps, data.timestamps)

    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )

    _test_regulartimeseries(data)

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)

        _test_regulartimeseries(data)

    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)),
        sampling_rate=10,
        domain="auto",
        domain_start=1.0,
    )

    def _test_regulartimeseries_with_domain_start(data):
        assert len(data) == 100

        assert data.domain.start[0] == 1.0
        assert data.domain.end[0] == 10.9

        data_slice = data.slice(3.0, 9.0)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])

        # try slicing with skewed start and end
        # the sampling frequency is 10
        data_slice = data.slice(3.03, 9.09)
        assert np.allclose(data_slice.lfp, data.lfp[21:80])

        data_slice = data.slice(5.051, 13.0)
        assert np.allclose(data_slice.lfp, data.lfp[41:])

    _test_regulartimeseries_with_domain_start(data)

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)
        _test_regulartimeseries_with_domain_start(data)


def test_lazy_regular_timeseries(test_filepath):
    raw = np.random.random((1000, 128))
    gamma = np.random.random((1000, 128))

    data = RegularTimeSeries(
        raw=raw.copy(),
        gamma=gamma.copy(),
        sampling_rate=250.0,
        domain="auto",
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)

        assert len(data) == 1000
        assert data.sampling_rate == 250.0

        # make sure that nothing is loaded yet
        assert all(isinstance(data.__dict__[key], h5py.Dataset) for key in data.keys())

        # make sure that the attribute is loaded
        assert isinstance(data.gamma, np.ndarray)
        # make sure that the attribute is loaded correctly
        assert np.allclose(data.gamma, gamma)
        # make sure that the loaded attribute replaced the h5py.Dataset reference
        assert isinstance(data.__dict__["gamma"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "gamma"
        )

        assert data.__class__ == LazyRegularTimeSeries

        data.raw
        # final attribute was accessed, the object should automatically convert to ArrayDict
        assert data.__class__ == RegularTimeSeries

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)

        data = data.slice(1.0, 3.0)
        assert len(data.gamma) == 500
        assert len(data.timestamps) == 500
        assert np.allclose(data.gamma, gamma[250:750])

        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "gamma"
        )

        data = data.slice(0.5, 1.5)

        assert np.allclose(data.gamma, gamma[375:625])
        assert np.allclose(data.raw, raw[375:625])

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)
        data = data.slice(1.0, 3.0)

        # timestamps is a property not an attribute, make sure it's defined properly
        # even if no other attribute is loaded
        assert len(data.timestamps) == 500

        assert np.allclose(data.timestamps, np.arange(0.0, 2.0, 1 / 250.0))

    with h5py.File(test_filepath, "r") as f:
        data = LazyRegularTimeSeries.from_hdf5(f)
        data = data.slice(1.0, 3.0, reset_origin=False)

        # timestamps is a property not an attribute, make sure it's defined properly
        # even if no other attribute is loaded
        assert len(data.timestamps) == 500
        assert data.domain.start[0] == 1.0
        assert data.domain.end[0] == 2.996
        assert np.allclose(data.timestamps, np.arange(1.0, 3.0, 1 / 250.0))

        data = data.slice(1.0, 2.0, reset_origin=True)
        assert data.domain.start[0] == 0.0
        assert data.domain.end[0] == 0.996
        assert len(data.timestamps) == 250
        assert np.allclose(data.timestamps, np.arange(0.0, 1.0, 1 / 250.0))

        assert np.allclose(data.gamma, gamma[250:500])

        data = LazyRegularTimeSeries.from_hdf5(f)
        data = data.slice(1.0, 3.0, reset_origin=False)
        data = data.slice(1.0, 2.0, reset_origin=True)

        assert len(data.timestamps) == 250
        assert np.allclose(data.timestamps, np.arange(0.0, 1.0, 1 / 250.0))
        assert np.allclose(data.gamma, gamma[250:500])

        data = LazyRegularTimeSeries.from_hdf5(f)
        timestamps = data.timestamps
        assert isinstance(timestamps, np.ndarray)
        data = data.slice(1.0, 3.0, reset_origin=False)

        assert len(data.timestamps) == 500
        assert np.allclose(data.timestamps, np.arange(1.0, 3.0, 1 / 250.0))


def test_regular_to_irregular_timeseries():
    a = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )
    b = a.to_irregular()
    assert np.allclose(b.timestamps, np.arange(0, 10, 0.1))
    assert np.allclose(b.lfp, a.lfp)
