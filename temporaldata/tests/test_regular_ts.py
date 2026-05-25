import os
import tempfile
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd
import pytest

from temporaldata import Interval, LazyRegularTimeSeries, RegularTimeSeries


@contextmanager
def _make_lazy(non_lazy, lazy_cls, test_filepath):
    with h5py.File(test_filepath, "w") as f:
        non_lazy.to_hdf5(f)
    f = h5py.File(test_filepath, "r")
    yield lazy_cls.from_hdf5(f)
    f.close()


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
        assert data.domain.end[0] == 10.0

        data_slice = data.slice(2.0, 8.0, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])
        assert data_slice.domain.start[0] == 2.0
        assert data_slice.domain.end[0] == 8.0
        assert np.allclose(data_slice.timestamps, np.arange(2.0, 8.0, 0.1))

        data_slice = data.slice(2.0, 8.0, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])
        assert data_slice.domain.start[0] == 0.0
        assert data_slice.domain.end[0] == 6.0
        assert np.allclose(data_slice.timestamps, np.arange(0.0, 6.0, 0.1))

        # try slicing with skewed start and end
        # the sampling frequency is 10
        data_slice = data.slice(2.03, 8.09, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[21:81])
        assert np.allclose(data_slice.domain.start, np.array([0.07]))
        assert np.allclose(data_slice.domain.end, np.array([6.07]))
        assert np.allclose(data_slice.timestamps, np.arange(0.07, 5.98, 0.1))

        data_slice = data.slice(4.051, 12.0, reset_origin=True)
        assert np.allclose(data_slice.lfp, data.lfp[41:])
        assert np.allclose(data_slice.domain.start, np.array([0.049]))
        assert np.allclose(data_slice.domain.end, np.array([5.949]))
        assert np.allclose(data_slice.timestamps, np.arange(0.049, 5.88, 0.1))

        data_slice = data.slice(4.051, 12.0, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp[41:])
        assert np.allclose(data_slice.domain.start, np.array([4.1]))
        assert np.allclose(data_slice.domain.end, np.array([10.0]))
        assert np.allclose(data_slice.timestamps, np.arange(4.1, 10.0, 0.1))

        data_slice = data.slice(-10, 20, reset_origin=False)
        assert np.allclose(data_slice.lfp, data.lfp)
        assert np.allclose(data_slice.domain.start, data.domain.start)
        assert np.allclose(data_slice.domain.end, data.domain.end)
        assert np.allclose(data_slice.timestamps, data.timestamps)

        domain_start, domain_end = data.domain.start[0], data.domain.end[-1]
        data_slice = data.slice(domain_start, domain_end, reset_origin=False)
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
        assert data.domain.end[0] == 11.0

        data_slice = data.slice(3.0, 9.0)
        assert np.allclose(data_slice.lfp, data.lfp[20:80])

        # try slicing with skewed start and end
        # the sampling frequency is 10
        data_slice = data.slice(3.03, 9.09)
        assert np.allclose(data_slice.lfp, data.lfp[21:81])

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
        assert data.domain.end[0] == 3.0
        assert np.allclose(data.timestamps, np.arange(1.0, 3.0, 1 / 250.0))

        data = data.slice(1.0, 2.0, reset_origin=True)
        assert data.domain.start[0] == 0.0
        assert data.domain.end[0] == 1.0
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


def test_slice_numerical_instability():
    ts = RegularTimeSeries(value=np.zeros((40)), sampling_rate=4, domain="auto")
    # Expected timestamps: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, ...]

    eps = 1e-14

    # `end` is infinitesimally smaller than the next timestamp (0.9999999 scenario).
    # As the interval is [start, end), it should still safely include up to 0.75.
    start = 0.0
    end = 1.0 - eps
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.0, 0.25, 0.5, 0.75]))
    assert sliced_ts.domain.start[0] == 0.0
    assert sliced_ts.domain.end[-1] == 1.0

    # `end` is infinitesimally larger than an exact timestamp (1.0000001 scenario).
    # As the end is larger due to numerical instability even if the interval is [start, end), 1.0 should still be EXCLUDED
    start = 0.25
    end = 1.0 + eps
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.25, 0.5, 0.75]))
    assert sliced_ts.domain.start[0] == 0.25
    assert sliced_ts.domain.end[-1] == 1.0

    # `start` is computed slightly larger than an exact timestamp.
    # As the start is larger due to numerical instability 0.25 should be INCLUDED.
    start = 0.25 + eps
    end = 1.0
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.25, 0.5, 0.75]))
    assert sliced_ts.domain.start[0] == 0.25
    assert sliced_ts.domain.end[-1] == 1.0

    # Maximum Precision Limits via np.nextafter
    # np.nextafter gives the very next representable float in memory.
    start = 0.5
    end = np.nextafter(1.0, 0.0)  # The largest possible float strictly less than 1.0
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.5, 0.75]))
    assert sliced_ts.domain.start[0] == 0.5
    assert sliced_ts.domain.end[-1] == 1.0

    # Should still treat `end` as 1.0 and EXCLUDED it
    start = 0.5
    end = np.nextafter(1.0, 2.0)  # The largest possible float strictly more than 1.0
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.5, 0.75]))

    # `start` is the smallest possible float strictly greater than 0.25
    # Should still treat `start` as 0.25 and INCLUDED it
    start = np.nextafter(
        0.25, 1.0
    )  # The largest possible float strictly more than 0.25
    end = 1.0
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.25, 0.5, 0.75]))
    assert sliced_ts.domain.start[0] == 0.25
    assert sliced_ts.domain.end[-1] == 1.0

    ts = RegularTimeSeries(value=np.zeros((40)), sampling_rate=10, domain="auto")
    # Expected timestamps: [0.0, 0.1, 0.2, ...]

    # Using math that natively generates known float anomalies.
    # Should safely INCLUDE 0.3 and EXCLUDE 0.9 as the end > 0.9 is due to numerical instability
    start = 0.1 + 0.2  # 0.30000000000000004
    end = start * 3  # 0.9000000000000001
    sliced_ts = ts.slice(start, end, reset_origin=False)
    assert np.allclose(sliced_ts.timestamps, np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    assert sliced_ts.domain.start[0] == 0.3
    assert sliced_ts.domain.end[-1] == 0.9


def test_slice_outside_domain(test_filepath):
    ts = RegularTimeSeries(
        value=np.zeros((100)), sampling_rate=10, domain="auto", domain_start=10.0
    )

    def _assert_slice_outside_domain(ts):
        sliced_ts = ts.slice(0, 5, reset_origin=False)
        assert len(sliced_ts) == 0
        assert (
            sliced_ts.domain.start[0] == sliced_ts.domain.end[-1] == ts.domain.start[0]
        )

        sliced_ts = ts.slice(0, 5, reset_origin=True)
        assert len(sliced_ts) == 0
        assert sliced_ts.domain.start[0] == sliced_ts.domain.end[-1] == 0.0

        sliced_ts = ts.slice(30, 45, reset_origin=False)
        assert len(sliced_ts) == 0
        assert (
            sliced_ts.domain.start[0] == sliced_ts.domain.end[-1] == ts.domain.end[-1]
        )

        sliced_ts = ts.slice(30, 45, reset_origin=True)
        assert len(sliced_ts) == 0
        assert sliced_ts.domain.start[0] == sliced_ts.domain.end[-1] == 0.0

    _assert_slice_outside_domain(ts)

    with h5py.File(test_filepath, "w") as f:
        ts.to_hdf5(f)

    with h5py.File(test_filepath, "r") as f:
        lazy_ts = LazyRegularTimeSeries.from_hdf5(f)
        _assert_slice_outside_domain(lazy_ts)


class TestFromGappyTimeseries:
    def test_basic(self):
        # 5 grid points at 100Hz, the 0.02s sample is missing.
        ts = np.array([0.0, 0.01, 0.03, 0.04])
        raw = np.array([1.0, 2.0, 3.0, 4.0])

        rts = RegularTimeSeries.from_gappy_timeseries(ts, sampling_rate=100.0, raw=raw)

        assert isinstance(rts, RegularTimeSeries)
        assert rts.sampling_rate == 100.0
        assert len(rts) == 5
        np.testing.assert_array_equal(
            np.isnan(rts.raw), [False, False, True, False, False]
        )
        np.testing.assert_array_equal(rts.raw[~np.isnan(rts.raw)], raw)
        np.testing.assert_allclose(rts.domain.start, [0.0, 0.03])
        np.testing.assert_allclose(rts.domain.end, [0.02, 0.05])

    def test_multiple_arrays_and_multidim(self):
        ts = np.array([10.0, 10.5, 11.5])  # missing 11.0 at sr=2Hz
        a = np.array([1.0, 2.0, 3.0])
        b = np.arange(12).reshape(3, 4).astype(float)

        rts = RegularTimeSeries.from_gappy_timeseries(ts, sampling_rate=2.0, a=a, b=b)

        assert len(rts) == 4
        np.testing.assert_array_equal(np.isnan(rts.a), [False, False, True, False])
        assert rts.b.shape == (4, 4)
        assert np.isnan(rts.b[2]).all()
        np.testing.assert_array_equal(rts.b[[0, 1, 3]], b)
        # Domain excludes the gap at 11.0–11.5.
        np.testing.assert_allclose(rts.domain.start, [10.0, 11.5])
        np.testing.assert_allclose(rts.domain.end, [11.0, 12.0])

    def test_integer_gap_preserves_dtype(self):
        ts = np.array([0.0, 0.1, 0.3])  # missing 0.2 at sr=10Hz
        vals = np.array([7, 8, 9], dtype=np.int32)

        rts = RegularTimeSeries.from_gappy_timeseries(
            ts, sampling_rate=10.0, gap_value=-1, raw=vals
        )

        assert rts.raw.dtype == np.int32
        np.testing.assert_array_equal(rts.raw, [7, 8, -1, 9])

    def test_validation(self):
        ts = np.array([0.0, 0.1, 0.2])
        raw = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="1-D"):
            RegularTimeSeries.from_gappy_timeseries(
                ts.reshape(-1, 1), sampling_rate=10.0, raw=raw
            )

        with pytest.raises(ValueError, match="at least 2"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0]), sampling_rate=10.0, raw=np.array([1.0])
            )

        with pytest.raises(ValueError, match="strictly increasing"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.0, 0.1]),
                sampling_rate=10.0,
                raw=np.array([1.0, 2.0, 3.0]),
            )

        # timestamps off the grid beyond rtol.
        with pytest.raises(ValueError, match="deviate from a regular grid"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.1, 0.205]),
                sampling_rate=10.0,
                raw=raw,
            )

        # sub-sample-spaced (two timestamps round to the same grid index).
        # rtol relaxed so the off-grid check doesn't fire first.
        with pytest.raises(ValueError, match="duplicate or sub-sample-spaced"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.04, 0.1]),
                sampling_rate=10.0,
                rtol=0.5,
                raw=raw,
            )

        # mismatched length.
        with pytest.raises(ValueError, match="length"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, raw=np.array([1.0, 2.0])
            )

        # sampling_rate too high: every gap is multiple grid steps wide.
        # Data is truly at 10 Hz but caller passes 20 Hz.
        with pytest.raises(ValueError, match="appears too high"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.1, 0.2, 0.3]),
                sampling_rate=20.0,
                raw=np.array([1.0, 2.0, 3.0, 4.0]),
            )

    def test_default_gap_value_per_kind(self):
        # ts has one gap (at the 0.2s grid point).
        ts = np.array([0.0, 0.1, 0.3])

        rts = RegularTimeSeries.from_gappy_timeseries(
            ts,
            sampling_rate=10.0,
            f=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            i=np.array([1, 2, 3], dtype=np.int32),
            u=np.array([1, 2, 3], dtype=np.uint8),
            b=np.array([True, False, True], dtype=np.bool_),
        )

        # float default: nan
        assert rts.f.dtype == np.float64
        np.testing.assert_array_equal(np.isnan(rts.f), [False, False, True, False])
        np.testing.assert_array_equal(rts.f[[0, 1, 3]], [1.0, 2.0, 3.0])

        # signed int default: -1
        assert rts.i.dtype == np.int32
        np.testing.assert_array_equal(rts.i, [1, 2, -1, 3])

        # unsigned int default: 0
        assert rts.u.dtype == np.uint8
        np.testing.assert_array_equal(rts.u, [1, 2, 0, 3])

        # bool default: False
        assert rts.b.dtype == np.bool_
        np.testing.assert_array_equal(rts.b, [True, False, False, True])

    def test_gap_value_dict_by_kind(self):
        ts = np.array([0.0, 0.1, 0.3])

        # Per-kind sentinels: signed -> -99, unsigned -> 255, float -> -1.0.
        rts = RegularTimeSeries.from_gappy_timeseries(
            ts,
            sampling_rate=10.0,
            gap_value={"i": -99, "u": 255, "f": -1.0},
            i=np.array([1, 2, 3], dtype=np.int32),
            u=np.array([1, 2, 3], dtype=np.uint8),
            f=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

        np.testing.assert_array_equal(rts.i, [1, 2, -99, 3])
        np.testing.assert_array_equal(rts.u, [1, 2, 255, 3])
        np.testing.assert_array_equal(rts.f, [1.0, 2.0, -1.0, 3.0])

    def test_gap_value_dict_missing_kind_raises(self):
        # Float array given, but dict only has signed/unsigned kinds.
        with pytest.raises(KeyError, match="kind 'f'"):
            RegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.1, 0.3]),
                sampling_rate=10.0,
                gap_value={"i": -1, "u": 0},
                raw=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            )

    def test_default_gap_value_passes_validation(self):
        from temporaldata.regular_ts import _DEFAULT_GAP_VALUE, _validate_gap_value_dict

        _validate_gap_value_dict(_DEFAULT_GAP_VALUE)

    def test_single_gap_value_validation(self):
        ts = np.array([0.0, 0.1, 0.3])

        with pytest.raises(ValueError, match="cannot be losslessly stored"):
            RegularTimeSeries.from_gappy_timeseries(
                ts,
                sampling_rate=10.0,
                gap_value=np.nan,
                raw=np.array([1, 2, 3], dtype=int),
            )

        with pytest.raises(ValueError, match="cannot be losslessly stored"):
            RegularTimeSeries.from_gappy_timeseries(
                ts,
                sampling_rate=10.0,
                gap_value=3,
                raw=np.array([True, False, True]),
            )

    def test_gap_value_dict_validation(self):
        ts = np.array([0.0, 0.1, 0.3])
        raw = np.array([1.0, 2.0, 3.0])

        # Unknown key.
        with pytest.raises(ValueError, match="unsupported key"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"int": -1}, raw=raw
            )

        # 'i' must be an integer, not a float.
        with pytest.raises(ValueError, match=r"gap_value\['i'\] must be an integer"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"i": 2.5}, raw=raw
            )

        # 'b' must be a bool, not an int.
        with pytest.raises(ValueError, match=r"gap_value\['b'\] must be a bool"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"b": 1}, raw=raw
            )

        # 'u' must be non-negative.
        with pytest.raises(ValueError, match=r"gap_value\['u'\] must be non-negative"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"u": -1}, raw=raw
            )

        # 'u' must be an integer, not a float.
        with pytest.raises(ValueError, match=r"gap_value\['u'\] must be an integer"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"u": 1.5}, raw=raw
            )

        # 'f' must be a number, not a bool.
        with pytest.raises(ValueError, match=r"gap_value\['f'\] must be a number"):
            RegularTimeSeries.from_gappy_timeseries(
                ts, sampling_rate=10.0, gap_value={"f": True}, raw=raw
            )

        # Numpy scalars should be accepted.
        rts = RegularTimeSeries.from_gappy_timeseries(
            ts,
            sampling_rate=10.0,
            gap_value={"i": np.int32(-7), "f": np.float64(-1.5)},
            i=np.array([1, 2, 3], dtype=np.int32),
            f=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(rts.i, [1, 2, -7, 3])
        np.testing.assert_array_equal(rts.f, [1.0, 2.0, -1.5, 3.0])

    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="not available"):
            LazyRegularTimeSeries.from_gappy_timeseries(
                np.array([0.0, 0.1, 0.3]),
                sampling_rate=10.0,
                raw=np.array([1.0, 2.0, 3.0]),
            )


class _SupportsArrayWrapper:
    def __init__(self, values):
        self._data = np.asarray(values)

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)


class TestFromGappyTimeseriesCoercion:
    @pytest.mark.parametrize(
        "wrap",
        [list, tuple, pd.Series, _SupportsArrayWrapper],
        ids=["list", "tuple", "pd.Series", "SupportsArray"],
    )
    def test_coercion(self, wrap):
        rts = RegularTimeSeries.from_gappy_timeseries(
            wrap([0.0, 0.01, 0.03, 0.04]),
            sampling_rate=100.0,
            raw=wrap([1.0, 2.0, 3.0, 4.0]),
        )
        assert isinstance(rts.raw, np.ndarray)
        np.testing.assert_array_equal(rts.raw, [1.0, 2.0, np.nan, 3.0, 4.0])


class TestSliceGappy:
    """Slicing must trim leading/trailing gap samples and preserve internal ones.

    Fixture: 5 grid samples at 100Hz, idx 2 (time 0.02) missing.
        data:   [1, 2, nan, 3, 4]
        domain: [0.0, 0.02) U [0.03, 0.05)
    """

    @staticmethod
    def _build():
        ts = np.array([0.0, 0.01, 0.03, 0.04])
        raw = np.array([1.0, 2.0, 3.0, 4.0])
        return RegularTimeSeries.from_gappy_timeseries(ts, sampling_rate=100.0, raw=raw)

    @pytest.fixture(params=["regular", "lazy"])
    def rts(self, request, test_filepath):
        if request.param == "regular":
            yield self._build()
        else:
            with _make_lazy(
                self._build(), LazyRegularTimeSeries, test_filepath
            ) as data:
                yield data

    def test_slice_trims_leading_gap(self, rts):
        # Window starts inside the gap, so data[0] would otherwise be nan.
        s = rts.slice(0.018, 0.05, reset_origin=False)
        np.testing.assert_array_equal(s.raw, [3.0, 4.0])
        np.testing.assert_allclose(s.domain.start, [0.03])
        np.testing.assert_allclose(s.domain.end, [0.05])

    def test_slice_trims_trailing_gap(self, rts):
        # Window ends inside the gap, so data[-1] would otherwise be nan.
        s = rts.slice(0.0, 0.03, reset_origin=False)
        np.testing.assert_array_equal(s.raw, [1.0, 2.0])
        np.testing.assert_allclose(s.domain.start, [0.0])
        np.testing.assert_allclose(s.domain.end, [0.02])

    def test_slice_preserves_internal_gap(self, rts):
        # Window spans the gap; the interior nan must be kept.
        s = rts.slice(0.0, 0.05, reset_origin=False)
        np.testing.assert_array_equal(
            np.isnan(s.raw), [False, False, True, False, False]
        )
        np.testing.assert_allclose(s.domain.start, [0.0, 0.03])
        np.testing.assert_allclose(s.domain.end, [0.02, 0.05])

    def test_slice_inside_gap_is_empty(self, rts):
        s = rts.slice(0.022, 0.028, reset_origin=False)
        assert len(s) == 0
        assert s.domain.start[0] == s.domain.end[-1] == 0.03

    def test_slice_reset_origin(self, rts):
        s = rts.slice(0.018, 0.05, reset_origin=True)
        np.testing.assert_array_equal(s.raw, [3.0, 4.0])
        # data[0] (= 3) was at t=0.03; after reset by start=0.018, t=0.012.
        np.testing.assert_allclose(s.timestamps, [0.012, 0.022])
        np.testing.assert_allclose(s.domain.start, [0.012])
        np.testing.assert_allclose(s.domain.end, [0.032])

    def test_slice_spans_full_range_reset_origin(self, rts):
        s = rts.slice(0.0, 0.05, reset_origin=True)
        np.testing.assert_allclose(s.domain.start, [0.0, 0.03])
        np.testing.assert_allclose(s.domain.end, [0.02, 0.05])
        np.testing.assert_allclose(s.timestamps, [0.0, 0.01, 0.02, 0.03, 0.04])

    def test_slice_outside_domain(self, rts):
        s = rts.slice(1.0, 2.0, reset_origin=True)
        assert len(s) == 0
        assert s.domain.start[0] == s.domain.end[-1] == 0.0

    def test_lazy_consecutive_slices(self, test_filepath):
        # Nested slice: outer trims trailing gap, inner trims leading gap.
        with _make_lazy(self._build(), LazyRegularTimeSeries, test_filepath) as lazy:
            s = lazy.slice(0.0, 0.05, reset_origin=False).slice(
                0.018, 0.05, reset_origin=False
            )
            np.testing.assert_array_equal(s.raw, [3.0, 4.0])


class TestRegularTimeSeriesCoercion:
    def test_list(self):
        data = RegularTimeSeries(
            raw=[[float(i)] * 4 for i in range(10)],
            sampling_rate=10.0,
            domain=Interval(0.0, 1.0),
        )
        assert isinstance(data.raw, np.ndarray)
        assert data.raw.shape == (10, 4)
        assert len(data) == 10

    def test_tuple(self):
        data = RegularTimeSeries(
            raw=tuple([float(i)] * 4 for i in range(10)),
            sampling_rate=10.0,
            domain=Interval(0.0, 1.0),
        )
        assert isinstance(data.raw, np.ndarray)
        assert data.raw.shape == (10, 4)

    def test_pandas_dataframe(self):
        df = pd.DataFrame(np.zeros((100, 4)))
        data = RegularTimeSeries(
            raw=df,
            sampling_rate=10.0,
            domain=Interval(0.0, 10.0),
        )
        assert isinstance(data.raw, np.ndarray)
        assert data.raw.shape == (100, 4)
