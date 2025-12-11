import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import IrregularTimeSeries, LazyIrregularTimeSeries, Interval


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


def test_irregular_timeseries():
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    assert data.keys() == ["timestamps", "unit_index", "waveforms"]
    assert len(data) == 6

    assert np.allclose(data.domain.start, np.array([0.1]))
    assert np.allclose(data.domain.end, np.array([0.6]))

    assert data.is_sorted()

    # setting an incorrect attribute
    with pytest.raises(ValueError):
        data.wrong_len = np.array([0, 1, 2, 3])

    with pytest.raises(AssertionError):
        data = IrregularTimeSeries(
            timestamps=np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6]),
            domain="auto",
        )


def test_irregular_timeseries_select_by_mask():
    # test masking
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    mask = data.unit_index == 0

    data = data.select_by_mask(mask)

    assert len(data) == 3
    assert np.array_equal(data.timestamps, np.array([0.1, 0.2, 0.4]))


def test_irregular_timeseries_slice():
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    data = data.slice(0.2, 0.5)

    assert len(data) == 3
    assert np.allclose(data.timestamps, np.array([0.0, 0.1, 0.2]))
    assert np.allclose(data.unit_index, np.array([0, 1, 0]))

    assert len(data.domain) == 1
    assert data.domain.start[0] == 0.0
    assert data.domain.end[0] == 0.3

    data = data.slice(0.05, 0.25)

    assert len(data) == 2
    assert np.allclose(data.timestamps, np.array([0.05, 0.15]))
    assert np.allclose(data.unit_index, np.array([1, 0]))

    assert len(data.domain) == 1
    assert data.domain.start[0] == 0.0
    assert data.domain.end[0] == 0.2

    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    data = data.slice(0.2, 0.5, reset_origin=False)

    assert len(data) == 3
    assert np.allclose(data.timestamps, np.array([0.2, 0.3, 0.4]))
    assert np.allclose(data.unit_index, np.array([0, 1, 0]))

    assert len(data.domain) == 1
    assert data.domain.start[0] == 0.2
    assert data.domain.end[0] == 0.5

    data = data.slice(0.25, 0.45, reset_origin=False)

    assert len(data) == 2
    assert np.allclose(data.timestamps, np.array([0.3, 0.4]))
    assert np.allclose(data.unit_index, np.array([1, 0]))

    assert len(data.domain) == 1
    assert data.domain.start[0] == 0.25
    assert data.domain.end[0] == 0.45


def test_irregular_timeseries_select_by_interval():
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    selection_interval = Interval(
        start=np.array([0.2, 0.5]),
        end=np.array([0.4, 0.6]),
    )
    data = data.select_by_interval(selection_interval)

    assert len(data) == 3
    assert np.allclose(data.timestamps, np.array([0.2, 0.3, 0.5]))
    assert np.allclose(data.unit_index, np.array([0, 1, 1]))

    assert len(data.domain) == 2
    assert np.allclose(data.domain.start, selection_interval.start)
    assert np.allclose(data.domain.end, selection_interval.end)


def test_irregular_timeseries_lazy_select_by_interval(test_filepath):
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        values=np.array([0, 1, 2, 3, 4, 5]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyIrregularTimeSeries.from_hdf5(f)

        selection_interval = Interval(
            start=np.array([0.2, 0.5]),
            end=np.array([0.4, 0.6]),
        )
        data = data.select_by_interval(selection_interval)

        assert len(data) == 3
        assert np.allclose(data.timestamps, np.array([0.2, 0.3, 0.5]))
        assert np.allclose(data.unit_index, np.array([0, 1, 1]))

        assert len(data.domain) == 2
        assert np.allclose(data.domain.start, selection_interval.start)
        assert np.allclose(data.domain.end, selection_interval.end)


def test_irregular_timeseries_sortedness():
    a = IrregularTimeSeries(np.array([0.0, 1.0, 2.0]), domain="auto")
    assert a.is_sorted()

    a.timestamps = np.array([0.0, 2.0, 1.0])
    assert not a.is_sorted()

    a = a.slice(0, 1.5)
    assert np.allclose(a.timestamps, np.array([0, 1]))


def test_lazy_irregular_timeseries(test_filepath):
    data = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        values=np.array([0, 1, 2, 3, 4, 5]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyIrregularTimeSeries.from_hdf5(f)

        assert len(data) == 6

        # make sure that nothing is loaded yet
        assert all(isinstance(data.__dict__[key], h5py.Dataset) for key in data.keys())

        # try loading one attribute
        unit_index = data.unit_index
        # make sure that the attribute is loaded
        assert isinstance(unit_index, np.ndarray)
        # make sure that the attribute is loaded correctly
        assert np.array_equal(unit_index, np.array([0, 0, 1, 0, 1, 2]))
        # make sure that the loaded attribute replaced the h5py.Dataset reference
        assert isinstance(data.__dict__["unit_index"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "unit_index"
        )

        assert data.__class__ == LazyIrregularTimeSeries

        data.timestamps
        assert data.__class__ == LazyIrregularTimeSeries

        data.waveforms
        data.values
        # final attribute was accessed, the object should automatically convert to ArrayDict
        assert data.__class__ == IrregularTimeSeries

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyIrregularTimeSeries.from_hdf5(f)

        # try masking
        mask = data.unit_index != 1
        data = data.select_by_mask(mask)

        # make sure only brain_region was loaded
        assert isinstance(data.__dict__["unit_index"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "unit_index"
        )

        assert np.array_equal(data.timestamps, np.array([0.1, 0.2, 0.4, 0.6]))
        assert len(data) == 4

        assert np.array_equal(data._lazy_ops["mask"], mask)

        # load another attribute
        values = data.values
        assert isinstance(values, np.ndarray)
        assert np.array_equal(values, np.array([0, 1, 3, 5]))

        # mask again!
        mask = data.unit_index == 2

        # make a new object data2 (mask is not inplace)
        data2 = data.select_by_mask(mask)

        assert len(data2) == 1
        # make sure that the attribute was never accessed, is still not accessed
        assert isinstance(data2.__dict__["waveforms"], h5py.Dataset)

        # check if the mask was applied twice correctly!
        assert np.allclose(data2.waveforms, np.zeros((1, 48)))

        # make sure that data is still intact
        assert len(data) == 4
        assert np.array_equal(data.unit_index, np.array([0, 0, 0, 2]))

        # try rewriting an attribute
        data.unit_index = np.array([0, -1, 2])[data.unit_index]

    del data, data2

    with h5py.File(test_filepath, "r") as f:
        data = LazyIrregularTimeSeries.from_hdf5(f)

        data = data.slice(0.15, 0.6)
        assert np.allclose(data.timestamps, np.array([0.05, 0.15, 0.25, 0.35]))

        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "timestamps"
        )

        assert np.allclose(data.values, np.array([1, 2, 3, 4]))

        data = data.slice(0.15, 0.3)

        assert np.allclose(data.timestamps, np.array([0.0, 0.1]))
        assert np.allclose(data.values, np.array([2, 3]))
        assert np.allclose(data.unit_index, np.array([1, 0]))

    del data

    # try slicing and masking

    with h5py.File(test_filepath, "r") as f:
        data = LazyIrregularTimeSeries.from_hdf5(f)

        data = data.slice(0.15, 0.6)
        mask = data.unit_index == 0
        data = data.select_by_mask(mask)

        assert np.allclose(data.timestamps, np.array([0.05, 0.25]))
        assert np.allclose(data.values, np.array([1, 3]))
