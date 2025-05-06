import pytest
import os
import copy
import h5py
import numpy as np
import pandas as pd
import tempfile
import logging
from temporaldata import (
    ArrayDict,
    LazyArrayDict,
    IrregularTimeSeries,
    LazyIrregularTimeSeries,
    RegularTimeSeries,
    LazyRegularTimeSeries,
    Interval,
    LazyInterval,
    Data,
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


def test_array_dict():
    data = ArrayDict(
        unit_id=np.array(["unit01", "unit02"]),
        brain_region=np.array(["M1", "M1"]),
        waveform_mean=np.random.random((2, 48)),
    )

    assert data.keys() == ["unit_id", "brain_region", "waveform_mean"]
    assert len(data) == 2
    assert "unit_id" in data
    assert "brain_region" in data
    assert "waveform_mean" in data

    # setting an incorrect attribute
    with pytest.raises(AssertionError):
        data.dummy_list = [1, 2]

    with pytest.raises(ValueError):
        data.wrong_len = np.array([0, 1, 2, 3])

    with pytest.raises(AssertionError):
        data = ArrayDict(unit_id=["unit01", "unit02"])

    with pytest.raises(ValueError):
        data = ArrayDict(
            unit_id=np.array(["unit01", "unit02", "unit03"]),
            brain_region=np.array(["M1"]),
        )

    # testing an empty ArrayDict
    data = ArrayDict()

    with pytest.raises(ValueError):
        len(data)

    data.unit_id = np.array(["unit01", "unit02", "unit03"])
    assert len(data) == 3


def test_array_dict_select_by_mask():
    # test masking
    data = ArrayDict(
        unit_id=np.array(["unit01", "unit02", "unit03", "unit04"]),
        brain_region=np.array(["PMd", "M1", "PMd", "M1"]),
        waveform_mean=np.ones((4, 48)),
    )

    mask = data.brain_region == "PMd"

    data = data.select_by_mask(mask)

    assert len(data) == 2
    assert np.array_equal(data.unit_id, np.array(["unit01", "unit03"]))
    assert np.array_equal(data.brain_region, np.array(["PMd", "PMd"]))

    mask = np.array([False, False])
    data = data.select_by_mask(mask)
    assert len(data) == 0
    assert data.unit_id.size == 0
    assert data.waveform_mean.shape == (0, 48)


def test_lazy_array_dict(test_filepath):
    data = ArrayDict(
        unit_id=np.array(["unit01", "unit02", "unit03", "unit04"]),
        brain_region=np.array([b"PMd", b"M1", b"PMd", b"M1"]),
        waveform_mean=np.tile(np.arange(4)[:, np.newaxis], (1, 48)),
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyArrayDict.from_hdf5(f)

        assert len(data) == 4

        # make sure that nothing is loaded yet
        assert all(isinstance(data.__dict__[key], h5py.Dataset) for key in data.keys())

        # try loading one attribute
        unit_id = data.unit_id
        # make sure that the attribute is loaded
        assert isinstance(unit_id, np.ndarray)
        # make sure that the attribute is loaded correctly
        assert np.array_equal(
            unit_id, np.array(["unit01", "unit02", "unit03", "unit04"])
        )
        # make sure that the loaded attribute replaced the h5py.Dataset reference
        assert isinstance(data.__dict__["unit_id"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "unit_id"
        )

        # make sure that the string arrays are loaded correctly
        assert data.brain_region.dtype == np.dtype("<S3")
        assert data.unit_id.dtype == np.dtype("<U6")

        assert data.__class__ == LazyArrayDict

        data.waveform_mean
        # final attribute was accessed, the object should automatically convert to ArrayDict
        assert data.__class__ == ArrayDict

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyArrayDict.from_hdf5(f)

        # try masking
        mask = data.brain_region == b"PMd"
        data = data.select_by_mask(mask)

        # make sure only brain_region was loaded
        assert isinstance(data.__dict__["brain_region"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "brain_region"
        )

        assert np.array_equal(data.brain_region, np.array([b"PMd", b"PMd"]))
        assert len(data) == 2

        assert np.array_equal(data._lazy_ops["mask"], mask)

        # load another attribute
        unit_id = data.unit_id
        assert isinstance(unit_id, np.ndarray)
        assert np.array_equal(unit_id, np.array(["unit01", "unit03"]))

        # mask again!
        mask = data.unit_id == "unit01"

        # make a new object data2 (mask is not inplace)
        data2 = data.select_by_mask(mask)

        assert len(data2) == 1
        # make sure that the attribute was never accessed, is still not accessed
        assert isinstance(data2.__dict__["waveform_mean"], h5py.Dataset)

        # check if the mask was applied twice correctly!
        assert np.allclose(data2.waveform_mean, np.zeros((1, 48)))

        # make sure that data is still intact
        assert len(data) == 2
        assert np.array_equal(data.unit_id, np.array(["unit01", "unit03"]))


def test_array_dict_from_dataframe():
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "col1": np.array([1, 2, 3]),  # ndarray
            "col2": [np.array(4), np.array(5), np.array(6)],  # list of ndarrays
            "col3": ["a", "b", "c"],  # list of strings
        }
    )

    # Call the from_dataframe method
    a = ArrayDict.from_dataframe(df)

    # Assert the correctness of the conversion
    assert np.array_equal(a.col1, np.array([1, 2, 3]))
    assert np.array_equal(a.col2, np.array([4, 5, 6]))
    assert np.array_equal(a.col3, np.array(["a", "b", "c"]))

    # Test unsigned_to_long parameter
    df_unsigned = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}, dtype=np.uint32)

    a_unsigned = ArrayDict.from_dataframe(df_unsigned, unsigned_to_long=False)

    assert np.array_equal(a_unsigned.col1, np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(a_unsigned.col2, np.array([4, 5, 6], dtype=np.int32))

    df_non_ascii = pd.DataFrame(
        {
            "col1": [
                "Ä",
                "é",
                "é",
            ],  # not ASCII, should catch thsi and not convert to ndarray
            "col2": [
                "d",
                "e",
                "f",
            ],  # should be converted to fixed length ASCII "S" type ndarray
        }
    )

    a_with_non_ascii_col = ArrayDict.from_dataframe(df_non_ascii)

    assert hasattr(a_with_non_ascii_col, "col2")
    assert not hasattr(a_with_non_ascii_col, "col1")


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


def test_interval_select_by_interval():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        go_cue_time=np.array([0.5, 1.5, 2.5]),
        drifting_gratings_dir=np.array([0, 45, 90]),
        timekeys=["start", "end", "go_cue_time"],
    )

    selection_interval = Interval(
        start=np.array([0.2, 2.5]),
        end=np.array([0.4, 3.12]),
    )
    data = data.select_by_interval(selection_interval)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0.0, 2.0]))
    assert np.allclose(data.end, np.array([1.0, 3.0]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 2.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([0, 90]))


def test_interval_iter():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        some_other_attribute=np.array([0, 1, 2]),
    )

    assert list(data) == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    # test a single interval
    data = Interval(0.0, 1.0)
    assert list(data) == [(0.0, 1.0)]

    # test an empty interval
    data = Interval(np.array([]), np.array([]))
    assert list(data) == []


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


def test_interval():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        go_cue_time=np.array([0.5, 1.5, 2.5]),
        drifting_gratings_dir=np.array([0, 45, 90]),
        timekeys=["start", "end", "go_cue_time"],
    )

    assert data.keys() == ["start", "end", "go_cue_time", "drifting_gratings_dir"]
    assert len(data) == 3

    assert data.is_sorted()
    assert data.is_disjoint()

    # setting an incorrect attribute
    with pytest.raises(ValueError):
        data.wrong_len = np.array([0, 1, 2, 3])

    with pytest.raises(AssertionError):
        data = Interval(
            start=np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6]),
            end=np.array([1, 2, 3, 4, 5, 6]),
        )


def test_interval_select_by_mask():
    # test masking
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    mask = data.drifting_gratings_dir == 90

    data = data.select_by_mask(mask)

    assert len(data) == 3
    assert np.array_equal(data.start, np.array([2, 5, 7]))
    assert np.array_equal(data.end, np.array([3, 6, 8]))


def test_interval_slice():
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    data = data.slice(2.0, 6.0)

    assert len(data) == 4
    assert np.allclose(data.start, np.array([0, 1, 2, 3]))
    assert np.allclose(data.end, np.array([1, 2, 3, 4]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5, 2.5, 3.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45, 180, 90]))

    data = data.slice(0.0, 2.0)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0, 1]))
    assert np.allclose(data.end, np.array([1, 2]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))

    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    data = data.slice(2.0, 6.0, reset_origin=False)

    assert len(data) == 4
    assert np.allclose(data.start, np.array([2, 3, 4, 5]))
    assert np.allclose(data.end, np.array([3, 4, 5, 6]))
    assert np.allclose(data.go_cue_time, np.array([2.5, 3.5, 4.5, 5.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45, 180, 90]))

    data = data.slice(2.0, 4.0, reset_origin=True)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0, 1]))
    assert np.allclose(data.end, np.array([1, 2]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))


def test_lazy_interval(test_filepath):
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        assert len(data) == 9

        # make sure that nothing is loaded yet
        assert all(isinstance(data.__dict__[key], h5py.Dataset) for key in data.keys())

        # try loading one attribute
        start = data.start
        # make sure that the attribute is loaded
        assert isinstance(start, np.ndarray)
        # make sure that the attribute is loaded correctly
        assert np.array_equal(start, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        # make sure that the loaded attribute replaced the h5py.Dataset reference
        assert isinstance(data.__dict__["start"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "start"
        )

        assert data.__class__ == LazyInterval

        data.end
        assert data.__class__ == LazyInterval

        data.go_cue_time
        data.drifting_gratings_dir
        # final attribute was accessed, the object should automatically convert to ArrayDict
        assert data.__class__ == Interval

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        # try masking
        mask = data.drifting_gratings_dir == 90
        data = data.select_by_mask(mask)

        # make sure only brain_region was loaded
        assert isinstance(data.__dict__["drifting_gratings_dir"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "drifting_gratings_dir"
        )

        assert len(data) == 3
        assert np.array_equal(data.start, np.array([2, 5, 7]))

        assert np.array_equal(data._lazy_ops["mask"], mask)

        # load another attribute
        go_cue_time = data.go_cue_time
        assert isinstance(go_cue_time, np.ndarray)
        assert np.array_equal(go_cue_time, np.array([2.5, 5.5, 7.5]))

        # mask again!
        mask = data.start >= 6

        # make a new object data2 (mask is not inplace)
        data2 = data.select_by_mask(mask)

        assert len(data2) == 1
        # make sure that the attribute was never accessed, is still not accessed
        assert isinstance(data2.__dict__["end"], h5py.Dataset)

        # check if the mask was applied twice correctly!
        assert np.allclose(data2.end, np.array([8]))

        # make sure that data is still intact
        assert len(data) == 3
        assert np.array_equal(data.end, np.array([3, 6, 8]))

    del data, data2

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        data = data.slice(2.0, 6.0)

        assert len(data) == 4
        assert np.allclose(data.start, np.array([0, 1, 2, 3]))
        assert np.allclose(data.end, np.array([1, 2, 3, 4]))

        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key not in ["start", "end"]
        )

        assert np.allclose(data.go_cue_time, np.array([0.5, 1.5, 2.5, 3.5]))

        data = data.slice(0.0, 2.0)

        assert len(data) == 2
        assert np.allclose(data.start, np.array([0, 1]))
        assert np.allclose(data.end, np.array([1, 2]))
        assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
        assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))

    del data

    # try slicing and masking

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        data = data.slice(2.0, 6.0)
        mask = data.drifting_gratings_dir == 90
        data = data.select_by_mask(mask)

        assert np.allclose(data.start, np.array([0, 3]))


def test_data():
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
    )

    assert data.keys() == [
        "session_id",
        "spikes",
        "lfp",
        "units",
        "trials",
        "drifting_gratings_imgs",
    ]

    data = data.slice(1.0, 3.0)
    assert data.absolute_start == 1.0

    assert ["session_id", "spikes", "lfp", "units", "trials", "drifting_gratings_imgs"]

    assert len(data.spikes) == 3
    assert np.allclose(data.spikes.timestamps, np.array([1.1, 1.2, 1.3]))
    assert np.allclose(data.spikes.unit_index, np.array([0, 1, 2]))

    assert len(data.lfp) == 500

    assert len(data.trials) == 2
    assert np.allclose(data.trials.start, np.array([0, 1]))

    data = data.slice(0.4, 1.4)
    assert data.absolute_start == 1.4


def test_data_copy():
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        some_numpy_array=np.array([1, 2, 3]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
    )

    ### test copy
    data_copy = copy.copy(data)
    data_copy.some_numpy_array[0] = 10
    # this is a shallow copy, so the original object should be modified
    assert data.some_numpy_array[0] == 10

    data_copy.spikes.unit_index[0] = 2
    # this is a shallow copy, so the original object should be modified
    assert data.spikes.unit_index[0] == 2

    data_copy.spikes.unit_index = np.array([0, 0, 0, 0, 0, 0])
    # the unit_index variable is not shared between the two objects
    assert data.spikes.unit_index[0] == 2

    ### test deepcopy
    data_deepcopy = copy.deepcopy(data)
    data_deepcopy.some_numpy_array[1] = 20
    # this is a deep copy, so the original object should not be modified
    assert data.some_numpy_array[1] == 2

    data_deepcopy.spikes.unit_index[1] = 2
    # this is a deep copy, so the original object should not be modified
    assert data.spikes.unit_index[1] == 0


def test_lazy_data_copy(test_filepath):
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        some_numpy_array=np.array([1, 2, 3]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)
        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)

        # this will copy all references to any h5py datasets
        data_copy = copy.copy(data)
        data_copy.some_numpy_array[0] = 10
        # TODO Data does not lazy load numpy arrays that are not wrapped in an
        # ArrayDict object, this will change in the future.
        # because some_numpy_array is not a h5py dataset, changing it will affect
        # the original object
        assert isinstance(data.__dict__["some_numpy_array"], np.ndarray)
        # this is a shallow copy, so the original object should be modified
        assert data.some_numpy_array[0] == 10

        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)
        data_copy.spikes.unit_index[0] = 2
        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)
        # this is a shallow copy, but the unit_index is a h5py dataset, so
        # the original object will not be modified
        assert data.spikes.unit_index[0] == 0

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)

        data_deepcopy = copy.deepcopy(data)
        data_deepcopy.some_numpy_array[0] = 10
        # this is a deep copy, so the original object should not be modified
        assert data.some_numpy_array[0] == 1

        data_deepcopy.spikes.unit_index[0] = 2
        # this is a deep copy, so the original object should not be modified
        assert data.spikes.unit_index[0] == 0


def test_data_absolute_start(test_filepath):
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        some_numpy_array=np.array([1, 2, 3]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
    )

    data = data.slice(1.0, 3.0)

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)
        assert data.absolute_start == 1.0


def test_timeless_data(test_filepath):
    # when defining a Data object that has no time-based attributes, we do no need to
    # specify a domain
    subject = Data(
        id="jenkins",
        age=5.0,
        species="HUMAN",
        description="À89!ÜÞ",
        image=np.ones((32, 32, 3)),
    )

    # we cannot slice this object because it has no domain or time-based attributes
    with pytest.raises(ValueError):
        subject.slice(0.1, 0.2)

    with h5py.File(test_filepath, "w") as f:
        subject.to_hdf5(f)

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f)

        assert data.id == "jenkins"
        assert data.age == 5.0
        assert data.species == "HUMAN"
        assert data.description == "À89!ÜÞ"

        # TODO(mehdi) image is a numpy array so it should be lazy loaded
        # assert isinstance(data.__dict__["image"], h5py.Dataset)
        assert np.allclose(data.image, np.ones((32, 32, 3)))

    data = Data(
        subject=subject,
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f)

        assert data.subject.id == "jenkins"
        assert data.subject.age == 5.0
        assert data.subject.species == "HUMAN"
        assert np.allclose(data.subject.image, np.ones((32, 32, 3)))

        assert len(data.spikes) == 6
        assert np.allclose(
            data.spikes.timestamps, np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3])
        )
        assert np.allclose(data.spikes.unit_index, np.array([0, 0, 1, 0, 1, 2]))


def test_precision(caplog):
    with caplog.at_level(logging.WARNING):
        spikes = IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3], dtype=np.float64),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        )
    assert caplog.text == ""

    with caplog.at_level(logging.WARNING):
        spikes = IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3], dtype=np.float16),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        )
    assert (
        f"timestamps is of type {spikes.timestamps.dtype} not of type float64."
        in caplog.text
    )

    with caplog.at_level(logging.WARNING):
        trials = Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        )
        assert (
            f"start is of type {trials.start.dtype} not of type float64." in caplog.text
        )

    lfp = RegularTimeSeries(
        raw=np.zeros((1000, 3)),
        sampling_rate=250.0,
        domain="auto",
    )
    assert lfp.timestamps.dtype == np.float64


def test_nested_attributes():
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
    )

    # test get_nested_attribute
    out = data.get_nested_attribute("spikes.timestamps")
    assert np.allclose(out, data.spikes.timestamps)

    out = data.get_nested_attribute("units.id")
    for i, id in enumerate(out):
        assert id == data.units.id[i]

    # test has_nested_attribute
    assert data.has_nested_attribute("spikes.timestamps")
    assert data.has_nested_attribute("units.id")
    assert not data.has_nested_attribute("spikes.unit_index")

    # test that an error is raised if the attribute does not exist
    with pytest.raises(AttributeError):
        data.get_nested_attribute("spikes.unit_index")

    # test that it is possible to access attributes that are not nested
    assert data.get_nested_attribute("session_id") == "session_0"
    assert data.has_nested_attribute("session_id")


def test_data_has_nested_attribute_lazy(test_filepath):
    """Tests the Data.has_nested_attribute method with lazily loaded objects."""
    data_to_save = Data(
        session_id="session_lazy_hsna_test",
        some_numpy_array=np.array([10, 20, 30]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["u0_hsna", "u1_hsna"]),
            type=np.array(["typeA_hsna", "typeB_hsna"]),
        ),
        nested_data=Data(
            level2_attr="hello_nested_hsna",
            level2_array_dict=ArrayDict(l2_field=np.array([100, 200])),
            level2_primitive=42,
            domain=Interval(0.0, 1.0),
        ),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as f:
        data_to_save.to_hdf5(f)
    del data_to_save

    with h5py.File(test_filepath, "r") as f:
        lazy_data = Data.from_hdf5(f, lazy=True)

        # Pre-checks for laziness: ensure some attributes are indeed h5py.Dataset
        assert isinstance(lazy_data.spikes.__dict__["unit_index"], h5py.Dataset)
        assert isinstance(lazy_data.units.__dict__["id"], h5py.Dataset)
        assert isinstance(
            lazy_data.nested_data.level2_array_dict.__dict__["l2_field"], h5py.Dataset
        )

        # === Test existing paths ===
        assert lazy_data.has_nested_attribute("session_id")
        assert lazy_data.has_nested_attribute("some_numpy_array")
        assert lazy_data.has_nested_attribute("spikes")
        assert lazy_data.has_nested_attribute("spikes.unit_index")
        assert lazy_data.has_nested_attribute("units.id")
        assert lazy_data.has_nested_attribute("nested_data")
        assert lazy_data.has_nested_attribute("nested_data.level2_attr")
        assert lazy_data.has_nested_attribute("nested_data.level2_array_dict")
        assert lazy_data.has_nested_attribute("nested_data.level2_array_dict.l2_field")
        assert lazy_data.has_nested_attribute("nested_data.level2_primitive")

        # Check attributes remain lazy after has_nested_attribute calls
        assert isinstance(
            lazy_data.spikes.__dict__["unit_index"], h5py.Dataset
        ), "spikes.unit_index was loaded"
        assert isinstance(
            lazy_data.units.__dict__["id"], h5py.Dataset
        ), "units.id was loaded"
        assert isinstance(
            lazy_data.nested_data.level2_array_dict.__dict__["l2_field"], h5py.Dataset
        ), "nested_data.l2_array_dict.l2_field was loaded"

        # === Test non-existent paths (should return False) ===
        assert not lazy_data.has_nested_attribute("non_existent_toplevel")
        assert not lazy_data.has_nested_attribute("spikes.foo_bar_baz")
        assert not lazy_data.has_nested_attribute("units.id.deeper_false")
        assert not lazy_data.has_nested_attribute(
            "nested_data.level2_array_dict.l2_field.deeper_false_too"
        )
        assert not lazy_data.has_nested_attribute("nested_data.non_existent_attr")

        # === Test empty path ===
        assert not lazy_data.has_nested_attribute("")

        # === Test paths that should fail due to AttributeError on __dict__ access for primitives ===
        with pytest.raises(
            AttributeError, match="'str' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("session_id.foo")

        with pytest.raises(
            AttributeError, match="'numpy.ndarray' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("some_numpy_array.shape")
        with pytest.raises(
            AttributeError, match="'numpy.ndarray' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("some_numpy_array.foo")

        with pytest.raises(
            AttributeError, match="'numpy.int64' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("nested_data.level2_primitive.foo")
