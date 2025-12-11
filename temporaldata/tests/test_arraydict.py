import pytest
import os
import copy
import h5py
import numpy as np
import pandas as pd
import tempfile
import logging
from temporaldata import ArrayDict, LazyArrayDict


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
