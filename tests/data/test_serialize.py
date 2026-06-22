import datetime
import os
import tempfile
from enum import Enum

import h5py
import pytest

from torch_brain.data import Data


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


def test_serialize(test_filepath):

    class MyEnum(Enum):
        A = 1
        B = 2

    d = Data(
        id="test",
        description="test",
        special_item=MyEnum.A,
        special_list=[MyEnum.A, MyEnum.B],
        special_tuple=(MyEnum.A, MyEnum.B),
        nested_special_objects=Data(
            id="nested",
            special_item=MyEnum.B,
            special_list=[MyEnum.B, MyEnum.A],
            special_tuple=(MyEnum.B, MyEnum.A),
        ),
    )

    def my_enum_serialize_fn(obj, serialize_fn_map=None):
        return obj.name

    with h5py.File(test_filepath, "w") as file:
        d.to_hdf5(file, serialize_fn_map={Enum: my_enum_serialize_fn})

    del d

    with h5py.File(test_filepath, "r") as file:
        d = Data.from_hdf5(file)

        assert d.id == "test"
        assert d.special_item == "A"
        assert all(d.special_list == ["A", "B"])
        assert all(d.special_tuple == ("A", "B"))
        assert d.nested_special_objects.special_item == "B"
        assert all(d.nested_special_objects.special_list == ["B", "A"])
        assert all(d.nested_special_objects.special_tuple == ("B", "A"))


def test_default_datetime_serialize(test_filepath):
    # When no serialize_fn_map is passed, to_hdf5 should fall back to the default
    # map, which serializes datetime.datetime objects to their string form.
    timestamp = datetime.datetime(2021, 1, 2, 3, 4, 5)

    d = Data(
        id="test",
        recording_date=timestamp,
        nested=Data(recording_date=timestamp),
    )

    with h5py.File(test_filepath, "w") as file:
        d.to_hdf5(file)

    del d

    with h5py.File(test_filepath, "r") as file:
        d = Data.from_hdf5(file)

        assert d.recording_date == str(timestamp)
        assert d.nested.recording_date == str(timestamp)
