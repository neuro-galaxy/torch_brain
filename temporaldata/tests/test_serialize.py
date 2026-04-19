import pytest
import os
import h5py
import tempfile
from temporaldata import Data
from enum import Enum


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
