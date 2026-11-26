import os
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any

import h5py
import numpy as np
import pytest

from torch_brain.data.utils import _get_array_dtype, _size_repr


class _UnicodeKeysMapping(Mapping):
    """Minimal mapping exposing a ``_unicode_keys`` attribute, which lazy objects have."""

    def __init__(self, data: dict[str, Any], unicode_keys: list[str]) -> None:
        self._data = data
        self._unicode_keys = unicode_keys

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


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


@pytest.fixture
def test_h5_file(test_filepath) -> Iterator[h5py.File]:
    with h5py.File(test_filepath, "w") as f:
        f.create_dataset("values", data=np.zeros((5, 3), dtype=np.float32))
        f.create_dataset(
            "names", data=np.array(["unit01", "unit02"], dtype="S6")
        )  # converted to bytes
        yield f


def test_get_array_dtype_for_ndarray():
    arr = np.array(["abc", "de"])
    assert _get_array_dtype(arr) == arr.dtype
    # h5_to_unicode has no effect on in-memory arrays
    assert _get_array_dtype(arr, h5_to_unicode=True) == arr.dtype


def test_get_array_dtype_for_h5py_dataset(test_h5_file: h5py.File):
    assert _get_array_dtype(test_h5_file["names"]) == np.dtype("S6")
    assert _get_array_dtype(test_h5_file["names"], h5_to_unicode=True) == np.dtype("U6")


@pytest.mark.parametrize("value", [[1, 2, 3], "abc", 5, {"a": 1}])
def test_get_array_dtype_invalid(value: Any):
    with pytest.raises(ValueError, match="Can only get the dtype of arrays"):
        _get_array_dtype(value)


def test_size_repr_h5py_dataset(test_h5_file: h5py.File):
    dset = test_h5_file["values"]
    assert _size_repr("x", dset) == "x: lazy array, shape=(5, 3), dtype=float32"
    assert (
        _size_repr("x", dset, first_dim=2)
        == "x: lazy array, shape=(2, 3), dtype=float32"
    )
    assert (
        _size_repr("names", test_h5_file["names"], h5_to_unicode=True)
        == "names: lazy array, shape=(2,), dtype=<U6"
    )
    with pytest.raises(ValueError, match="first_dim \\(6\\) is greater"):
        _size_repr("x", test_h5_file["values"], first_dim=6)


def test_size_repr_sequence():
    assert _size_repr("seq", [1, 2, 3]) == "seq: sequence, length=3"
    assert _size_repr("seq", (1, 2)) == "seq: sequence, length=2"


def test_size_repr_empty_mapping():
    assert _size_repr("m", {}) == "m={}"


def test_size_repr_single_item_mapping():
    value = {"a": np.zeros(3, dtype=np.int64)}
    assert _size_repr("m", value) == "m={a: array, shape=(3,), dtype=int64}"


def test_size_repr_single_item_mapping_unicode_keys(test_h5_file: h5py.File):
    value = _UnicodeKeysMapping(
        {"names": test_h5_file["names"]}, unicode_keys=["names"]
    )
    assert _size_repr("m", value) == "m={names: lazy array, shape=(2,), dtype=<U6}"

    # without the key in _unicode_keys, the stored byte dtype is reported
    value = _UnicodeKeysMapping({"names": test_h5_file["names"]}, unicode_keys=[])
    assert _size_repr("m", value) == "m={names: lazy array, shape=(2,), dtype=|S6}"


def test_size_repr_multi_item_mapping():
    value = {"a": 1, "b": "text"}
    expected = "  m={\n    a=1,\n    b='text'\n  }"
    assert _size_repr("m", value, indent=2) == expected


def test_size_repr_nested_mapping():
    value = {"inner": {"a": 1}}
    expected = "m={\n  inner={a=1}\n}"
    assert _size_repr("m", value) == expected


def test_size_repr_multi_item_mapping_unicode_keys(test_h5_file: h5py.File):
    value = _UnicodeKeysMapping(
        {"names": test_h5_file["names"], "values": test_h5_file["values"]},
        unicode_keys=["names"],
    )
    expected = (
        "m={\n"
        "  names: lazy array, shape=(1,), dtype=<U6,\n"
        "  values: lazy array, shape=(1, 3), dtype=float32\n"
        "}"
    )
    assert _size_repr("m", value, first_dim=1) == expected
