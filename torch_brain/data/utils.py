from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np


class DeferredH5Dataset:
    r"""A reference to a dataset in an HDF5 group. The dataset is only opened when
    it is first accessed, which avoids opening every dataset when lazy loading an
    object from an HDF5 file. Supports indexing, ``shape``, ``dtype`` and ``len()``.

    Args:
        group: The HDF5 group containing the dataset.
        name: The name of the dataset in the group.
    """

    def __init__(self, group: h5py.Group, name: str):
        self._group = group
        self._name = name
        self._dataset = None

    @property
    def dataset(self) -> h5py.Dataset:
        r"""Returns the :obj:`h5py.Dataset`, opening it if needed."""
        if self._dataset is None:
            # the low-level API is faster than group[name]
            dsid = h5py.h5d.open(self._group.id, self._name.encode("utf-8"))
            self._dataset = h5py.Dataset(dsid)
        return self._dataset

    @property
    def valid(self) -> bool:
        r"""Returns ``True`` if the HDF5 file is still open."""
        return bool(self._group.id.valid)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self.dataset.dtype

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # h5py objects cannot be deepcopied
        return self

    def __repr__(self) -> str:
        if not self.valid:
            return f'<DeferredH5Dataset "{self._name}" (file closed)>'
        if self._dataset is None:
            return f'<DeferredH5Dataset "{self._name}" (not opened)>'
        return f"<DeferredH5Dataset of {self._dataset!r}>"


def _size_repr(key: Any, value: Any, indent: int = 0, first_dim: int = None) -> str:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        out = f": array, shape={value.shape}, dtype={value.dtype}"
    elif isinstance(value, DeferredH5Dataset):
        shape = value.shape
        if first_dim is not None:
            if first_dim > shape[0]:
                raise ValueError(
                    f"first_dim ({first_dim}) is greater than the dataset's first dimension ({shape[0]})."
                )
            shape = tuple([int(first_dim), *shape[1:]])
        out = f": lazy array, shape={shape}, dtype={value.dtype}"
    elif isinstance(value, str):
        out = f"='{value}'"
    elif isinstance(value, Sequence):
        out = f": sequence, length={len(value)}"
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "={}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [_size_repr(k, v, 0, first_dim) for k, v in value.items()]
        out = "={ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [_size_repr(k, v, indent + 2, first_dim) for k, v in value.items()]
        out = "={\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        out = f"={value}"
    key = str(key).replace("'", "")
    return f"{pad}{key}{out}"


def _validate_select_by_mask_input(mask, length):
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy array (bool, 1D)")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got {mask.ndim}D mask")
    if mask.dtype != bool:
        raise ValueError(f"mask must be boolean, got {mask.dtype}")

    if len(mask) != length:
        raise ValueError(
            f"mask length {len(mask)} does not match object length ({length})"
        )
