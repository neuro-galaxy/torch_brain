from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np


class DeferredH5Dataset:
    r"""Reference to an HDF5 dataset that is only opened when first needed.

    Lazy objects store one of these for every attribute that has not been loaded
    yet. Opening an :obj:`h5py.Dataset` costs ~20 µs, so opening all of them in
    ``from_hdf5`` dominated loading objects with many attributes, even when only
    a few were ever read. Supports the subset of the :obj:`h5py.Dataset` API the
    lazy classes use: indexing, ``shape``, ``dtype`` and ``len()``.
    """

    # no __slots__: Data.has_nested_attribute walks __dict__, as on h5py.Dataset

    def __init__(self, group: h5py.Group, name: str):
        self._group = group
        self._name = name
        self._dataset = None

    @property
    def dataset(self) -> h5py.Dataset:
        if self._dataset is None:
            self._dataset = self._group[self._name]
        return self._dataset

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
        # read-only reference into an open file: share it, like h5py.Dataset
        return self

    def __repr__(self) -> str:
        if not self._group.id.valid:
            return f'<DeferredH5Dataset "{self._name}" (file closed)>'
        return f"<DeferredH5Dataset of {self.dataset!r}>"


def _size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [_size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [_size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        out = str(value)
    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


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
