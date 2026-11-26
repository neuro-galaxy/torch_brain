from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np


def _get_array_dtype(value: Any, h5_to_unicode: bool = False) -> np.dtype:
    """
    Get the dtype of a numpy array or h5py dataset.

    Args:
        value: The array or dataset to get the dtype of.
        h5_to_unicode: If True, convert byte string dtypes to unicode dtypes
            for h5py Dataset objects only.

    Returns:
        The dtype of the array or dataset.
    """
    if isinstance(value, np.ndarray):
        dtype = value.dtype
    elif isinstance(value, h5py.Dataset):
        dtype = value.dtype
        if h5_to_unicode:
            dtype = np.dtype(f"U{dtype.itemsize}")
    else:
        raise ValueError("Can only get the dtype of arrays.")
    return dtype


def _size_repr(
    key: Any,
    value: Any,
    indent: int = 0,
    first_dim: int = None,
    h5_to_unicode: bool = False,
) -> str:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        dtype = _get_array_dtype(value, h5_to_unicode)
        out = f": array, shape={value.shape}, dtype={dtype}"
    elif isinstance(value, h5py.Dataset):
        shape = value.shape
        if first_dim is not None:
            if first_dim > shape[0]:
                raise ValueError(
                    f"first_dim ({first_dim}) is greater than the dataset's first dimension ({shape[0]})."
                )
            shape = tuple([int(first_dim), *shape[1:]])
        dtype = _get_array_dtype(value, h5_to_unicode)
        out = f": lazy array, shape={shape}, dtype={dtype}"
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
        k, v = list(value.items())[0]
        unicode_keys = getattr(value, "_unicode_keys", [])
        key_lines = _size_repr(k, v, 0, first_dim, h5_to_unicode=k in unicode_keys)
        out = "={" + key_lines + "}"
    elif isinstance(value, Mapping):
        unicode_keys = getattr(value, "_unicode_keys", [])
        lines = []
        for k, v in value.items():
            key_lines = _size_repr(
                k, v, indent + 2, first_dim, h5_to_unicode=k in unicode_keys
            )
            lines.append(key_lines)

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
