from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


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


def _validate_select_by_mask_input(mask: np.ndarray, length: int) -> None:
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


def _validate_object_shapes(
    shape_dict: Mapping[str, Sequence[int]], ndims: int | None = None
) -> None:

    try:
        first_dims = {shape[0] for shape in shape_dict.values()}

    except IndexError:
        names = [name for name, shape in shape_dict.items() if len(shape) == 0]
        raise ValueError(
            "Expected objects to have at least 1 dimension, but found "
            f"{len(names)} 0-dimensional objects: {names}."
        ) from None  # Suppress the context of the IndexError

    if len(first_dims) > 1:
        if len(shape_dict) == 2:
            first_obj, second_obj = shape_dict.items()

            raise ValueError(
                f"First dimensions of objects are inconsistent: {first_obj[1]} ({first_obj[0]}) "
                f"and {second_obj[1]} ({second_obj[0]})."
            )

        dims = [shape[0] for shape in shape_dict.values()]
        standard = max(dims, key=dims.count)
        mismatched = [
            f"{name} ({shape[0]})"
            for name, shape in shape_dict.items()
            if shape[0] != standard
        ]

        raise ValueError(
            f"First dimensions of objects are inconsistent. The most common is {standard}, "
            f"but these differ: ({', '.join(mismatched)})."
        )

    if ndims is not None:
        _validate_object_ndims(shape_dict, ndims=ndims)


def _validate_object_ndims(shape_dict: Mapping[str, Sequence[int]], ndims: int) -> None:

    if not set(map(len, shape_dict.values())) <= {ndims}:
        bad_ndims = [
            f"{name} ({len(shape)}D)"
            for name, shape in shape_dict.items()
            if len(shape) != ndims
        ]
        raise ValueError(
            f"Objects are expected to have {ndims} dimensions, but these objects do not: ({', '.join(bad_ndims)})."
        )
