from collections import Counter, defaultdict
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


def _validate_object_shapes(*shape_list, **shape_dict):
    objects = list(shape_dict.items()) + [(None, obj) for obj in shape_list]

    zero_dim_names = [name for name, shape in objects if len(shape) == 0]

    if len(zero_dim_names) > 0:
        names = [name for name in zero_dim_names if name is not None]
        if len(names) > 0:
            if len(names) == len(zero_dim_names):
                name_str = f": {names}"
            else:
                name_str = f", including {names}"

        raise ValueError(
            "Expected objects to have at least 1 dimension, but found "
            f"{len(zero_dim_names)} 0-dimensional objects{name_str}."
        )

    counts = Counter(shape[0] for _, shape in objects)

    if len(counts) > 1:
        standard, standard_count = counts.most_common(1)[0]

        by_dim = defaultdict(list)
        for name, shape in objects:
            if shape[0] != standard:
                by_dim[shape[0]].append(name)

        mismatches = sorted(
            by_dim.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        details = "\n".join(
            f"{dim} ({len(names)}): {', '.join(names)}" for dim, names in mismatches
        )

        raise ValueError(
            f"First dimensions of objects are inconsistent. The most common is {standard} "
            f"({standard_count} objects), but found:\n{details}."
        )
