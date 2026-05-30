from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np


@runtime_checkable
class _SupportsArray(Protocol):
    r"""
    See spec: https://numpy.org/devdocs/user/basics.interoperability.html
    TLDR: If an arbitrary object is passed into numpy.array(), then
    numpy attempts to use the object's `__array__` for the conversion
    """

    def __array__(self, dtype=None, copy=None) -> np.ndarray: ...


ArrayLike: TypeAlias = np.ndarray | list | tuple | _SupportsArray
