__all__ = [
    "datetime_serialize_fn",
    "serialize_fn_map",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Functions",
            "autosummary": [
                "datetime_serialize_fn",
            ],
        },
        {
            "title": "Constants",
            "autosummary": [
                "serialize_fn_map",
            ],
        },
    ],
}

import datetime
from typing import Callable


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


serialize_fn_map: dict[type, Callable] = {
    datetime.datetime: datetime_serialize_fn,
}
r"""A dict that maps classes to their serialization functions"""
