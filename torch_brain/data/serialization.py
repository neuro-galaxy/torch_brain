import datetime
from collections.abc import Callable


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


serialize_fn_map: dict[type, Callable] = {
    datetime.datetime: datetime_serialize_fn,
}
r"""A dict that maps classes to their serialization functions"""
