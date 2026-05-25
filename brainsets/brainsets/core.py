_functions = ["datetime_serialize_fn"]
_constants = ["serialize_fn_map"]

__all__ = _functions + _constants

import datetime


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


serialize_fn_map = {
    datetime.datetime: datetime_serialize_fn,
}
r"""A dict that maps classes to their serialization functions"""
