import datetime
from collections.abc import Callable


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


_DEFAULT_SERIALIZE_FN_MAP: dict[type, Callable] = {
    datetime.datetime: datetime_serialize_fn,
}
