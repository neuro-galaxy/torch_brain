import datetime
from collections.abc import Callable


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


_DEFAULT_SERIALIZE_FN_MAP: dict[type, Callable] = {
    datetime.datetime: datetime_serialize_fn,
}


def get_default_serialize_fn_map() -> dict[type, Callable]:
    r"""Returns the default serialization map used when saving :class:`Data` to HDF5.

    :meth:`torch_brain.data.Data.to_hdf5` uses this map to serialize attribute
    values whose types HDF5 cannot store natively. By default it maps:

    - :obj:`datetime.datetime` to :obj:`str`

    A fresh copy is returned on every call, so mutating the result never
    affects the global default. You can extend the returned map to support
    additional types:

    .. code:: python

        serialize_fn_map = get_default_serialize_fn_map()
        serialize_fn_map[MyType] = my_serialize_fn
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


    Returns:
        A copy of the default mapping from type to serialization function.
    """
    return dict(_DEFAULT_SERIALIZE_FN_MAP)
