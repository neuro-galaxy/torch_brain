"""Core data structures for representing neural and behavioral data.

See :ref:`data_guide` for further details.
"""

from .arraydict import ArrayDict, LazyArrayDict
from .concat import concat
from .data import Data
from .descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from .interval import Interval, LazyInterval
from .irregular_ts import IrregularTimeSeries, LazyIrregularTimeSeries
from .regular_ts import LazyRegularTimeSeries, RegularTimeSeries
from .serialization import get_default_serialize_fn_map


def __getattr__(name):
    # `serialize_fn_map` is deprecated: `Data.to_hdf5` now applies the default
    # serialization map automatically, so passing it explicitly is no longer
    # necessary. To extend the defaults, use `get_default_serialize_fn_map()`,
    # which returns a fresh, safe-to-mutate copy.
    if name == "serialize_fn_map":
        import warnings

        warnings.warn(
            "torch_brain.data.serialize_fn_map is deprecated and will be removed "
            "in a future version. Data.to_hdf5 now uses the default serialization "
            "map automatically; to extend it, use get_default_serialize_fn_map().",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_default_serialize_fn_map()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArrayDict",
    "LazyArrayDict",
    "IrregularTimeSeries",
    "LazyIrregularTimeSeries",
    "RegularTimeSeries",
    "LazyRegularTimeSeries",
    "Interval",
    "LazyInterval",
    "Data",
    "BrainsetDescription",
    "DeviceDescription",
    "SessionDescription",
    "SubjectDescription",
    "concat",
    "get_default_serialize_fn_map",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Core Objects",
            "template": "data_objects.rst",
            "autosummary": [
                "ArrayDict",
                "Data",
                "Interval",
                "IrregularTimeSeries",
                "RegularTimeSeries",
            ],
        },
        {
            "title": "Lazy Variants",
            "template": "data_objects.rst",
            "autosummary": [
                "LazyArrayDict",
                "LazyInterval",
                "LazyIrregularTimeSeries",
                "LazyRegularTimeSeries",
            ],
        },
        {
            "title": "Description Containers",
            "autosummary": [
                "BrainsetDescription",
                "DeviceDescription",
                "SessionDescription",
                "SubjectDescription",
            ],
        },
        {
            "title": "Utility Functions",
            "autosummary": [
                "concat",
                "get_default_serialize_fn_map",
            ],
        },
    ],
}
