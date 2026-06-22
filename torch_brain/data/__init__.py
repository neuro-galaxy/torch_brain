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


def __getattr__(name):
    # `serialize_fn_map` was renamed to the private `_DEFAULT_SERIALIZE_FN_MAP`.
    # Keep the old public name importable but deprecated; `Data.to_hdf5` now
    # applies this default map automatically, so passing it explicitly is no
    # longer necessary.
    if name == "serialize_fn_map":
        import warnings

        from .serialization import _DEFAULT_SERIALIZE_FN_MAP

        warnings.warn(
            "torch_brain.data.serialize_fn_map is deprecated and will be removed "
            "in a future version. Data.to_hdf5 now uses this default serialization "
            "map automatically, so it no longer needs to be passed explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEFAULT_SERIALIZE_FN_MAP
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
            "autosummary": ["concat"],
        },
    ],
}
