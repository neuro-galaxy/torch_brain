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
from .serialization import serialize_fn_map

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
    "serialize_fn_map",
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
        {
            "title": "Constants",
            "autosummary": ["serialize_fn_map"],
        },
    ],
}
