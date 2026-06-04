"""
**User guide.** See :ref:`data_guide` section for further details.
"""

from .arraydict import ArrayDict, LazyArrayDict
from .irregular_ts import IrregularTimeSeries, LazyIrregularTimeSeries
from .regular_ts import RegularTimeSeries, LazyRegularTimeSeries
from .interval import Interval, LazyInterval
from .data import Data
from .descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from .serialization import serialize_fn_map

from .concat import concat

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
            "autosummary": __all__,
        },
    ],
}
