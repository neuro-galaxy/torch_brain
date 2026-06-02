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
