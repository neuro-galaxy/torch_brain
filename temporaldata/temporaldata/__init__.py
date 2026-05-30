from importlib.metadata import version, PackageNotFoundError
from .arraydict import ArrayDict, LazyArrayDict
from .irregular_ts import IrregularTimeSeries, LazyIrregularTimeSeries
from .regular_ts import RegularTimeSeries, LazyRegularTimeSeries
from .interval import Interval, LazyInterval
from .data import Data

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

try:
    __version__ = version("temporaldata")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing the package without installing
    __version__ = "unknown"  # pragma: no cover
