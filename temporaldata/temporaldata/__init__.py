from importlib.metadata import version, PackageNotFoundError
from .arraydict import ArrayDict, LazyArrayDict
from .irregular_ts import IrregularTimeSeries, LazyIrregularTimeSeries
from .regular_ts import RegularTimeSeries, LazyRegularTimeSeries
from .interval import Interval, LazyInterval
from .data import Data

from .concat import concat

try:
    __version__ = version("temporaldata")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing the package without installing
    __version__ = "unknown"  # pragma: no cover
