from importlib.metadata import version, PackageNotFoundError
from . import data
from . import nn
from . import models
from . import utils
from . import transforms
from . import samplers

try:
    __version__ = version("torch_brain")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing the package without installing
    __version__ = "unknown"  # pragma: no cover
