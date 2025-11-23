from importlib.metadata import version, PackageNotFoundError
from . import data
from . import nn
from . import models
from . import optim
from . import utils
from . import transforms
from . import registry

from .registry import register_modality, get_modality_by_id, MODALITY_REGISTRY

try:
    __version__ = version("pytorch-brain")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing torch_brain without installing
    pass  # pragma: no cover
