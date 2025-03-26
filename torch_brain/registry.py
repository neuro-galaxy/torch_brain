from enum import Enum
from typing import Dict, Tuple, Optional, Any, Callable

from pydantic.dataclasses import dataclass
import torch_brain


class DataType(Enum):
    """Enum defining the possible data types.

    Attributes:
        CONTINUOUS: For continuous-valued variables
        BINARY: For binary variables
        MULTINOMIAL: For multi-class variables
        MULTILABEL: For multi-label variables
    """

    CONTINUOUS = 0
    BINARY = 1
    MULTINOMIAL = 2
    MULTILABEL = 3


@dataclass
class ModalitySpec:
    """Specification for a modality.

    Attributes:
        dim: Dimension for this modality
        type: DataType enum specifying the data type
        loss_fn: Name of loss function to use for this modality
        timestamp_key: Key to access timestamps in the data object
        value_key: Key to access values in the data object
        id: Unique numeric ID assigned to this modality
    """

    id: int
    dim: int
    type: DataType
    timestamp_key: str  # can be overwritten
    value_key: str  # can be overwritten
    loss_fn: Callable  # can be overwritten


MODALITY_REGISTRY: Dict[str, ModalitySpec] = {}
_ID_TO_MODALITY: Dict[int, str] = {}


def register_modality(name: str, **kwargs: Any) -> int:
    """Register a new modality specification in the global registry.

    Args:
        name: Unique identifier for this modality
        **kwargs: Keyword arguments used to construct the ModalitySpec
            Must include: dim, type, loss_fn, timestamp_key, value_key

    Returns:
        int: Unique numeric ID assigned to this modality

    Raises:
        ValueError: If a modality with the given name already exists
    """
    # Check if modality already exists
    if name in MODALITY_REGISTRY:
        raise ValueError(f"Modality {name} already exists in registry")

    # Get next available ID
    next_id = len(MODALITY_REGISTRY) + 1

    # Create DecoderSpec from kwargs and set ID
    decoder_spec = ModalitySpec(**kwargs, id=next_id)

    # Add to registries
    MODALITY_REGISTRY[name] = decoder_spec
    _ID_TO_MODALITY[next_id] = name

    return next_id


def get_modality_by_id(modality_id: int) -> ModalitySpec:
    """Get a modality specification by its ID.

    Args:
        modality_id: The numeric ID of the modality to retrieve

    Returns:
        ModalitySpec: The modality specification

    Raises:
        KeyError: If no modality exists with the given ID
    """
    if modality_id not in _ID_TO_MODALITY:
        raise KeyError(f"No modality found with ID {modality_id}")
    return _ID_TO_MODALITY[modality_id]


register_modality(
    "cursor_velocity_2d",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="cursor.timestamps",
    value_key="cursor.vel",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "cursor_position_2d",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="cursor.timestamps",
    value_key="cursor.pos",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "arm_velocity_2d",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="behavior.timestamps",
    value_key="behavior.hand_vel",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "drifting_gratings_orientation",
    dim=8,
    type=DataType.MULTINOMIAL,
    timestamp_key="drifting_gratings.timestamps",
    value_key="drifting_gratings.orientation_id",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "drifting_gratings_temporal_frequency",
    dim=5,  # [1,2,4,8,15]
    type=DataType.MULTINOMIAL,
    timestamp_key="drifting_gratings.timestamps",
    value_key="drifting_gratings.temporal_frequency_id",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "natural_movie_one_frame",
    dim=900,
    type=DataType.MULTINOMIAL,
    timestamp_key="natural_movie_one.timestamps",
    value_key="natural_movie_one.frame",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "natural_movie_two_frame",
    dim=900,
    type=DataType.MULTINOMIAL,
    timestamp_key="natural_movie_two.timestamps",
    value_key="natural_movie_two.frame",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "natural_movie_three_frame",
    dim=3600,
    type=DataType.MULTINOMIAL,
    timestamp_key="natural_movie_three.timestamps",
    value_key="natural_movie_three.frame",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "locally_sparse_noise_frame",
    dim=8000,
    type=DataType.MULTINOMIAL,
    timestamp_key="locally_sparse_noise.timestamps",
    value_key="locally_sparse_noise.frame",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "static_gratings_orientation",
    dim=6,
    type=DataType.MULTINOMIAL,
    timestamp_key="static_gratings.timestamps",
    value_key="static_gratings.orientation_id",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "static_gratings_spatial_frequency",
    dim=5,
    type=DataType.MULTINOMIAL,
    timestamp_key="static_gratings.timestamps",
    value_key="static_gratings.spatial_frequency_id",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "static_gratings_phase",
    dim=5,
    type=DataType.MULTINOMIAL,
    timestamp_key="static_gratings.timestamps",
    value_key="static_gratings.phase_id",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "natural_scenes",
    dim=119,  # image classes [0,...,118]
    type=DataType.MULTINOMIAL,
    timestamp_key="natural_scenes.timestamps",
    value_key="natural_scenes.frame",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "gabor_orientation",
    dim=4,  # [0, 1, 2, 3]
    type=DataType.MULTINOMIAL,
    timestamp_key="gabors.timestamps",
    value_key="gabors.gabors_orientation",
    loss_fn=torch_brain.nn.loss.CrossEntropyLoss(),
)

register_modality(
    "gabor_pos_2d",  # 9x9 grid modeled as (x, y) coordinates
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="gabors.timestamps",
    value_key="gabors.pos_2d",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "running_speed",
    dim=1,
    type=DataType.CONTINUOUS,
    timestamp_key="running.timestamps",
    value_key="running.running_speed",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "gaze_pos_2d",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="gaze.timestamps",
    value_key="gaze.pos_2d",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "pupil_location",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="pupil.timestamps",
    value_key="pupil.location",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

register_modality(
    "pupil_size_2d",
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="pupil.timestamps",
    value_key="pupil.size_2d",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)
