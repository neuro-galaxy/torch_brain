from typing import Dict, List, Tuple, Optional, Union, Any

from pydantic.dataclasses import dataclass

from .core import StringIntEnum


class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3


class Decoder(StringIntEnum):
    NA = 0
    # Classic BCI outputs.
    ARMVELOCITY2D = 1
    CURSORPOSITION2D = 2
    EYE2D = 3
    FINGER3D = 4

    # Shenoy handwriting style outputs.
    WRITING_CHARACTER = 5
    WRITING_LINE = 6

    DISCRETE_TRIAL_ONSET_OFFSET = 7
    CONTINUOUS_TRIAL_ONSET_OFFSET = 8

    CURSORVELOCITY2D = 9

    # Allen data
    DRIFTING_GRATINGS_ORIENTATION = 13
    DRIFTING_GRATINGS_TEMPORAL_FREQUENCY = 23
    STATIC_GRATINGS_ORIENTATION = 17
    STATIC_GRATINGS_SPATIAL_FREQUENCY = 18
    STATIC_GRATINGS_PHASE = 19

    RUNNING_SPEED = 24
    PUPIL_SIZE_2D = 25
    GAZE_POS_2D = 26
    GABOR_ORIENTATION = 21  #
    GABOR_POS_2D = 27
    NATURAL_SCENES = 28
    NATURAL_MOVIE_ONE_FRAME = 30
    NATURAL_MOVIE_TWO_FRAME = 31
    NATURAL_MOVIE_THREE_FRAME = 32
    LOCALLY_SPARSE_NOISE_FRAME = 33

    # Openscope calcium
    UNEXPECTED_OR_NOT = 20  #
    PUPIL_MOVEMENT_REGRESSION = 22
    PUPIL_LOCATION = 34

    # speech
    SPEAKING_CVSYLLABLE = 14
    SPEAKING_CONSONANT = 15
    SPEAKING_VOWEL = 16


@dataclass
class DecoderSpec:
    dim: int
    type: OutputType
    loss_fn: str
    timestamp_key: str
    value_key: str
    # Optional fields
    task_key: Optional[str] = None
    subtask_key: Optional[str] = None
    # target_dtype: str = "float32"  # torch.dtype is not serializable.


decoder_registry = {
    str(Decoder.ARMVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="behavior.timestamps",
        value_key="behavior.hand_vel",
        subtask_key="behavior.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSORVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.vel",
        subtask_key="cursor.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSORPOSITION2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.pos",
        subtask_key="cursor.subtask_index",
        loss_fn="mse",
    ),
    # str(Decoder.WRITING_CHARACTER): DecoderSpec(
    #     dim=len(Character),
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="stimuli_segments.timestamps",
    #     value_key="stimuli_segments.letters",
    #     loss_fn="bce",
    # ),
    # str(Decoder.WRITING_LINE): DecoderSpec(
    #     dim=len(Line),
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="stimuli_segments.timestamps",
    #     value_key="stimuli_segments.letters",
    #     loss_fn="bce",
    # ),
    str(Decoder.DRIFTING_GRATINGS_ORIENTATION): DecoderSpec(
        dim=8,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="drifting_gratings.timestamps",
        value_key="drifting_gratings.orientation_id",
        loss_fn="bce",
    ),
    str(Decoder.DRIFTING_GRATINGS_TEMPORAL_FREQUENCY): DecoderSpec(
        dim=5,  # [1,2,4,8,15]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="drifting_gratings.timestamps",
        value_key="drifting_gratings.temporal_frequency_id",
        loss_fn="bce",
    ),
    str(Decoder.NATURAL_MOVIE_ONE_FRAME): DecoderSpec(
        dim=900,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="natural_movie_one.timestamps",
        value_key="natural_movie_one.frame",
        loss_fn="bce",
    ),
    str(Decoder.NATURAL_MOVIE_TWO_FRAME): DecoderSpec(
        dim=900,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="natural_movie_two.timestamps",
        value_key="natural_movie_two.frame",
        loss_fn="bce",
    ),
    str(Decoder.NATURAL_MOVIE_THREE_FRAME): DecoderSpec(
        dim=3600,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="natural_movie_three.timestamps",
        value_key="natural_movie_three.frame",
        loss_fn="bce",
    ),
    str(Decoder.LOCALLY_SPARSE_NOISE_FRAME): DecoderSpec(
        dim=8000,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="locally_sparse_noise.timestamps",
        value_key="locally_sparse_noise.frame",
        loss_fn="bce",
    ),
    str(Decoder.STATIC_GRATINGS_ORIENTATION): DecoderSpec(
        dim=6,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="static_gratings.timestamps",
        value_key="static_gratings.orientation_id",
        loss_fn="bce",
    ),
    str(Decoder.STATIC_GRATINGS_SPATIAL_FREQUENCY): DecoderSpec(
        dim=5,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="static_gratings.timestamps",
        value_key="static_gratings.spatial_frequency_id",
        loss_fn="bce",
    ),
    str(Decoder.STATIC_GRATINGS_PHASE): DecoderSpec(
        dim=5,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="static_gratings.timestamps",
        value_key="static_gratings.phase_id",
        loss_fn="bce",
    ),
    # str(Decoder.SPEAKING_CVSYLLABLE): DecoderSpec(
    #     dim=len(CVSyllable),  # empty label is included
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="speech.timestamps",
    #     value_key="speech.consonant_vowel_syllables",
    #     loss_fn="bce",
    # ),
    str(Decoder.NATURAL_SCENES): DecoderSpec(
        dim=119,  # image classes [0,...,118]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="natural_scenes.timestamps",
        value_key="natural_scenes.image_ids",
        loss_fn="bce",
    ),
    str(Decoder.GABOR_ORIENTATION): DecoderSpec(
        dim=4,  # [0, 1, 2, 3]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="gabors.timestamps",
        value_key="gabors.gabors_orientation",
        loss_fn="bce",
    ),
    str(Decoder.GABOR_POS_2D): DecoderSpec(  # 9x9 grid modeled as (x, y) coordinates
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="gabors.timestamps",
        value_key="gabors.pos_2d",
        loss_fn="mse",
    ),
    str(Decoder.RUNNING_SPEED): DecoderSpec(
        dim=1,
        target_dim=1,
        type=OutputType.CONTINUOUS,
        timestamp_key="running.timestamps",
        value_key="running.running_speed",
        loss_fn="mse",
    ),
    str(Decoder.GAZE_POS_2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="gaze.timestamps",
        value_key="gaze.pos_2d",
        loss_fn="mse",
    ),
    str(Decoder.PUPIL_LOCATION): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="pupil.timestamps",
        value_key="pupil.location",
        loss_fn="mse",
    ),
    str(Decoder.PUPIL_SIZE_2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="pupil.timestamps",
        value_key="pupil.size_2d",
        loss_fn="mse",
    ),
}
