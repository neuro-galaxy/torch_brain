from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from brainsets.taxonomy import StringIntEnum, Task
from torch_brain.data.collate import collate, chain, track_batch
from torch_brain.nn import compute_loss_or_metric, OutputType

from typing import Dict, List, Tuple, Optional, Union, Any

from pydantic.dataclasses import dataclass


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
        value_key="natural_scenes.frame",
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


class MultitaskReadout(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        decoder_specs: Dict[str, DecoderSpec],
        batch_type="stacked",
    ):
        super().__init__()

        # Create a bunch of projection layers. One for each task
        self.projections = nn.ModuleDict({})
        for decoder_id, spec in decoder_specs.items():
            self.projections[decoder_id] = nn.Linear(latent_dim, spec.dim)

        # Need task specs layer to decide loss type
        self.decoder_specs = decoder_specs
        self.batch_type = batch_type

    def forward(
        self,
        output_latents: Union[
            TensorType["batch", "max_ntout", "dim"], TensorType["total_ntout", "dim"]
        ],
        output_decoder_index: Union[
            TensorType["batch", "max_ntout"], TensorType["total_ntout"]
        ],
        output_batch_index: Optional[TensorType["total_ntout"]] = None,
        output_values: Dict[str, TensorType["*ntout_task", "*nchannelsout"]] = None,
        output_weights: Dict[str, TensorType["*ntout_task"]] = None,
        unpack_output: bool = False,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        Union[None, torch.Tensor],
        Union[None, Dict[str, torch.Tensor]],
    ]:
        """
        Args:
            output_latents: Outputs of the last transformer layer.
            output_task_indices: Task index for each token in (batch, max_ntout).
            output_values: Ground-truth values for loss computation.
                output_values[task] is the ground truth value for the task
            output_weights: Sample-wise weights for loss computation.
                output_weights[task] is the weight for a given task.
        """

        if output_batch_index is not None:
            # Inputs were chained, make sure input dimensions make sense
            assert output_latents.dim() == 2
            assert output_decoder_index.dim() == 1
            assert output_batch_index.dim() == 1
            batch_size = output_batch_index.max().item() + 1
        else:
            # Inputs were not chained, make sure input dimensions make sense
            assert output_latents.dim() == 3
            assert output_decoder_index.dim() == 2
            batch_size = output_latents.shape[0]

        outputs = [{} for _ in range(batch_size)]
        taskwise_loss = {}
        loss = torch.tensor(0, device=output_latents.device, dtype=torch.float32)

        for decoder_id, spec in self.decoder_specs.items():
            # the taskid is a universal unique identifier for the task
            decoder_index = Decoder.from_string(decoder_id).value

            # get the mask of tokens that belong to this task
            mask = output_decoder_index == decoder_index

            if not torch.any(mask):
                # there is not a single token in the batch for this task, so we skip
                continue

            # apply the projection
            task_output = self.projections[decoder_id](output_latents[mask])

            # we need to distribute the outputs to their respective samples
            if self.batch_type == "stacked":
                token_batch = torch.where(mask)[0]
            elif self.batch_type == "chained":
                token_batch = output_batch_index[mask]
            else:
                raise ValueError(f"Unknown batch_type: {self.batch_type}")

            unique_batch_indices = torch.unique(token_batch)
            for batch_idx in unique_batch_indices:
                outputs[batch_idx][decoder_id] = task_output[token_batch == batch_idx]

            # compute loss
            if output_values is not None:
                target = output_values[decoder_id]

                weights = 1.0
                if (
                    decoder_id in output_weights
                    and output_weights[decoder_id] is not None
                ):
                    weights = output_weights[decoder_id]

                taskwise_loss[decoder_id] = compute_loss_or_metric(
                    spec.loss_fn, spec.type, task_output, target, weights
                )

            # we need to distribute the outputs to their respective samples
            if output_batch_index is None:
                batch_index_filtered_by_decoder = torch.where(mask)[0]
            else:
                # Inputs where chained, and we have batch-indices for each token
                batch_index_filtered_by_decoder = output_batch_index[mask]

            targeted_batch_elements, batch_index_filtered_by_decoder = torch.unique(
                batch_index_filtered_by_decoder, return_inverse=True
            )
            for i in range(len(targeted_batch_elements)):
                outputs[targeted_batch_elements[i]][decoder_id] = task_output[
                    batch_index_filtered_by_decoder == i
                ]

            if output_values is not None:
                # Since we calculate a mean across all elements, scale by the number of
                # items in the batch so we don't get wild swings in loss depending on
                # whether we have large or small numbers of non-dominant classes.
                loss = loss + taskwise_loss[decoder_id] * len(targeted_batch_elements)

        loss = loss / batch_size

        if output_values is None:
            return outputs, None, None

        return outputs, loss, taskwise_loss


def prepare_for_multitask_readout(
    data,
    decoder_registry: Dict[str, DecoderSpec],
):
    decoder_index = list()
    timestamps = list()
    values = dict()
    # task_index = dict()
    subtask_index = dict()
    weights = dict()

    config = data.config["multitask_readout"]

    for decoder in config:
        key = decoder["decoder_id"]
        weight = decoder.get("weight", 1.0)
        subtask_weights = decoder.get("subtask_weights", {})

        decoder = decoder_registry[key].__dict__ | decoder  # config overrides registry

        decoder_index.append(Decoder.from_string(key).value)
        values[key] = data.get_nested_attribute(decoder["value_key"])

        # z-scale the values if mean/std are specified in the config file
        if "normalize_mean" in decoder:
            # if mean is a list, its a per-channel mean (usually for x,y coordinates)
            if isinstance(decoder["normalize_mean"], list):
                mean = np.array(decoder["normalize_mean"])
            else:
                mean = decoder["normalize_mean"]
            values[key] = values[key] - mean
        if "normalize_std" in decoder:
            # if std is a list, its a per-channel std (usually for x,y coordinates)
            if isinstance(decoder["normalize_std"], list):
                std = np.array(decoder["normalize_std"])
            else:
                std = decoder["normalize_std"]
            values[key] = values[key] / std

        timestamps.append(data.get_nested_attribute(decoder["timestamp_key"]))

        # here we assume that we won't be running a model at float64 precision
        # TODO do this in decoder spec?
        if values[key].dtype == np.float64:
            values[key] = values[key].astype(np.float32)

        # if decoder["task_index"] is not None:
        #     task_index[key] = data.get_nested_attribute(decoder["task_index"])
        # else:
        #     task_index[key] = np.zeros(len(values[key]), dtype=np.int64)

        if decoder["subtask_key"] is not None:
            subtask_index[key] = data.get_nested_attribute(decoder["subtask_key"])
            num_subtasks = Task.from_string(list(subtask_weights.keys())[0]).max_value()
            subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
            for subtask, subtask_weight in subtask_weights.items():
                subtask_weight_map[Task.from_string(subtask).value] = subtask_weight

            subtask_weight_map *= weight
            weights[key] = subtask_weight_map[subtask_index[key]]
        else:
            subtask_index[key] = np.zeros(len(values[key]), dtype=np.int64)
            weights[key] = np.ones(len(values[key]), dtype=np.float32) * weight

    # chain
    timestamps, batch = collate(
        [
            (chain(timestamps[i]), track_batch(timestamps[i]))
            for i in range(len(timestamps))
        ]
    )
    decoder_index = torch.tensor(decoder_index)[batch]

    return timestamps, decoder_index, values, weights, subtask_index
