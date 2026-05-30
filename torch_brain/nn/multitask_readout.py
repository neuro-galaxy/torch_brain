from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from brainsets.taxonomy import DecoderSpec, Decoder, Task
from torch_brain.data.collate import collate, chain, track_batch
from torch_brain.nn import compute_loss_or_metric


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
