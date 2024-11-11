from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from torch_brain.data.collate import collate, chain, track_batch
from torch_brain.registry import ModalitySpec, MODALITIY_REGISTRY

from typing import Dict, List


class MultitaskReadout(nn.Module):
    """A module that performs multi-task linear readouts from output embeddings."""

    def __init__(
        self,
        dim: int,
        readout_specs: Dict[str, ModalitySpec],
    ):
        super().__init__()

        self.readout_specs = readout_specs

        # Create a bunch of projection layers. One for each readout
        self.projections = nn.ModuleDict({})
        for readout_id, spec in self.readout_specs.items():
            self.projections[readout_id] = nn.Linear(dim, spec.dim)

    def forward(
        self,
        output_embs: TensorType["batch", "n_out", "dim"],
        output_readout_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ) -> List[Dict[str, TensorType["*nqueries", "*nchannelsout"]]]:
        """Forward pass of the multi-task readout module.

        Args:
            output_embs: Transformer output embeddings of shape (batch, n_out, dim)
            output_readout_index: Integer indices indicating which readout head to use for each
                output token. Shape (batch, n_out)
            unpack_output: By default False, which concatenates all outputs into a single dictionary
                organized by task. Set to True to break down outputs by individual samples in the batch.

        Returns:
            If unpack_output=False (default):
                Single dictionary containing outputs from all samples concatenated together,
                organized by task name. Shape per task: (total_queries, n_channels)
            If unpack_output=True:
                List of dictionaries, where each dictionary contains the outputs for a single batch
                sample organized by task name. Shape per task: (n_queries, n_channels)
        """
        if unpack_output:
            batch_size = output_embs.shape[0]
            outputs = [{} for _ in range(batch_size)]
        else:
            outputs = {}

        for readout_name, readout_spec in self.readout_specs.items():
            # get the mask of tokens that belong to this task
            mask = output_readout_index == readout_spec.id

            if not torch.any(mask):
                # there is not a single token in the batch for this task, so we skip
                continue

            # apply the appropriate projection for all tokens in the batch that belong to this task
            task_output = self.projections[readout_name](output_embs[mask])

            if unpack_output:
                # we need to distribute the outputs to their respective samples
                batch_index_filtered_by_decoder = torch.where(mask)[0]

                targeted_batch_elements, batch_index_filtered_by_decoder = torch.unique(
                    batch_index_filtered_by_decoder, return_inverse=True
                )
                for i in range(len(targeted_batch_elements)):
                    outputs[targeted_batch_elements[i]][readout_name] = task_output[
                        batch_index_filtered_by_decoder == i
                    ]
            else:
                outputs[readout_name] = task_output

        return outputs

    def forward_varlen(
        self,
        output_embs: TensorType["total_ntokens", "dim"],
        output_readout_index: TensorType["total_ntokens", int],
        output_batch_index: TensorType["total_ntout"],
        unpack_output: bool = False,
    ) -> List[Dict[str, TensorType["*nqueries", "*nchannelsout"]]]:
        """Forward pass of the multi-task readout module for variable length sequences.

        This version handles sequences that are chained together in a single batch dimension
        rather than padded. This can be more memory efficient since it avoids padding.

        Args:
            output_embs: Transformer output embeddings of shape (total_ntokens, dim)
                where total_ntokens is the sum of sequence lengths across the batch
            output_readout_index: Integer indices indicating which readout head to use for each
                output token. Shape (total_ntokens,)
            output_batch_index: Tensor containing batch indices for each token.
                Shape (total_ntokens,)
            unpack_output: By default False, which concatenates all outputs into a single dictionary
                organized by task. Set to True to break down outputs by individual samples using
                the batch indices.

        Returns:
            If unpack_output=False (default):
                Single dictionary containing outputs from all samples concatenated together,
                organized by task name. Shape per task: (total_queries, n_channels)
            If unpack_output=True:
                List of dictionaries, where each dictionary contains the outputs for a single batch
                sample organized by task name. Shape per task: (n_queries, n_channels)
        """
        if unpack_output:
            batch_size = output_batch_index.max().item() + 1
            outputs = [{} for _ in range(batch_size)]
        else:
            outputs = {}

        for readout_name, readout_spec in self.readout_specs.items():
            # get the mask of tokens that belong to this task
            mask = output_readout_index == readout_spec.id

            if not torch.any(mask):
                # there is not a single token in the batch for this task, so we skip
                continue

            # apply the projection
            task_output = self.projections[readout_name](output_embs[mask])

            if unpack_output:
                # Inputs where chained, and we have batch-indices for each token
                batch_index_filtered_by_decoder = output_batch_index[mask]

                targeted_batch_elements, batch_index_filtered_by_decoder = torch.unique(
                    batch_index_filtered_by_decoder, return_inverse=True
                )
                for i in range(len(targeted_batch_elements)):
                    outputs[targeted_batch_elements[i]][readout_name] = task_output[
                        batch_index_filtered_by_decoder == i
                    ]
            else:
                outputs[readout_name] = task_output

        return outputs


def prepare_for_multitask_readout(
    data,
    readout_registry: Dict[str, ModalitySpec],
):
    required_keys = ["readout_id"]
    optional_keys = [
        "weight",
        "subtask_weights",
        "normalize_mean",
        "normalize_std",
        "timestamp_key",
        "value_key",
        "context_key",
        "metrics",
    ]

    readout_index = list()
    timestamps = list()
    values = dict()
    context_index = dict()
    weights = dict()

    for readout_config in data.config["multitask_readout"]:
        # check that the readout config contains all required keys
        for key in required_keys:
            if key not in readout_config:
                raise ValueError(
                    f"multitask_readout config is missing required key: {key}"
                )

        # check that the readout config contains only valid keys
        if not all(
            key in required_keys + optional_keys for key in readout_config.keys()
        ):
            raise ValueError(
                f"Readout {readout_config} contains invalid keys, please use only {required_keys + optional_keys}"
            )

        key = readout_config["readout_id"]

        if key not in MODALITIY_REGISTRY:
            raise ValueError(
                f"Readout {key} not found in modality registry, please register it "
                "using torch_brain.register_modality()"
            )

        readout_spec = readout_registry[key]
        value_key = readout_config.get("value_key", readout_spec.value_key)
        timestamp_key = readout_config.get("timestamp_key", readout_spec.timestamp_key)
        context_key = readout_config.get("context_key", readout_spec.context_key)
        weight = readout_config.get("weight", 1.0)
        subtask_weights = readout_config.get("subtask_weights", {})

        readout_index.append(readout_spec.id)
        timestamps.append(data.get_nested_attribute(timestamp_key))
        values[key] = data.get_nested_attribute(value_key)

        # z-scale the values if mean/std are specified in the config file
        if "normalize_mean" in readout_config:
            # if mean is a list, its a per-channel mean (usually for x,y coordinates)
            if isinstance(readout_config["normalize_mean"], list):
                mean = np.array(readout_config["normalize_mean"])
            else:
                mean = readout_config["normalize_mean"]
            values[key] = values[key] - mean
        if "normalize_std" in readout_config:
            # if std is a list, its a per-channel std (usually for x,y coordinates)
            if isinstance(readout_config["normalize_std"], list):
                std = np.array(readout_config["normalize_std"])
            else:
                std = readout_config["normalize_std"]
            values[key] = values[key] / std

        # here we assume that we won't be running a model at float64 precision
        if values[key].dtype == np.float64:
            values[key] = values[key].astype(np.float32)

        if False and context_key is not None:
            context_index[key] = data.get_nested_attribute(context_key)
            num_subtasks = Task.from_string(list(subtask_weights.keys())[0]).max_value()
            subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
            for subtask, subtask_weight in subtask_weights.items():
                subtask_weight_map[Task.from_string(subtask).value] = subtask_weight

            subtask_weight_map *= weight
            weights[key] = subtask_weight_map[context_index[key]]
        else:
            context_index[key] = np.zeros(len(values[key]), dtype=np.int64)
            weights[key] = np.ones(len(values[key]), dtype=np.float32) * weight

    # chain
    timestamps, batch = collate(
        [
            (chain(timestamps[i]), track_batch(timestamps[i]))
            for i in range(len(timestamps))
        ]
    )
    readout_index = torch.tensor(readout_index)[batch]

    return timestamps, readout_index, values, weights, context_index
