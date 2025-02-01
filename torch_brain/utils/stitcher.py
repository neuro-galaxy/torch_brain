import logging
from collections import defaultdict
from typing import Callable, Iterable, Optional

import hydra
import numpy as np
import pandas as pd
from rich import print as rprint
import torch
import lightning as L
import torchmetrics
import wandb

import torch_brain
from torch_brain.registry import ModalitySpec, DataType


def stitch(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    r"""This function performs pooling operations (mean or mode) on a tensor based on
    unique timestamps and the datatype of the values.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps.
        values (torch.Tensor): A tensor of values that correspond to the timestamps. It
            expects a tensor of shape (N, ...), where N is the number of timestamps.

    Returns:
        torch.Tensor: A tensor with the pooled values for each unique timestamp. If the
          values are continuous, the function performs mean pooling, averaging the
          values for each unique timestamp. If the values are categorical (labels),
          the function returns the mode of the values for each unique timestamp.

    Note:
        For mean pooling, this function leverages `torch.scatter_add_` to efficiently
        aggregate values for each unique timestamp
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    # Prepare a tensor for summing values for each unique timestamp
    pooled_sum = torch.zeros(
        (len(unique_timestamps), *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )

    # Use mode for integers
    if values.dtype == torch.long:
        # NOT IDEAL, IT IS FASTER TO AVERAGE THE LOGITS THAN TO PERFORM A VOTE
        mode_values = torch.zeros_like(pooled_sum)
        for i, timestamp in enumerate(unique_timestamps):
            mask = timestamps == timestamp
            group_values = values[mask]
            mode, _ = torch.mode(group_values, dim=0)
            mode_values[i] = mode
        return mode_values

    # Count occurrences of each unique timestamp
    counts = torch.zeros(
        len(unique_timestamps), device=timestamps.device, dtype=values.dtype
    )
    counts = counts.scatter_add_(
        0, indices, torch.ones_like(indices, dtype=values.dtype)
    )
    # Accumulate values for each unique timestamp
    indices_expanded = indices.unsqueeze(-1).expand_as(values)
    pooled_sum.scatter_add_(0, indices_expanded, values)
    # Calculate the average
    epsilon = 1e-8  # small constant to prevent division by zero
    averages = torch.div(pooled_sum, counts.unsqueeze(-1) + epsilon)

    return averages


class DecodingStitchEvaluator(L.Callback):
    r"""A convenient stitching and evaluation framework to use when:
     1. Your model outputs have associated timestamps
     2. And your sampling strategy involves overlapping time windows, requiring
        stitching to coalesce the predictions and targets before computing the
        evaluation metric.
     3. (Optional) You are training on multiple sessions/recordings and want to
        compute metrics for each session individually.

    This callback handles the stitching of the predictions and targets for each
    session and computes the metric for each session. The average metric value
    across all sessions is also computed and logged.
    Since the stitching is done only on tensors on the same GPU, sequences that are
    split across multiple GPUs will not be stitched together. In this case, it is
    recommended to use a stitching-aware sampler like
    `DistributedStitchingFixedWindowSampler` which ensures that the sequences are
    split across GPUs in a way that allows for correct stitching.

    This callback is called _after_ the validation_step, and has two main inputs:
    - The output of the validation_step function, which is expected to be the
      model predictions. These should be a :class:`~torch.Tensor` of shape (B, N, ...),
      where B is the batch size, N is the number of timestamps.
    - The batch dictionary that should have atleast the following keys:
        - "target_values": :class:`~torch.Tensor` of shape (B, N, ...)
        - "output_timestamps": :class:`~torch.Tensor` of shape (B, N) and dtype float
        - "output_mask": :class:`~torch.Tensor` of shape (B, N) and dtype bool
        - "session_id": A list of session IDs for each sample in the batch
        - "absolute_start": :class:`~torch.Tensor` of shape (B) and dtype float
    Please refer to the examples/poyo/train.py script for an example of how to write
    a validation_step(...) function that outputs the required tensors.

    This callback operates by maintaining a cache of the predictions, targets, and
    timestamps for each session. This cache is updated at the end of each batch.
    Finally, at the end of the epoch, the cache is coalesced and the metric is
    computed for each session.
    """

    def __init__(
        self,
        session_ids: Iterable[str],
        modality_spec: Optional[ModalitySpec] = None,
        quiet=False,
    ):
        r"""
        Args:
            session_ids: An iterable of session IDs for which the metrics are to be computed.
            modality_spec: (Optional) The modality specification for the task. Either this
                or metric_factory must be provided.
            metric_factory: (Optional) A callable that returns an instance of the metric to be used.
                If not provided, the metric is inferred based on the modality_spec.
            quiet: If True, disables the logging of the metrics to the console.
        """
        self.quiet = quiet

        if metric_factory is not None:
            pass
        elif modality_spec.type == DataType.CONTINUOUS:
            metric_factory = lambda: torchmetrics.R2Score()
        elif modality_spec.type in [DataType.BINARY, DataType.MULTINOMIAL]:
            metric_factory = lambda: torchmetrics.Accuracy(
                task="multiclass", num_classes=modality_spec.dim
            )
        else:
            raise ValueError(f"Unsupported datatype: {modality_spec.type}")

        self.metrics = {k: metric_factory() for k in session_ids}

    def on_validation_epoch_start(self, trainer, pl_module):
        # Cache to store the predictions, targets, and timestamps for each
        # validation step. This will be coalesced at the end of the validation,
        # using the stitch function.
        self.cache = defaultdict(
            lambda: {
                "pred": [],
                "target": [],
                "timestamps": [],
            }
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update the cache with the predictions, targets, timestamps, and subtask index
        # TODO: Mask timestamps based on subtask_index
        batch_size = len(outputs)
        for i in range(batch_size):
            mask = batch["output_mask"][i]
            session_id = batch["session_id"][i]
            absolute_start = batch["absolute_start"][i]

            pred = outputs[i][mask]
            target = batch["target_values"][i][mask]
            timestamps = batch["output_timestamps"][i][mask] + absolute_start

            self.cache[session_id]["pred"].append(pred.detach())
            self.cache[session_id]["target"].append(target.detach())
            self.cache[session_id]["timestamps"].append(timestamps.detach())

    def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
        # compute metric for each session
        metrics = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self.cache[session_id]
            pred = torch.cat(cache["pred"])
            target = torch.cat(cache["target"])
            timestamps = torch.cat(cache["timestamps"])

            stitched_pred = stitch(timestamps, pred)
            stitched_target = stitch(timestamps, target)

            metric_fn.to(pl_module.device).update(stitched_pred, stitched_target)
            metrics[session_id] = metric_fn.compute()
            metric_fn.reset()

        # compute the average metric
        metrics[f"average_{prefix}_metric"] = torch.tensor(
            list(metrics.values())
        ).mean()

        # log the metrics
        self.log_dict(metrics)

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value.item()})

        metrics_df = pd.DataFrame(metrics_data)

        if trainer.is_global_zero:
            if not self.quiet:
                logging.info(f"Logged {len(metrics)} {prefix} metrics.")
                rprint(metrics_df)

            for logger in trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    logger.experiment.add_text(
                        f"{prefix}_metrics", metrics_df.to_markdown()
                    )
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_validation_epoch_start(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs, prefix="test")


class MultiTaskDecodingStitchEvaluator(L.Callback):
    def __init__(self, dataset_config_dict: dict):
        metrics = defaultdict(lambda: defaultdict(dict))
        # setup the metrics
        for recording_id, recording_config in dataset_config_dict.items():
            for readout_config in recording_config["multitask_readout"]:
                readout_id = readout_config["readout_id"]
                for metric_config in readout_config["metrics"]:
                    metric = hydra.utils.instantiate(metric_config["metric"])
                    metrics[recording_id][readout_id][str(metric)] = metric
        self.metrics = metrics

    def on_validation_epoch_start(self, trainer, pl_module):
        # prepare a cache for each contiguous sequence
        self.sequence_index = trainer.datamodule.val_sequence_index
        num_sequences = self.sequence_index.max().item() + 1
        self.sample_ptr = 0

        self.cache = [
            {
                "target": defaultdict(list),
                "pred": defaultdict(list),
                "timestamps": defaultdict(list),
                "subtask_index": defaultdict(list),
            }
            for _ in range(num_sequences)
        ]

        self.counter = [0] * num_sequences
        # set the target of the couter based on unique in sequence_index
        # use torch.unique to get the count
        _, self.cache_flush_threshold = torch.unique(
            self.sequence_index, return_counts=True
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        target_values = batch.pop("target_values")
        absolute_starts = batch.pop("absolute_start")
        session_ids = batch.pop("session_id")
        output_subtask_index = batch.pop("output_subtask_index")

        # forward pass
        output_values = outputs  # pl_module.model(**batch, unpack_output=True)

        # update the cache with the predictions and targets
        for readout_index in torch.unique(batch["output_decoder_index"]):
            if readout_index.item() == 0:
                # skip the padding token
                continue

            mask = batch["output_decoder_index"] == readout_index
            readout_id = torch_brain.get_modality_by_id(readout_index.item())

            token_sample_idx = torch.where(mask)[0]

            for i in torch.unique(token_sample_idx):
                pred = output_values[i][readout_id]
                target = target_values[readout_id][token_sample_idx == i]
                timestamps = (
                    batch["output_timestamps"][mask][token_sample_idx == i]
                    + absolute_starts[i]
                )
                subtask_idx = output_subtask_index[readout_id][token_sample_idx == i]

                self.cache[self.sequence_index[self.sample_ptr]]["pred"][
                    readout_id
                ].append(pred.detach().cpu())
                self.cache[self.sequence_index[self.sample_ptr]]["target"][
                    readout_id
                ].append(target.detach().cpu())
                self.cache[self.sequence_index[self.sample_ptr]]["timestamps"][
                    readout_id
                ].append(timestamps.detach().cpu())
                self.cache[self.sequence_index[self.sample_ptr]]["subtask_index"][
                    readout_id
                ].append(subtask_idx.detach().cpu())

        # update counter then check if the cache should be flushed
        for i in range(len(output_values)):
            j = self.sequence_index[self.sample_ptr]
            self.counter[j] += 1
            self.sample_ptr += 1

            if self.counter[j] >= self.cache_flush_threshold[j]:
                self.flush_cache(j, session_id=session_ids[i])

    def flush_cache(self, i, session_id):
        for task_name in self.cache[i]["pred"].keys():
            pred = torch.cat(self.cache[i]["pred"][task_name])
            timestamps = torch.cat(self.cache[i]["timestamps"][task_name])
            subtask_index = torch.cat(self.cache[i]["subtask_index"][task_name])
            target = torch.cat(self.cache[i]["target"][task_name])

            # Pool data wherever timestamps overlap
            stitched_pred = stitch(timestamps, pred)
            stitched_target = stitch(timestamps, target)

            if target.dtype == torch.long:
                stitched_target = torch.round(stitched_target).long()

            for metric_name in self.metrics[session_id][task_name].keys():
                self.metrics[session_id][task_name][metric_name].update(
                    stitched_pred, stitched_target
                )

        # delete the cache to free memory
        self.cache[i] = None

    def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
        # check that all caches have been flushed
        for i, cache in enumerate(self.cache):
            if cache is not None:
                raise RuntimeError(
                    f"Cache at index {i} was not flushed before end of validation epoch. "
                    "This likely indicates a bug in the cache flushing logic."
                )

        metrics = {}
        for recording_id in self.metrics.keys():
            for task_name in self.metrics[recording_id].keys():
                for metric_name in self.metrics[recording_id][task_name].keys():
                    metrics[f"{recording_id}/{task_name}/{metric_name}/{prefix}"] = (
                        self.metrics[recording_id][task_name][metric_name]
                        .to(pl_module.device)
                        .compute()
                    )
                    self.metrics[recording_id][task_name][metric_name].reset()
                    self.metrics[recording_id][task_name][metric_name].to("cpu")

        # compute the average metric
        metrics[f"average_{prefix}_metric"] = torch.tensor(
            list(metrics.values())
        ).mean()

        # log the metrics
        self.log_dict(metrics)
        logging.info(f"Logged {len(metrics)} {prefix} metrics.")

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value.item()})

        metrics_df = pd.DataFrame(metrics_data)
        rprint(metrics_df)

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    logger.experiment.add_text(
                        f"{prefix}_metrics", metrics_df.to_markdown()
                    )
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_validation_epoch_start(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs, prefix="test")
