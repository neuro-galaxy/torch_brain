from dataclasses import dataclass
import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

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


class DecodingStitchEvaluator:
    r"""A convenient stitching and evaluation framework for handling overlapping time windows in model predictions.

    This class is useful when:
    1. Your model outputs have associated timestamps
    2. Your sampling strategy involves overlapping time windows, requiring stitching to
       coalesce the predictions and targets before computing evaluation metrics
    3. (Optional) You are training on multiple sessions/recordings and want to compute
       metrics for each session individually

    This class handles stitching of predictions and targets for each session and computes
    metrics individually. The average metric across all sessions is also computed and logged.

    Note:
        Since stitching is done only on tensors on the same GPU, sequences split across
        multiple GPUs will not be stitched together. In this case, use a stitching-aware
        sampler like :class:`~torch_brain.data.sampler.DistributedStitchingFixedWindowSampler`
        which ensures correct sequence splitting across GPUs.

    This callback is called _after_ the validation_step, and expects you to return a
    :class:`~DataForDecodingStitchEvaluator` object from the validation_step lightning
    module method.
    Please refer to the examples/poyo/train.py script for an example of how to write
    such a validation_step(...) function.

    This callback operates by maintaining a cache of the predictions, targets, and
    timestamps for each session. The cache is updated using :meth:`.update`,
    once you're ready to stitch and compute metrics, call :meth:`.compute`, and reset
    the cache with :meth:`.reset`.

    Example:
        >>> # Initialize evaluator
        >>> stitch_evaluator = DecodingStitchEvaluator(
        ...     session_ids=session_ids,
        ...     modality_spec=modality_spec
        ... )
        >>>
        >>> # Update cache at end of each validation/test batch
        >>> stitch_evaluator.update(
        ...     timestamps=batch_timestamps,     # FloatTensor, [B, N]
        ...     preds=batch_predictions,         # Tensor, [B, N, D]
        ...     targets=batch_targets,           # Tensor, [B, N, D]
        ...     eval_masks=batch_masks,          # BoolTensor, [B, N]
        ...     session_ids=batch_session_ids,   # List[str], length=B
        ...     absolute_starts=batch_starts,    # FloatTensor, [B]
        ... )
        >>>
        >>> # Compute metrics at end of validation/test epoch
        >>> metric_dict = stitch_evaluator.compute()
        >>> stitch_evaluator.reset()  # Reset cache for next epoch
    """

    def __init__(
        self,
        session_ids: Iterable[str],
        modality_spec: Optional[ModalitySpec] = None,
        metric_factory: Optional[Callable[[int], ModalitySpec]] = None,
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
        self._init_cache()

    def _init_cache(self):
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

    def update(
        self,
        timestamps: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        eval_masks: torch.Tensor,
        session_ids: List[str],
        absolute_starts: torch.Tensor,
    ):
        r"""Update the validation cache with predictions, targets, and timestamps.

        Args:
            timestamps: A tensor of shape (batch_size, seq_len) containing timestamps
                for each prediction
            preds: A tensor of shape (batch_size, seq_len, dim) containing model predictions
            targets: A tensor of shape (batch_size, seq_len, dim) containing target values
            eval_masks: A tensor of shape (batch_size, seq_len) containing boolean masks
                indicating which timesteps should be evaluated
            session_ids: A list of strings of length batch_size containing session IDs
                for each sequence
            absolute_starts: A tensor of shape (batch_size,) containing the absolute start
                time of each sequence (since timestamps are expected to be relative to
                the sample start time)
        """
        batch_size = len(timestamps)
        for i in range(batch_size):
            mask = eval_masks[i]
            session_id = session_ids[i]

            _preds = preds[i][mask]
            _targets = targets[i][mask]
            _timestamps = timestamps[i][mask] + absolute_starts[i]

            self.cache[session_id]["pred"].append(_preds.detach())
            self.cache[session_id]["target"].append(_targets.detach())
            self.cache[session_id]["timestamps"].append(_timestamps.detach())

    def compute(self):
        r"""Stitch/Coalesce the cache using :func:`stitch`, and compute the metrics
        based on the metric function provided.

        Returns: A dictionary of computed metrics, with keys being recording IDs.
        """
        metric_dict = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self.cache[session_id]
            pred = torch.cat(cache["pred"])
            target = torch.cat(cache["target"])
            timestamps = torch.cat(cache["timestamps"])

            stitched_pred = stitch(timestamps, pred)
            stitched_target = stitch(timestamps, target)

            device = stitched_pred.device
            metric_fn.to(device).update(stitched_pred, stitched_target)
            metric_dict[session_id] = metric_fn.compute().item()
            metric_fn.reset()

        return metric_dict

    def reset(self):
        r"""Reset the cache. Should be called at the end of validation epoch."""
        self._init_cache()


@dataclass
class DataForMultiTaskDecodingStitchEvaluator:
    r"""A batch's worth of data for :class:`MultiTaskDecodingStitchEvaluator`"""

    timestamps: torch.FloatTensor  # B x T_max
    preds: List[Dict[str, torch.Tensor]]  # B-long list, Dict keys are task names
    targets: List[Dict[str, torch.Tensor]]  #  B-long list, Dict keys are task names
    decoder_indices: torch.LongTensor  # B x T_max
    # eval_masks: Keyed by task names, each tensor is mask that can be applied to a
    # task-concatenated tensor of predictions (look at output format of
    # `torch_brain.nn.multitask_readout.MultitaskReadout`)
    eval_masks: Dict[str, torch.BoolTensor]
    session_ids: List[str]  # A list of session ID strings, 1 for each sample in batch
    absolute_starts: torch.Tensor  # Batch


class MultiTaskDecodingStitchEvaluator(L.Callback):
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def on_validation_epoch_start(self, trainer, pl_module):
        self._setup_cache(trainer, mode="val")

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        data: DataForMultiTaskDecodingStitchEvaluator,
        *args,
        **kwargs,
    ):
        # update the cache with the predictions and targets
        for readout_index in torch.unique(data.decoder_indices):
            if readout_index.item() == 0:
                # skip the padding token
                continue

            mask = data.decoder_indices == readout_index
            readout_id = torch_brain.get_modality_by_id(readout_index.item())

            token_sample_idx = torch.where(mask)[0]

            curr_sample_ptr = self.sample_ptr

            for i in torch.unique(token_sample_idx):
                pred = data.preds[i][readout_id]
                target = data.targets[readout_id][token_sample_idx == i]
                timestamps = (
                    data.timestamps[mask][token_sample_idx == i]
                    + data.absolute_starts[i]
                )
                eval_mask = data.eval_masks[readout_id][token_sample_idx == i]

                timestamps = timestamps[eval_mask]
                pred = pred[eval_mask]
                target = target[eval_mask]

                self.cache[self.sequence_index[curr_sample_ptr]]["pred"][
                    readout_id
                ].append(pred.detach().cpu())
                self.cache[self.sequence_index[curr_sample_ptr]]["target"][
                    readout_id
                ].append(target.detach().cpu())
                self.cache[self.sequence_index[curr_sample_ptr]]["timestamps"][
                    readout_id
                ].append(timestamps.detach().cpu())

                curr_sample_ptr += 1

        # update counter then check if the cache should be flushed
        for i in range(len(data.preds)):
            j = self.sequence_index[self.sample_ptr]
            self.counter[j] += 1
            self.sample_ptr += 1

            if self.counter[j] >= self.cache_flush_threshold[j]:
                self.flush_cache(j, session_id=data.session_ids[i])

    def flush_cache(self, i, session_id):
        for task_name in self.cache[i]["pred"].keys():
            pred = torch.cat(self.cache[i]["pred"][task_name])
            timestamps = torch.cat(self.cache[i]["timestamps"][task_name])
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

    def on_test_epoch_start(self, trainer, pl_module):
        self._setup_cache(trainer, mode="test")

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs, prefix="test")

    def _setup_cache(self, trainer, mode: str = "val"):
        if mode == "val":
            self.sequence_index = trainer.datamodule.val_sequence_index
        elif mode == "test":
            self.sequence_index = trainer.datamodule.test_sequence_index
        else:
            raise ValueError(f"Invalid mode: {mode}")

        num_sequences = self.sequence_index.max().item() + 1
        self.sample_ptr = 0

        self.cache = [
            {
                "target": defaultdict(list),
                "pred": defaultdict(list),
                "timestamps": defaultdict(list),
            }
            for _ in range(num_sequences)
        ]

        self.counter = [0] * num_sequences
        # set the target of the couter based on unique in sequence_index
        # use torch.unique to get the count
        _, self.cache_flush_threshold = torch.unique(
            self.sequence_index, return_counts=True
        )
