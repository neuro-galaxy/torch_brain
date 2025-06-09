from dataclasses import dataclass
import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
from rich import print as rprint
import torch
import lightning as L
import torchmetrics

try:
    import wandb
except ImportError:
    wandb = None

import torch_brain
from torch_brain.registry import ModalitySpec, DataType


def stitch(
    timestamps: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Pools values that share the same timestamp using mean or mode operations.

    This function is useful when you have multiple predictions or values for the same
    timestamp (e.g., from overlapping windows) and need to combine them into a single
    value per timestamp.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps. Shape: (N,)
        values (torch.Tensor): A tensor of values corresponding to the timestamps.
            Shape:
                - For floating point types: (N, ...)
                - For categorical types (torch.long only): (N,)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - unique_timestamps: A 1D tensor of sorted unique timestamps
            - pooled_values: A tensor containing the pooled values for each unique timestamp
                - For continuous data (float types): Uses mean pooling
                - For categorical data (long type): Uses mode pooling

    Examples:
        >>> # Mean pooling for continuous values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3])
        >>> values = torch.tensor([0.1, 0.3, 0.2, 0.4, 0.6])
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([0.2000, 0.2000, 0.5000]))

        >>> # Mode pooling for categorical values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3, 3])
        >>> values = torch.tensor([1, 1, 2, 3, 3, 1], dtype=torch.long)
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([1, 2, 3]))
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    if values.dtype == torch.long:
        # Mode pooling for categorical values

        if values.ndim != 1:
            raise ValueError(
                "For categorical values (long type), only 1D tensors are supported. "
                f"Got values with shape {values.shape} instead."
            )

        # 1. Construct a N x C class-wise vote tensor
        votes = values.new_zeros((len(unique_timestamps), values.max() + 1))
        votes.index_put_((indices, values), torch.ones_like(indices), accumulate=True)
        # 2. Mode class is the one with most votes
        mode_values = votes.argmax(dim=-1)
        return unique_timestamps, mode_values

    elif torch.is_floating_point(values):
        # Mean-pool for floating points
        # 1. Count occurrences of each unique timestamp
        counts = torch.zeros_like(unique_timestamps, dtype=torch.long)
        counts.index_add_(0, indices, torch.ones_like(indices))
        if values.dim() > 1:
            counts = counts.unsqueeze(-1)
        # 2. Accumulate and average values for each unique timestamp
        avg_values = values.new_zeros((len(unique_timestamps), *values.shape[1:]))
        avg_values.index_add_(0, indices, values).div_(counts)
        # Regarding division by zero: all elements of counts will be >= 1.
        # Reasoning: Since it was built using unique_timestamps, each index will have
        # atleast one timestamp attached to it.

        return unique_timestamps, avg_values

    else:
        raise TypeError(
            f"Unsupported dtype {values.dtype} for stitching. "
            "Only floating points supported for mean pooling, "
            " and torch.long type supported for mode pooling."
        )


@dataclass
class DataForDecodingStitchEvaluator:
    r"""A batch's worth of data for :class:`DecodingStitchEvaluator`"""

    timestamps: torch.FloatTensor  # B x T_max
    preds: torch.FloatTensor  # B x T_max x D_output
    targets: torch.FloatTensor  # B x T_max x D_output
    eval_masks: torch.BoolTensor  # B x T_max
    session_ids: List[str]  # A list of session ID strings, 1 for each sample in batch
    absolute_starts: torch.Tensor  # Batch


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
    :class:`~torch_brain.data.sampler.DistributedStitchingFixedWindowSampler`
    which ensures that the sequences are split across GPUs in a way that allows for
    correct stitching.

    This callback is called _after_ the validation_step, and expects you to return a
    :class:`~DataForDecodingStitchEvaluator` object from the validation_step lightning
    module method.
    Please refer to the examples/poyo/train.py script for an example of how to write
    such a validation_step(...) function.

    This callback operates by maintaining a cache of the predictions, targets, and
    timestamps for each session. This cache is updated at the end of each batch.
    Finally, at the end of the epoch, the cache is coalesced and the metric is
    computed for each session.
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

        self.dim = modality_spec.dim
        self.data_type = modality_spec.type

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

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        data: DataForDecodingStitchEvaluator,
        *args,
        **kwargs,
    ):
        # Update the cache with the predictions, targets, and timestamps
        batch_size = len(data.timestamps)
        for i in range(batch_size):
            mask = data.eval_masks[i]
            session_id = data.session_ids[i]
            absolute_start = data.absolute_starts[i]

            pred = data.preds[i][mask]
            target = data.targets[i][mask]
            timestamps = data.timestamps[i][mask] + absolute_start

            self.cache[session_id]["pred"].append(pred.detach())
            self.cache[session_id]["target"].append(target.detach())
            self.cache[session_id]["timestamps"].append(timestamps.detach())

    def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
        # compute metric for each session
        metrics = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self.cache[session_id]
            if len(cache["pred"]) > 0:
                pred = torch.cat(cache["pred"])
                target = torch.cat(cache["target"])
                timestamps = torch.cat(cache["timestamps"])

                stitched_pred = stitch(timestamps, pred)[1]
                stitched_target = stitch(timestamps, target)[1]

                metric_fn.to(pl_module.device).update(stitched_pred, stitched_target)
            else:
                if self.data_type == DataType.CONTINUOUS:
                    metric_fn.to(pl_module.device).update(
                        torch.empty(0, self.dim, device=pl_module.device),
                        torch.empty(0, self.dim, device=pl_module.device),
                    )
                else:
                    metric_fn.to(pl_module.device).update(
                        torch.empty(0, self.dim, device=pl_module.device),
                        torch.empty(0, device=pl_module.device),
                    )

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
                if (
                    isinstance(logger, L.pytorch.loggers.WandbLogger)
                    and wandb is not None
                ):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_validation_epoch_start(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs, prefix="test")


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
            stitched_pred = stitch(timestamps, pred)[1]
            stitched_target = stitch(timestamps, target)[1]

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
                if (
                    isinstance(logger, L.pytorch.loggers.WandbLogger)
                    and wandb is not None
                ):
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
