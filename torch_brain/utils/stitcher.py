from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torchmetrics

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
        >>> from torch_brain.registry import MODALITY_REGISTRY
        >>>
        >>> B = 16   # batch size
        >>> N = 100  # tokens per sample
        >>> D = 2    # prediction dimension
        >>> num_epochs = 3
        >>> num_steps_per_epoch = 10
        >>> session_ids = ["session1", "session2", "session3"]
        >>> modality_spec =  MODALITY_REGISTRY["cursor_velocity_2d"]
        >>>
        >>> # Initialize evaluator
        >>> stitch_evaluator = DecodingStitchEvaluator(
        ...     session_ids=session_ids,
        ...     modality_spec=modality_spec
        ... )
        >>>
        >>> # Train loop
        >>> for epoch in range(num_epochs):
        ...     # Training epoch
        ...     # ...
        ...
        ...     # Vaidation epoch:
        ...     for batch_idx in range(num_steps_per_epoch):
        ...         # Dummy batch data
        ...         batch_timestamps = torch.linspace(0, 1, N).repeat(B, 1)
        ...         batch_predictions = torch.rand(B, N, D)
        ...         batch_targets = torch.rand(B, N, D)
        ...         batch_eval_masks = torch.rand(B, N) > 0.5
        ...         batch_session_ids = [session_ids[idx] for idx in torch.randint(3, (B,))]
        ...         batch_absolute_starts = torch.rand(B)
        ...
        ...         # Update cache at end of each validation/test batch
        ...         stitch_evaluator.update(
        ...             timestamps=batch_timestamps,           # FloatTensor, [B, N]
        ...             preds=batch_predictions,               # Tensor, [B, N, D]
        ...             targets=batch_targets,                 # Tensor, [B, N, *]
        ...             eval_masks=batch_eval_masks,           # BoolTensor, [B, N]
        ...             session_ids=batch_session_ids,         # List[str], length=B
        ...             absolute_starts=batch_absolute_starts, # FloatTensor, [B]
        ...         )
        ...
        ...     # Compute metrics at end of validation/test epoch
        ...     metric_dict = stitch_evaluator.compute()
        ...     stitch_evaluator.reset()  # Reset cache for next epoch
    """

    def __init__(
        self,
        session_ids: Iterable[str],
        modality_spec: Optional[ModalitySpec] = None,
        metric_factory: Optional[Callable[[int], ModalitySpec]] = None,
    ):
        r"""
        Args:
            session_ids: An iterable of session IDs for which the metrics are to be computed.
            modality_spec: (Optional) The modality specification for the task. Either this
                or metric_factory must be provided.
            metric_factory: (Optional) A callable that returns an instance of the metric to be used.
                If not provided, the metric is inferred based on the modality_spec.
        """

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
        self._cache = defaultdict(
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

            self._cache[session_id]["pred"].append(_preds.detach())
            self._cache[session_id]["target"].append(_targets.detach())
            self._cache[session_id]["timestamps"].append(_timestamps.detach())

    def compute(self):
        r"""Stitch/Coalesce the cache using :func:`stitch`, and compute the metrics
        based on the metric function provided.

        Returns: A dictionary of computed metrics, with keys being recording IDs.
        """
        metric_dict = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self._cache[session_id]
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


class MultiTaskDecodingStitchEvaluator:
    def __init__(
        self,
        metrics: Dict[str, torchmetrics.Metric],
        sequence_index: torch.Tensor,
    ):
        self.metrics = metrics
        self.sequence_index = sequence_index
        self._init_cache()

    def update(
        self,
        timestamps: torch.Tensor,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        readout_indices: torch.LongTensor,
        eval_masks: Dict[str, torch.BoolTensor],
        session_ids: List[str],
        absolute_starts: torch.Tensor,
    ):
        r"""
        Args:
            timestamps: A tensor of shape (batch_size, seq_len) containing timestamps
                for each prediction
            preds: A list of length batch_size, where each element is a dictionary.
                The key for each dictionary is the task name, and the value is the corresponding
                prediction tensor.
                This is expected to be the output of
                :class:`~torch_brain.nn.multitask_readout.MultitaskReadout`
            targets: Same as preds, but for targets.
                This is expected to be the "values" output of
                :class:`~torch_brain.nn.multitask_readout.prepare_for_multitask_readout`
            readout_indices: Expected to be the "readout_index" output of
                :function:`~torch_brain.nn.multitask_readout.prepare_for_multitask_readout`
            eval_masks: Expected to be the eval_mask output of
                :function:`~torch_brain.nn.multitask_readout.prepare_for_multitask_readout`
            session_ids: A list of session ID strings. The length of this list is equal
                to batch_size, i.e. one element for each sample in the batch
            absolute_starts: A tensor of shape (batch_size,) containing the absolute start
                time of each sequence (since timestamps are expected to be relative to
                the sample start time)
        """

        self._device = timestamps.device

        # update the cache with the predictions and targets
        for readout_index in torch.unique(readout_indices):
            if readout_index.item() == 0:
                # skip the padding token
                continue

            mask = readout_indices == readout_index
            readout_id = torch_brain.get_modality_by_id(readout_index.item())

            token_sample_idx = torch.where(mask)[0]

            curr_sample_ptr = self._sample_ptr

            for i in torch.unique(token_sample_idx):
                _eval_mask = eval_masks[readout_id][token_sample_idx == i]
                _pred = preds[i][readout_id][_eval_mask]
                _target = targets[readout_id][token_sample_idx == i][_eval_mask]
                _timestamps = (
                    timestamps[mask][token_sample_idx == i][_eval_mask]
                    + absolute_starts[i]
                )

                _cache = self._cache[self.sequence_index[curr_sample_ptr]]
                _cache["pred"][readout_id].append(_pred.detach().cpu())
                _cache["target"][readout_id].append(_target.detach().cpu())
                _cache["timestamps"][readout_id].append(_timestamps.detach().cpu())

                curr_sample_ptr += 1

        # update counter then check if the cache should be flushed
        for i in range(len(preds)):
            j = self.sequence_index[self._sample_ptr]
            self._counter[j] += 1
            self._sample_ptr += 1

            if self._counter[j] >= self._cache_flush_threshold[j]:
                self._flush_cache(j, session_id=session_ids[i])

    def compute(self):
        # check that all caches have been flushed
        for i, cache in enumerate(self._cache):
            if cache is not None:
                raise RuntimeError(
                    f"Cache at index {i} was not flushed before end of validation epoch. "
                    "This likely indicates a bug in the cache flushing logic."
                )

        metric_dict = {}
        for recording_id in self.metrics.keys():
            for task_name in self.metrics[recording_id].keys():
                for metric_name in self.metrics[recording_id][task_name].keys():
                    metric_dict[f"{recording_id}/{task_name}/{metric_name}"] = (
                        self.metrics[recording_id][task_name][metric_name]
                        .to(self._device)
                        .compute()
                        .item()
                    )
                    self.metrics[recording_id][task_name][metric_name].reset()
                    self.metrics[recording_id][task_name][metric_name].to("cpu")

        return metric_dict

    def reset(self):
        self._init_cache()

    def _init_cache(self):
        num_sequences = self.sequence_index.max().item() + 1
        self._sample_ptr = 0

        self._cache = [
            {
                "target": defaultdict(list),
                "pred": defaultdict(list),
                "timestamps": defaultdict(list),
            }
            for _ in range(num_sequences)
        ]

        self._counter = [0] * num_sequences
        # set the target of the couter based on unique in sequence_index
        # use torch.unique to get the count
        _, self._cache_flush_threshold = torch.unique(
            self.sequence_index, return_counts=True
        )

    def _flush_cache(self, i: int, session_id: str):
        for task_name in self._cache[i]["pred"].keys():
            pred = torch.cat(self._cache[i]["pred"][task_name])
            timestamps = torch.cat(self._cache[i]["timestamps"][task_name])
            target = torch.cat(self._cache[i]["target"][task_name])

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
        self._cache[i] = None
