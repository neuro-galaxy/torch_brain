from collections.abc import Iterable
from typing import Union, Any, Optional
from copy import deepcopy

import torch
from torch import Tensor, nn
import torchmetrics
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.wrappers import MultitaskWrapper

from torch_brain.utils.stitcher import stitch
from torch_brain.data.sampler import StitcherSamplerWrapper


class TorchBrainMetricWrapper(WrapperMetric):
    """A wrapper for computing metrics across multiple recordings with intelligent data stitching.

    This class manages a collection of torchmetrics.Metric objects, one for each recording in your dataset.
    The key feature is that it can "stitch together" predictions from overlapping time windows before
    computing the final metrics.

    **How Stitching Works:**
    When you have overlapping time windows in your data, the same time points might appear in multiple
    predictions. Instead of computing metrics on each window separately, this wrapper:
    1. Collects all predictions, targets, and timestamps during an epoch
    2. Uses timestamps to identify overlapping regions
    3. Combines (stitches) the overlapping predictions intelligently
    4. Computes metrics on the final stitched result

    **Timestamps Requirements:**
    - Must be absolute timestamps (relative to the start of each recording)
    - Should NOT be relative to individual context windows
    - Used to determine which predictions overlap and need stitching

    **Memory Optimization:**
    By default, all data is cached until the end of the epoch. For large datasets, you can use
    `convert_to_stitcher_sampler()` to enable smart memory management that processes data in chunks.

    Args:
        metrics (dict): A dictionary mapping recording names to their corresponding metrics.
            Each value can be a Metric, MetricCollection, or MultitaskWrapper.
        prefix (str, optional): String to add before metric names in output. Defaults to "".
        postfix (str, optional): String to add after metric names in output. Defaults to "".
        stitch (bool, optional): Whether to enable the stitching functionality. Defaults to False.

    Raises:
        TypeError: If `metrics` is not a dictionary or contains invalid metric types.
        ValueError: If `prefix` or `postfix` are not strings.

    Example:
        Basic usage:

        >>> import torch
        >>> from torchmetrics.regression import MeanSquaredError
        >>> from torch_brain.metrics import TorchBrainMetricWrapper
        >>>
        >>> # Set up metrics for two recordings
        >>> metrics = TorchBrainMetricWrapper({
        ...     "recording_001": MeanSquaredError(),
        ...     "recording_002": MeanSquaredError()
        ... })
        >>>
        >>> preds = torch.tensor([1.0, 2.0, 3.0])      # Your model predictions
        >>> targets = torch.tensor([1.1, 2.1, 3.1])    # Ground truth values
        >>>
        >>> metrics.update(preds, targets, None, recording_id="recording_001")
        >>>
        >>> metric.update(preds, targets, None, recording_id="recording_002")
        >>>
        >>> # At epoch end, compute final metrics
        >>> results = metrics.compute()
        >>> # Results: {"recording_001": tensor(0.01), "recording_002": tensor(0.01)}

        Example that uses stitching:

        >>> import torch
        >>> from torchmetrics.regression import MeanSquaredError
        >>> from torch_brain.metrics import TorchBrainMetricWrapper
        >>>
        >>> # Set up metrics for two recordings
        >>> metrics = TorchBrainMetricWrapper({
        ...     "recording_001": MeanSquaredError(),
        ...     "recording_002": MeanSquaredError()
        ... }, stitch=True)
        >>>
        >>> preds = torch.tensor([1.0, 2.0, 3.0])      # Your model predictions
        >>> targets = torch.tensor([1.1, 2.1, 3.1])    # Ground truth values
        >>> timestamps = torch.tensor([10.0, 11.0, 12.0])  # Absolute time in recording
        >>>
        >>> metrics.update(preds, targets, timestamps, recording_id="recording_001")
        >>>
        >>> preds = torch.tensor([2.0, 3.0, 4.0])      # Your model predictions
        >>> targets = torch.tensor([2.1, 3.1, 4.1])    # Ground truth values
        >>> timestamps = torch.tensor([11.0, 12.0, 13.0])  # Absolute time in recording
        >>>
        >>> metrics.update(preds, targets, timestamps, recording_id="recording_001")
        >>>
        >>> metric.update(preds, targets, timestamps, recording_id="recording_002")
        >>>
        >>> # At epoch end, compute final metrics
        >>> results = metrics.compute()
        >>> # Results: {"recording_001": tensor(0.01), "recording_002": tensor(0.01)}

        Example with multiple metrics:

        >>> import torch
        >>> from torchmetrics.regression import MeanSquaredError, R2Score
        >>> from torchmetrics import MetricCollection
        >>> from torch_brain.metrics import TorchBrainMetricWrapper
        >>>
        >>> # Set up metrics for two recordings
        >>> metrics = TorchBrainMetricWrapper({
        ...     "recording_001": MetricCollection({
        ...         "mse": MeanSquaredError(),
        ...         "r2": R2Score()
        ...     }),
        ...     "recording_002": MetricCollection({
        ...         "mse": MeanSquaredError(),
        ...         "r2": R2Score()
        ...     })
        ... }, stitch=True)
        >>>
        >>> preds = torch.tensor([1.0, 2.0, 3.0])      # Your model predictions
        >>> targets = torch.tensor([1.1, 2.1, 3.1])    # Ground truth values
        >>> timestamps = torch.tensor([10.0, 11.0, 12.0])  # Absolute time in recording
        >>>
        >>> metrics.update(preds, targets, timestamps, recording_id="recording_001")
        >>>
        >>> preds = torch.tensor([2.0, 3.0, 4.0])      # Your model predictions
        >>> targets = torch.tensor([2.1, 3.1, 4.1])    # Ground truth values
        >>> timestamps = torch.tensor([11.0, 12.0, 13.0])  # Absolute time in recording
        >>>
        >>> metrics.update(preds, targets, timestamps, recording_id="recording_001")
        >>>
        >>> metric.update(preds, targets, timestamps, recording_id="recording_002")
        >>>
        >>> # At epoch end, compute final metrics
        >>> results = metrics.compute()
        >>> # Results: {"recording_001": {"mse": tensor(0.01), "r2": tensor(0.99)}, "recording_002": {"mse": tensor(0.01), "r2": tensor(0.99)}}

        Example with multiple tasks:

        >>> import torch
        >>> from torchmetrics.regression import MeanSquaredError, R2Score
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> from torchmetrics import MetricCollection, MultitaskWrapper
        >>> from torch_brain.metrics import TorchBrainMetricWrapper
        >>>
        >>> # Set up metrics for two recordings
        >>> metrics = TorchBrainMetricWrapper({
        ...     "recording_001": MultitaskWrapper({
        ...         "task_0": MetricCollection({
        ...             "mse": MeanSquaredError(),
        ...             "r2": R2Score()
        ...         }),
        ...         "task_1": MulticlassAccuracy(num_classes=3),
        ...     }),
        ...     "recording_002": MultitaskWrapper({
        ...         "task_0": MetricCollection({
        ...             "mse": MeanSquaredError(),
        ...             "r2": R2Score()
        ...         }),
        ...         "task_3": MulticlassAccuracy(num_classes=5),
        ...     })
        ... }, stitch=True)
        >>>
        >>> preds = {"task_0": torch.tensor([1.0, 2.0, 3.0]), "task_1": torch.tensor([1.0, 2.0, 3.0])}
        >>> targets = {"task_0": torch.tensor([1, 2, 3]), "task_1": torch.tensor([1, 2, 3])}
        >>> timestamps = {"task_0": torch.tensor([10.0, 11.0, 12.0]), "task_1": torch.tensor([10.0, 11.0, 12.0])}
        >>>
        >>> metrics.update(preds, targets, timestamps, recording_id="recording_001")
        >>>
        >>> preds = {"task_0": torch.tensor([2.0, 3.0, 4.0]), "task_1": torch.tensor([2.0, 3.0, 4.0])}
        >>> targets = {"task_0": torch.tensor([2, 3, 4]), "task_1": torch.tensor([2, 3, 4])}
        >>> timestamps = {"task_0": torch.tensor([11.0, 12.0, 13.0]), "task_1": torch.tensor([11.0, 12.0, 13.0])}
        >>> ...

        Example with efficient memory caching:
        >>> from torch_brain.data.sampler import SequentialFixedWindowSampler
        >>>
        >>> metrics = TorchBrainMetricWrapper(...)
        >>> # Create your sampler as usual
        >>> sampler = SequentialFixedWindowSampler(dataset, window_size=1000)
        >>>
        >>> # Convert it to enable smart memory management
        >>> sampler = metrics.convert_to_stitcher_sampler(sampler)
        >>>
        >>> # Now the wrapper will automatically flush data when appropriate
        >>> # reducing memory usage for large datasets

    """

    is_differentiable: bool = False

    def __init__(
        self,
        metrics: dict[str, Union[Metric, MetricCollection, MultitaskWrapper]],
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        stitch=False,
    ) -> None:
        super().__init__()

        if not isinstance(metrics, dict):
            raise TypeError(
                f"Expected argument `metrics` to be a dict. Found metrics = {metrics}"
            )

        for metric in metrics.values():
            if not isinstance(metric, (Metric, MetricCollection, MultitaskWrapper)):
                raise TypeError(
                    f"Expected each metric to be a Metric or a MetricCollection or a MultitaskWrapper. Found a metric of type {type(metric)}"
                )

        self.metrics = nn.ModuleDict(metrics)

        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(
                f"Expected argument `prefix` to either be `None` or a string but got {prefix}"
            )
        self._prefix = prefix or ""

        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(
                f"Expected argument `postfix` to either be `None` or a string but got {postfix}"
            )
        self._postfix = postfix or ""

        self._stitch = stitch

        if self._stitch:
            self._sequence_index = None
            self._smart_flushing = False
            self._init_cache()

    def _init_cache(self):
        if not self._smart_flushing:
            self._cache = {
                recording_id: {"preds": [], "targets": [], "timestamps": []}
                for recording_id in self.metrics.keys()
            }

        else:
            assert self._sequence_index is not None
            num_sequences = self._sequence_index.max().item() + 1

            self._cache = [
                {"preds": [], "targets": [], "timestamps": []}
                for _ in range(num_sequences)
            ]

            self._sample_ptr = 0
            self._counter = [0] * num_sequences
            # set the target of the couter based on unique in sequence_index
            # use torch.unique to get the count
            _, self._cache_flush_threshold = torch.unique(
                self._sequence_index, return_counts=True
            )

    def update(
        self,
        preds: dict[str, Union[Tensor, dict[str, Tensor]]],
        targets: dict[str, Union[Tensor, dict[str, Tensor]]],
        timestamps: dict[str, Union[Tensor, dict[str, Tensor]]],
        recording_id: str,
    ) -> None:
        """Update each recording's metrics with its corresponding pred and target.

        Args:
            preds: Dictionary associating each recording to a Tensor (or dictionary of Tensors) of preds.
            targets: Dictionary associating each recording to a Tensor (or dictionary of Tensors) of targets.
            timestamps: Dictionary associating each recording to a Tensor (or dictionary of Tensors) of timestamps.
            recording_id: The id of the recording to update the metrics for.
        """
        if not self._stitch:
            # no stitching required, just update the metrics
            self.metrics[recording_id].update(preds, targets)

        else:
            # stitching required, update the cache
            if not self._smart_flushing:
                _cache = self._cache[recording_id]
            else:
                j = self._sequence_index[self._sample_ptr]
                _cache = self._cache[j]

            _cache["preds"].append(preds)
            _cache["targets"].append(targets)
            _cache["timestamps"].append(timestamps)

            if self._smart_flushing:
                # flag that the cache has been updated
                self._counter[j] += 1
                self._sample_ptr += 1
                # check if the cache can be flushed
                if self._counter[j] >= self._cache_flush_threshold[j]:
                    self._process_cache_element(
                        self._cache[j], self.metrics[recording_id]
                    )
                    # delete the cache to free memory
                    self._cache[j] = None

    @staticmethod
    def _concat_and_stitch(preds_list, targets_list, timestamps_list):
        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)
        timestamps = torch.cat(timestamps_list)

        # Pool data wherever timestamps overlap
        stitched_preds = stitch(timestamps, preds)
        stitched_targets = stitch(timestamps, targets)

        if targets.dtype == torch.long:
            stitched_targets = torch.round(stitched_targets).long()

        return stitched_preds, stitched_targets

    def _process_cache_element(self, _cache, metric: torchmetrics.Metric):
        preds_list = _cache["preds"]
        targets_list = _cache["targets"]
        timestamps_list = _cache["timestamps"]

        if isinstance(metric, MultitaskWrapper):
            task_preds, task_targets = {}, {}
            for task_name in metric.task_metrics.keys():
                task_preds_list = [e[task_name] for e in preds_list]
                task_targets_list = [e[task_name] for e in targets_list]
                task_timestamps_list = [e[task_name] for e in timestamps_list]

                stitched_task_preds, stitched_task_targets = self._concat_and_stitch(
                    task_preds_list, task_targets_list, task_timestamps_list
                )

                task_preds[task_name] = stitched_task_preds
                task_targets[task_name] = stitched_task_targets

            metric.update(task_preds, task_targets)
        else:
            preds, targets = self._concat_and_stitch(
                preds_list, targets_list, timestamps_list
            )
            metric.update(preds, targets)

    def _flush_cache(self):
        assert not self._smart_flushing

        for recording_id, metric in self.metrics.items():
            self._process_cache_element(self._cache[recording_id], metric)

        # delete the cache to free memory
        self._cache = None

    def compute(self) -> dict[str, Any]:
        """Compute metrics for all tasks."""
        if self._stitch and not self._smart_flushing:
            self._flush_cache()
        elif self._stitch and self._smart_flushing:
            # cache should be empty because flushing is automatically triggered when all
            # expected samples for a given sequence have been seen
            for i, cache in enumerate(self._cache):
                if cache is not None:
                    raise RuntimeError(
                        f"Cache at index {i} was not automatically flushed before end of epoch. This indicates that not all expected updates were received."
                        "."
                    )

        return self._convert_output(
            {
                recording_id: metric.compute()
                for recording_id, metric in self.metrics.items()
            }
        )

    def reset(self) -> None:
        """Reset all underlying metrics."""
        if self._stitch:
            self._init_cache()

        for metric in self.metrics.values():
            metric.reset()
        super().reset()

    def convert_to_stitcher_sampler(self, sampler, num_replicas=1, rank=0):
        """Convert a sampler to a stitcher sampler.

        This method is used to allow the tracking of the origin of the data across
        multiple recordings, and enable smart flushing of the cache.

        Args:
            sampler: The sampler to convert.
            num_replicas: The number of replicas in the distributed training.
            rank: The rank of the current process.
        """
        if not self._stitch:
            raise ValueError(
                "Stitching is not enabled, so a stitcher sampler cannot be created."
            )

        stitch_sampler = StitcherSamplerWrapper(
            sampler, num_replicas=num_replicas, rank=rank
        )
        self._sequence_index = stitch_sampler.sequence_index
        # enable smart flushing
        self._smart_flushing = True
        self._init_cache()
        return stitch_sampler

    def items(self, flatten: bool = True) -> Iterable[tuple[str, nn.Module]]:
        """Iterate over recording metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the recording names and the corresponding metrics.

        """
        for recording_id, metric in self.metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name, sub_metric in metric.items():
                    yield f"{self._prefix}{recording_id}_{sub_metric_name}{self._postfix}", sub_metric
            else:
                yield f"{self._prefix}{recording_id}{self._postfix}", metric

    def keys(self, flatten: bool = True) -> Iterable[str]:
        """Iterate over recording names.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the recording names and the corresponding metrics.

        """
        for recording_id, metric in self.metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name in metric:
                    yield f"{self._prefix}{recording_id}_{sub_metric_name}{self._postfix}"
            else:
                yield f"{self._prefix}{recording_id}{self._postfix}"

    def values(self, flatten: bool = True) -> Iterable[nn.Module]:
        """Iterate over recording metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the recording names and the corresponding metrics.

        """
        for metric in self.metrics.values():
            if flatten and isinstance(metric, MetricCollection):
                yield from metric.values()
            else:
                yield metric

    def _convert_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Convert the output of the underlying metrics to a dictionary with the recording names as keys."""
        return {
            f"{self._prefix}{recording_id}{self._postfix}": recording_output
            for recording_id, recording_output in output.items()
        }
