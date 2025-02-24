from collections import defaultdict
from typing import Dict, List, Union

import torch
import torchmetrics

from torch_brain.utils.stitcher import stitch
from torch_brain.data.sampler import StitcherSamplerWrapper


class MetricGroupWithStitcher:
    r"""A group of `torchmetrics.Metric` objects, organized by recording
    (and optionally task), that stitch together multiple predictions, before
    computing the metrics.

    The stitching is done thanks to `timestamps` provided when calling `update`. The
    timestamps need to be absolute timestamps, in other words, relative to the start of
    the recording, and not relative to the start of the context window.

    Args:
        metrics (Union[Dict[str, torchmetrics.Metric], Dict[str, Dict[str, torchmetrics.Metric]],
                Dict[str, Dict[str, Dict[str, torchmetrics.Metric]]]]): nested dictionary
                of `torchmetrics.Metric` objects.

    This object holds a cache of predictions, targets and timestamps, until the end of the epoch,
    when the predictions are stitched together and the metrics are computed.

    To minimize memory usage, you can convert the sampler to a `StitcherSamplerWrapper`
    by calling `convert_to_stitcher_sampler` on the sampler.

    Example:

        >>> sampler = torch_brain.data.sampler.SequentialFixedWindowSampler(...)
        >>> metric_group = MetricGroup(metrics)
        >>> sampler = metric_group.convert_to_stitcher_sampler(sampler)
    """

    def __init__(
        self,
        metrics: Union[
            Dict[str, Union[torchmetrics.Metric, List[torchmetrics.Metric]]],
            Dict[str, Dict[str, Union[torchmetrics.Metric, List[torchmetrics.Metric]]]],
        ],
    ):
        if not isinstance(metrics, dict):
            raise ValueError(
                f"{__class__.__name__} requires a dictionary of metrics, got {type(metrics)}"
            )
        for v in metrics.values():
            if not isinstance(v, (torchmetrics.Metric, list, dict)):
                raise ValueError(
                    f"The values in `metrics` can be either a torchmetrics.Metric or a "
                    f"list or dict of torchmetrics.Metric, got {type(v)}."
                )
            if isinstance(v, dict):
                self._depth = 2
                for e in v.values():
                    if isinstance(e, list):
                        if any(not isinstance(e_, torchmetrics.Metric) for e_ in e):
                            raise ValueError(
                                f"Found {e} in a list of metrics. All metrics in a group "
                                f"must be of type torchmetrics.Metric"
                            )
                    elif isinstance(e, dict):
                        raise ValueError(
                            f"{__class__.__name__} does not support nested dictionaries "
                            f"of metrics for more than 2 levels of depth."
                        )
                    elif not isinstance(e, torchmetrics.Metric):
                        raise ValueError(
                            f"Found {e} in metrics. All metrics in a group must be of type "
                            f"torchmetrics.Metric, got {type(e)}."
                        )
            elif isinstance(v, list):
                self._depth = 1
                if any(not isinstance(e, torchmetrics.Metric) for e in v):
                    raise ValueError(
                        f"Found {v} in a list of metrics. All metrics in a group "
                        f"must be of type torchmetrics.Metric, got {type(v)}."
                    )
            elif not isinstance(v, torchmetrics.Metric):
                self._depth = 1
                raise ValueError(
                    f"Found {v} in metrics. All metrics in a group must be of type "
                    f"torchmetrics.Metric, got {type(v)}."
                )

        self.metrics = metrics

        self._sequence_index = None
        self._smart_flushing = False
        self._init_cache()

    def _init_cache(self):
        if self._smart_flushing:
            num_sequences = self._sequence_index.max().item() + 1

            self._sample_ptr = 0

            if self._depth == 1:
                self._cache = [
                    {"target": [], "pred": [], "timestamps": []}
                    for _ in range(num_sequences)
                ]

            elif self._depth == 2:
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
                self._sequence_index, return_counts=True
            )
        else:
            if self._depth == 1:
                self._cache = {
                    recording_id: {"target": [], "pred": [], "timestamps": []}
                    for recording_id in self.metrics.keys()
                }
            elif self._depth == 2:
                self._cache = {
                    recording_id: {
                        "target": defaultdict(list),
                        "pred": defaultdict(list),
                        "timestamps": defaultdict(list),
                    }
                    for recording_id in self.metrics.keys()
                }

    def update(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        timestamps: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recording_id: str,
    ):
        if self._depth == 1:
            assert isinstance(preds, torch.Tensor)

            # get ref to the cache
            if self._smart_flushing:
                j = self._sequence_index[self._sample_ptr]
                _cache = self._cache[j]
            else:
                _cache = self._cache[recording_id]

            _cache["pred"].append(preds)
            _cache["target"].append(targets)
            _cache["timestamps"].append(timestamps)

            if self._smart_flushing:
                # flag that the cache has been updated
                self._counter[j] += 1
                self._sample_ptr += 1
                # check if the cache can be flushed
                if self._counter[j] >= self._cache_flush_threshold[j]:
                    self._smart_flush_cache(j, recording_id=recording_id)

        elif self._depth == 2:
            # get ref to the cache
            if self._smart_flushing:
                j = self._sequence_index[self._sample_ptr]
                _cache = self._cache[j]
            else:
                _cache = self._cache[recording_id]

            for task_name in preds.keys():
                _cache["pred"][task_name].append(preds[task_name])
                _cache["target"][task_name].append(targets[task_name])
                _cache["timestamps"][task_name].append(timestamps[task_name])

            if self._smart_flushing:
                # flag that the cache has been updated
                self._counter[j] += 1
                self._sample_ptr += 1
                # check if the cache can be flushed
                if self._counter[j] >= self._cache_flush_threshold[j]:
                    self._smart_flush_cache(j, recording_id=recording_id)

    def compute(self, device: str = "cpu"):
        if self._smart_flushing:
            for i, cache in enumerate(self._cache):
                if cache is not None:
                    raise RuntimeError(
                        f"Cache at index {i} was not flushed before end of validation epoch. "
                        "This likely indicates a bug in the cache flushing logic."
                    )
        else:
            self._flush_cache()

        metric_dict = {}

        if self._depth == 2:
            for recording_id in self.metrics.keys():
                for task_name in self.metrics[recording_id].keys():
                    metrics = self.metrics[recording_id][task_name]
                    if isinstance(metrics, torchmetrics.Metric):
                        metrics = [metrics]
                    for metric in metrics:
                        metric_name = str(metric)
                        metric_dict[f"{recording_id}/{task_name}/{metric_name}"] = (
                            metric.to(device).compute().item()
                        )
                        metric.to("cpu")
        elif self._depth == 1:
            for recording_id in self.metrics.keys():
                metrics = self.metrics[recording_id]
                if isinstance(metrics, torchmetrics.Metric):
                    metrics = [metrics]
                for metric in metrics:
                    metric_name = str(metric)
                    metric_dict[f"{recording_id}/{metric_name}"] = (
                        metric.to(device).compute().item()
                    )
                    metric.to("cpu")

        return metric_dict

    def reset(self):
        self._init_cache()

        # reset the metrics
        def _reset_metrics(metrics):
            if isinstance(metrics, dict):
                for v in metrics.values():
                    _reset_metrics(v)
            elif isinstance(metrics, list):
                for m in metrics:
                    _reset_metrics(m)
            else:
                metrics.reset()
                metrics.to("cpu")

        _reset_metrics(self.metrics)

    def convert_to_stitcher_sampler(self, sampler, num_replicas=None, rank=None):
        stitch_sampler = StitcherSamplerWrapper(
            sampler, num_replicas=num_replicas, rank=rank
        )
        self._sequence_index = stitch_sampler.sequence_index
        # enable smart flushing
        self._smart_flushing = True
        self._init_cache()
        return stitch_sampler

    def _smart_flush_cache(self, i: int, recording_id: str):
        if self._depth == 1:
            preds = torch.cat(self._cache[i]["pred"])
            targets = torch.cat(self._cache[i]["target"])
            timestamps = torch.cat(self._cache[i]["timestamps"])

            # Pool data wherever timestamps overlap
            stitched_pred = stitch(timestamps, preds)
            stitched_target = stitch(timestamps, targets)

            if targets.dtype == torch.long:
                stitched_target = torch.round(stitched_target).long()

            metrics = self.metrics[recording_id]
            if isinstance(metrics, torchmetrics.Metric):
                metrics = [metrics]

            for metric in metrics:
                metric.update(stitched_pred, stitched_target)

        elif self._depth == 2:
            for task_name in self._cache[i]["pred"].keys():
                preds = torch.cat(self._cache[i]["pred"][task_name])
                targets = torch.cat(self._cache[i]["target"][task_name])
                timestamps = torch.cat(self._cache[i]["timestamps"][task_name])

                # Pool data wherever timestamps overlap
                stitched_pred = stitch(timestamps, preds)
                stitched_target = stitch(timestamps, targets)

                if targets.dtype == torch.long:
                    stitched_target = torch.round(stitched_target).long()

                metrics = self.metrics[recording_id][task_name]
                if isinstance(metrics, torchmetrics.Metric):
                    metrics = [metrics]

                for metric in metrics:
                    metric.update(stitched_pred, stitched_target)

        # delete the cache to free memory
        self._cache[i] = None

    def _flush_cache(self):
        for recording_id in self._cache.keys():
            _cache = self._cache[recording_id]
            if self._depth == 1:
                preds = torch.cat(_cache["pred"])
                targets = torch.cat(_cache["target"])
                timestamps = torch.cat(_cache["timestamps"])

                # Pool data wherever timestamps overlap
                stitched_pred = stitch(timestamps, preds)
                stitched_target = stitch(timestamps, targets)

                if targets.dtype == torch.long:
                    stitched_target = torch.round(stitched_target).long()

                metrics = self.metrics[recording_id]
                if isinstance(metrics, torchmetrics.Metric):
                    metrics = [metrics]

                for metric in metrics:
                    metric.update(stitched_pred, stitched_target)

            elif self._depth == 2:
                for task_name in _cache["target"].keys():
                    preds = torch.cat(_cache["pred"][task_name])
                    targets = torch.cat(_cache["target"][task_name])
                    timestamps = torch.cat(_cache["timestamps"][task_name])

                    # Pool data wherever timestamps overlap
                    stitched_pred = stitch(timestamps, preds)
                    stitched_target = stitch(timestamps, targets)

                    if targets.dtype == torch.long:
                        stitched_target = torch.round(stitched_target).long()

                    metrics = self.metrics[recording_id][task_name]
                    if isinstance(metrics, torchmetrics.Metric):
                        metrics = [metrics]

                    for metric in metrics:
                        metric.update(stitched_pred, stitched_target)

        # delete the cache to free memory
        self._cache = None
