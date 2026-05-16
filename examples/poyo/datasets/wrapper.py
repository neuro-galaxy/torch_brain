from typing import Callable
from dataclasses import dataclass
import numpy as np
import torch

from temporaldata import Data, Interval
from torch_brain.data.collate import pad8
from torch_brain.utils import isin_interval
from torch_brain.dataset import DatasetIndex, Dataset


@dataclass(frozen=True)
class PoyoReadoutConfig:
    timestamp_key: str
    value_key: str
    eval_interval: str | None = None
    normalize_mean: float = 0.0
    normalize_std: float = 1.0
    weights: dict[str, float] | None = None


class PoyoDatasetWrapper(torch.utils.data.Dataset):
    """Adapts a :class:`Dataset` for POYO training by producing ``(X, Y)`` pairs.

    Requires the wrapped dataset to emit :obj:`Data` objects with a
    ``readout_config`` attribute of type :class:`PoyoReadoutConfig`
    (typically attached in ``get_recording_hook``).

    For each sample, ``X`` is built by running ``tokenizer`` on the raw
    :obj:`Data` window (typically ``model.tokenize``), and ``Y`` specifies
    the (normalized) regression target.

    The wrapper forwards attribute access to the underlying dataset, so it
    behaves like a :class:`Dataset` to samplers and the rest of the pipeline.
    """

    def __init__(self, dataset: Dataset, tokenizer: Callable):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return self.dataset.__getattribute__(key)

    def __getitem__(self, index: DatasetIndex):
        data = self.dataset[index]

        # Prepare encoder input
        X = self.tokenizer(data)

        # Prepare target (also part of decoder query)
        cfg = data.readout_config
        if not isinstance(cfg, PoyoReadoutConfig):
            raise TypeError(
                "Expected data.readout_config to be a PoyoReadoutConfig, "
                f"got {type(cfg).__name__}. Set data.readout_config on the "
                "returned Data object, typically in get_recording_hook."
            )

        timestamps = data.get_nested_attribute(cfg.timestamp_key)
        values = data.get_nested_attribute(cfg.value_key)

        mean = cfg.normalize_mean
        std = cfg.normalize_std
        values = (values - mean) / std

        weights = _resolve_weights(timestamps, data, weight_cfg=cfg.weights)

        # resolve eval mask
        eval_mask = np.ones(len(timestamps), dtype=bool)
        if cfg.eval_interval is not None:
            eval_interval = data.get_nested_attribute(cfg.eval_interval)
            eval_mask = isin_interval(timestamps, eval_interval)

        Y = dict(
            timestamps=pad8(timestamps.astype(np.float32)),
            values=pad8(values.astype(np.float32)),
            weights=pad8(weights),
            output_mask=pad8(np.ones(len(timestamps), dtype=bool)),
            eval_mask=pad8(eval_mask),
            session_id=data.session.id,
            absolute_start=data.absolute_start,
        )

        return X, Y


def _resolve_weights(timestamps, data, weight_cfg):
    weights = np.ones_like(timestamps, dtype=np.float32)
    if weight_cfg is None:
        return weights

    for weight_key, weight_value in weight_cfg.items():
        # extract the interval from the weight key
        weight = data.get_nested_attribute(weight_key)
        if not isinstance(weight, Interval):
            raise ValueError(
                f"Weight {weight_key} is of type {type(weight)}. "
                "Expected an Interval object."
            )
        weights[isin_interval(timestamps, weight)] *= weight_value
    return weights
