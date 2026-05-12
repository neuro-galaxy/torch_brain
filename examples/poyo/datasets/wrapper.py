import torch
from torch_brain.data import pad8
from torch_brain.utils import (
    resolve_weights_based_on_interval_membership,
    isin_interval,
)
from typing import Callable
import numpy as np
from temporaldata import Data
from torch_brain.dataset import DatasetIndex, Dataset


class POYODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Callable):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return self.dataset.__getattribute__(key)

    def __getitem__(self, index: DatasetIndex):
        data = self.dataset[index]

        X = self.tokenizer(data)

        timestamps, values, weights, eval_mask = prepare_for_readout(data)
        Y = dict(
            timestamps=pad8(timestamps),
            values=pad8(values),
            weights=pad8(weights),
            output_mask=pad8(np.ones(len(timestamps), dtype=bool)),
            eval_mask=pad8(eval_mask),
            session_id=data.session.id,
            absolute_start=data.absolute_start,
        )

        return X, Y


def prepare_for_readout(data: Data):
    readout_config = data.config["readout"]

    value_key = readout_config["value_key"]
    timestamp_key = readout_config["timestamp_key"]

    timestamps = data.get_nested_attribute(timestamp_key)
    values = data.get_nested_attribute(value_key)

    mean = readout_config["normalize_mean"]
    std = readout_config["normalize_std"]
    values = (values - mean) / std

    if values.dtype == np.float64:
        values = values.astype(np.float32)

    # resolve weights
    weights = resolve_weights_based_on_interval_membership(
        timestamps, data, config=readout_config.get("weights", None)
    )

    # resolve eval mask
    eval_mask = np.ones(len(timestamps), dtype=np.bool_)
    eval_interval_key = readout_config.get("eval_interval", None)
    if eval_interval_key is not None:
        eval_interval = data.get_nested_attribute(eval_interval_key)
        eval_mask = isin_interval(timestamps, eval_interval)

    return timestamps, values, weights, eval_mask
