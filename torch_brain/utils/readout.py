from typing import TYPE_CHECKING
import numpy as np
from temporaldata import Data

import torch_brain
from torch_brain.utils import (
    resolve_weights_based_on_interval_membership,
    isin_interval,
)

if TYPE_CHECKING:
    from torch_brain.registry import ModalitySpec


def prepare_for_readout(
    data: Data,
    readout_spec: "ModalitySpec",
):
    required_keys = ["readout_id"]
    optional_keys = [
        "weights",
        "normalize_mean",
        "normalize_std",
        "timestamp_key",
        "value_key",
        "metrics",
        "eval_interval",
    ]

    readout_config = data.config["readout"]

    # check that the readout config contains all required keys
    for key in required_keys:
        if key not in readout_config:
            raise ValueError(f"readout config is missing required key: {key}")

    # check that the readout config contains only valid keys
    if not all(key in required_keys + optional_keys for key in readout_config.keys()):
        raise ValueError(
            f"Readout {readout_config} contains invalid keys, please use only {required_keys + optional_keys}"
        )

    key = readout_config["readout_id"]

    if key not in torch_brain.MODALITY_REGISTRY:
        raise ValueError(
            f"Readout {key} not found in modality registry, please register it "
            "using torch_brain.register_modality()"
        )

    value_key = readout_config.get("value_key", readout_spec.value_key)
    timestamp_key = readout_config.get("timestamp_key", readout_spec.timestamp_key)

    timestamps = data.get_nested_attribute(timestamp_key)
    values = data.get_nested_attribute(value_key)

    # z-scale the values if mean/std are specified in the config file
    if "normalize_mean" in readout_config:
        # if mean is a list, its a per-channel mean (usually for x,y coordinates)
        if isinstance(readout_config["normalize_mean"], list):
            mean = np.array(readout_config["normalize_mean"])
        else:
            mean = readout_config["normalize_mean"]
        values = values - mean
    if "normalize_std" in readout_config:
        # if std is a list, its a per-channel std (usually for x,y coordinates)
        if isinstance(readout_config["normalize_std"], list):
            std = np.array(readout_config["normalize_std"])
        else:
            std = readout_config["normalize_std"]
        values = values / std

    # here we assume that we won't be running a model at float64 precision
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
