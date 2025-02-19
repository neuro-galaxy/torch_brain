"""
Script that ingests a dataset config file, calculates the mean, std for all continous outputs across all sessions (respecting only the training interval splits).
1. Gets the sessions, training intervals to be used from the dataset config file and dataset instance.
3. Loads the data for each session.
3. Looks at the various "decoder_ids" used in the "multitask_readout" section of the config file, and for each continuous decoder_id, aggregates the mean, std, and count of the tensors.
4. Uses all the aggregated means, stds, and counts of all the sessions includes, to calculate the overall mean and std.
5. The above is done for each decoder_id (that is continous, i.e. not categorical)
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple
from rich.table import Table
from rich.console import Console
from rich import print

from torch_brain.data import Dataset
import numpy as np
from omegaconf import OmegaConf
from brainsets_utils.taxonomy.multitask_readout import OutputType, decoder_registry
from tqdm import tqdm


def calculate_overall_mean_std(means, stds, counts):
    total_count = counts.sum()
    if means.ndim > 1 and counts.ndim == 1:
        counts = np.expand_dims(counts, axis=-1)
    # Calculate the overall mean
    overall_mean = (counts * means).sum(axis=0) / total_count

    # Calculate the overall standard deviation
    # step 1 : weighted sum of groupwise variances
    group_variances = (counts - 1) * stds**2
    # step 2 : weighted, groupwise sum of squares of means after subtracting overall mean
    mean_differences = counts * (means - overall_mean) ** 2
    # step 3: combine both variances
    pooled_variance = (group_variances.sum(axis=0) + mean_differences.sum(axis=0)) / (
        total_count - 1
    )
    # step 4: take square root
    overall_std = np.sqrt(pooled_variance)
    return overall_mean, overall_std


def calculate_zscales(dataset: Dataset) -> Dict[str, Tuple[float, float]]:
    """
    This helper looks at all the tasks that have continous outputs (if any),
    loads the values and calculates the mean and std for each output across all sessions of this split,
    respecting the sampling intervals.
    """

    chunk_metrics = {}
    print("[blue] calculating normalization scales")
    for session_id in tqdm(dataset.session_ids):
        task_readouts = dataset.session_dict[session_id]["config"]["multitask_readout"]
        # get a data object that is sliced according to the training sample intervals
        this_session_data = dataset.get_session_data(session_id)
        for task_readout in task_readouts:
            task_id = task_readout["decoder_id"]
            decoder = decoder_registry[task_id]
            if decoder.type == OutputType.CONTINUOUS:
                values = this_session_data.get_nested_attribute(decoder.value_key)
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                n = len(values)

                if task_id not in chunk_metrics:
                    chunk_metrics[task_id] = {
                        "means": [mean],
                        "stds": [std],
                        "counts": [n],
                    }
                else:
                    chunk_metrics[task_id]["means"].append(mean)
                    chunk_metrics[task_id]["stds"].append(std)
                    chunk_metrics[task_id]["counts"].append(n)

    results = {}
    for task_id, metrics in chunk_metrics.items():

        metrics["means"] = np.array(metrics["means"]).squeeze()
        metrics["stds"] = np.array(metrics["stds"]).squeeze()
        metrics["counts"] = np.array(metrics["counts"]).squeeze()
        results[task_id] = calculate_overall_mean_std(**metrics)
    return results


def main(args):
    dataset_config = OmegaConf.load(args.dataset_config)

    # We instantiate a train_dataset object
    # as the object contains useful methods for obtaining the data
    # needed to calculate the zscales
    train_dataset = Dataset(
        args.data_root,
        "train",
        include=dataset_config,
        transform=None,
    )

    zscales = calculate_zscales(train_dataset)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("decoder_id")
    table.add_column("mean")
    table.add_column("std")
    for decoder_id, (mean, std) in zscales.items():
        if isinstance(mean, np.ndarray):
            if mean.shape != (2,):
                raise ValueError(f"Expected last dimension to be 2, got {mean.shape}")
            table.add_row(f"{decoder_id}.x", f"{mean[0]:.8f}", f"{std[0]:.8f}")
            table.add_row(f"{decoder_id}.y", f"{mean[1]:.8f}", f"{std[1]:.8f}")
        else:
            table.add_row(decoder_id, f"{mean:.8f}", f"{std:.8f}")
    console.print(table)
    print("[green] Done calculating mean, std for all continous outputs")
    print(
        "[yellow] Manually copy the zscales for each decoder_id into the dataset config file"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./processed",
        help="Path to the dataset config file",
    )
    args = parser.parse_args()
    main(args)
