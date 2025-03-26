import argparse
from typing import Any, Dict, List, Optional, Tuple
from rich.table import Table
from rich.console import Console
from rich import print
from collections import defaultdict

from torch_brain.data import Dataset
import numpy as np
from tqdm import tqdm


def aggregate_mean_std(
    mean: np.ndarray, std: np.ndarray, sample_count: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the overall mean and standard deviation from group-wise statistics.

    This implements the formula for pooled standard deviation that accounts for both
    within-group variance and between-group variance:

    $$\sigma_{\text{overall}} = \sqrt{\frac{\sum_{i=1}^{k} (n_i - 1) \sigma_i^2 + \sum_{i=1}^{k} n_i(\bar{x}_i - \bar{x}_{\text{overall}})^2}{N - 1}}$$

    Where:
    - $k$ is the number of groups (recordings)
    - $n_i$ is the sample count of group $i$
    - $\sigma_i$ is the standard deviation of group $i$
    - $\bar{x}_i$ is the mean of group $i$
    - $\bar{x}_{\text{overall}}$ is the overall mean across all groups
    - $N$ is the total sample count across all groups

    Args:
        mean: Array of means for each group, shape (n_groups, n_features)
        std: Array of standard deviations for each group, shape (n_groups, n_features)
        sample_count: Array of sample counts for each group, shape (n_groups,)

    Returns:
        Tuple of (overall_mean, overall_std)
    """
    total_sample_count = sample_count.sum()

    # aggregate mean by taking into account the number of samples
    aggregated_mean = (sample_count * mean).sum(axis=0) / total_sample_count

    # aggregate the standard deviation
    # step 1 : weighted sum of groupwise variances
    group_variances = (sample_count - 1) * std**2
    # step 2 : weighted, groupwise sum of squares of means after subtracting overall mean
    mean_differences = sample_count * (mean - aggregated_mean) ** 2
    # step 3: combine both variances
    pooled_variance = (group_variances.sum(axis=0) + mean_differences.sum(axis=0)) / (
        total_sample_count - 1
    )
    # step 4: take square root
    aggregated_std = np.sqrt(pooled_variance)
    return aggregated_mean, aggregated_std


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./processed",
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        nargs="+",
        help="List of attributes to process",
        required=True,
    )

    args = parser.parse_args()

    # We instantiate a train_dataset object
    # as the object contains useful methods for obtaining the data
    # needed to calculate the zscales
    dataset = Dataset(
        args.data_root,
        "train",
        config=args.config,
        transform=None,
    )

    mean_std_dict = defaultdict(lambda: defaultdict(list))
    for recording_id in tqdm(dataset.get_recording_ids(), desc="Processing recordings"):
        recording_data = dataset.get_recording_data(recording_id)
        for attribute in args.attributes:
            values = recording_data.get_nested_attribute(attribute)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            n = len(values)

            mean_std_dict[attribute]["mean"].append(mean)
            mean_std_dict[attribute]["std"].append(std)
            mean_std_dict[attribute]["count"].append(n)

    aggregated_mean_std = {}
    for attribute in mean_std_dict.keys():
        mean_std_dict[attribute]["mean"] = np.array(
            mean_std_dict[attribute]["mean"]
        ).squeeze()
        mean_std_dict[attribute]["std"] = np.array(
            mean_std_dict[attribute]["std"]
        ).squeeze()
        mean_std_dict[attribute]["count"] = np.array(
            mean_std_dict[attribute]["count"]
        ).squeeze()
        aggregated_mean_std[attribute] = aggregate_mean_std(**mean_std_dict[attribute])

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("decoder_id")
    table.add_column("mean")
    table.add_column("std")
    for attribute, (mean, std) in aggregated_mean_std.items():
        if isinstance(mean, np.ndarray):
            if mean.shape != (2,):
                raise ValueError(f"Expected last dimension to be 2, got {mean.shape}")
            table.add_row(f"{attribute}.x", f"{mean[0]:.8f}", f"{std[0]:.8f}")
            table.add_row(f"{attribute}.y", f"{mean[1]:.8f}", f"{std[1]:.8f}")
        else:
            table.add_row(attribute, f"{mean:.8f}", f"{std:.8f}")
    console.print(table)
    print("[green] Done calculating mean, std for all continous outputs")
    print(
        "[yellow] Manually copy the zscales for each decoder_id into the dataset config file"
    )


if __name__ == "__main__":
    main()
