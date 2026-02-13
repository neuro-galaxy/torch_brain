import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
from temporaldata import Interval, Data


def split_one_epoch(
    epoch: Interval,
    grid: Interval,
    split_ratios: Optional[List[float]] = None,
) -> Tuple[Interval, Interval, Interval]:
    """Split a single epoch into train, validation, and test intervals.

    Args:
        epoch: The full time interval to split (must contain a single interval)
        grid: Grid intervals used to align split boundaries
        split_ratios: List of three ratios [train_ratio, valid_ratio, test_ratio]
            that sum to 1.0. Defaults to [0.6, 0.1, 0.3].

    Returns:
        Tuple of (train_interval, valid_interval, test_interval)

    Raises:
        ValueError:
            if split_ratios is not a sequence of exactly three numbers,
            if any ratio is negative,
            if split_ratios do not sum to 1, or
            if the epoch does not contain a single interval
    """
    if split_ratios is None:
        split_ratios = [0.6, 0.1, 0.3]

    if not hasattr(split_ratios, "__len__") or len(split_ratios) != 3:
        raise ValueError(
            "split_ratios must be a sequence of three numbers (train, valid, test)"
        )

    if any(r < 0 for r in split_ratios):
        raise ValueError("split_ratios elements must be non-negative")

    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")

    if len(epoch) != 1:
        raise ValueError("Epoch must contain a single interval")
    epoch_start = epoch.start[0]
    epoch_end = epoch.end[0]

    train_val_split_time = epoch_start + split_ratios[0] * (epoch_end - epoch_start)
    val_test_split_time = train_val_split_time + split_ratios[1] * (
        epoch_end - epoch_start
    )

    grid_match = grid.slice(
        train_val_split_time, train_val_split_time, reset_origin=False
    )
    if len(grid_match) > 0:
        if (
            train_val_split_time - grid_match.start[0]
            > grid_match.end[0] - train_val_split_time
        ):
            train_val_split_time = grid_match.end[0]
        else:
            train_val_split_time = grid_match.start[0]

    grid_match = grid.slice(
        val_test_split_time, val_test_split_time, reset_origin=False
    )
    if len(grid_match) > 0:
        if (
            val_test_split_time - grid_match.start[0]
            > grid_match.end[0] - val_test_split_time
        ):
            val_test_split_time = grid_match.end[0]
        else:
            val_test_split_time = grid_match.start[0]

    train_interval = Interval(
        start=np.array([epoch_start]), end=np.array([train_val_split_time])
    )
    val_interval = Interval(
        start=train_interval.end[0:1], end=np.array([val_test_split_time])
    )
    test_interval = Interval(start=val_interval.end[0:1], end=np.array([epoch_end]))

    return train_interval, val_interval, test_interval


def split_two_epochs(
    epoch: Interval,
    grid: Interval,
) -> Tuple[Interval, Interval, Interval]:
    assert len(epoch) == 2
    first_epoch_start = epoch.start[0]
    first_epoch_end = epoch.end[0]

    split_time = first_epoch_start + 0.5 * (first_epoch_end - first_epoch_start)
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval = Interval(
        start=np.array([first_epoch_start]),
        end=np.array([split_time]),
    )
    val_interval = Interval(
        start=train_interval.end[0:1], end=np.array([first_epoch_end])
    )
    test_interval = epoch.select_by_mask(np.array([False, True]))

    return train_interval, val_interval, test_interval


def split_three_epochs(
    epoch: Interval, grid: Interval
) -> Tuple[Interval, Interval, Interval]:
    assert len(epoch) == 3

    test_interval = epoch.select_by_mask(np.array([False, False, True]))
    train_interval = epoch.select_by_mask(np.array([True, True, False]))

    split_time = train_interval.end[1] - 0.3 * (
        train_interval.end[1] - train_interval.start[1]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[1] = split_time
    val_interval = Interval(start=train_interval.end[1:2], end=epoch.end[1:2])

    return train_interval, val_interval, test_interval


def split_four_epochs(
    epoch: Interval, grid: Interval
) -> Tuple[Interval, Interval, Interval]:
    assert len(epoch) == 4

    test_interval = epoch.select_by_mask(np.array([False, False, False, True]))
    train_interval = epoch.select_by_mask(np.array([True, True, True, False]))
    split_time = train_interval.end[2] - 0.5 * (
        train_interval.end[2] - train_interval.start[2]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[2] = split_time
    val_interval = Interval(start=train_interval.end[2:3], end=epoch.end[2:3])

    return train_interval, val_interval, test_interval


def split_five_epochs(
    epoch: Interval, grid: Interval
) -> Tuple[Interval, Interval, Interval]:
    assert len(epoch) == 5

    train_interval = epoch.select_by_mask(np.array([True, True, True, False, False]))
    test_interval = epoch.select_by_mask(np.array([False, False, False, True, True]))

    split_time = train_interval.end[2] - 0.5 * (
        train_interval.end[2] - train_interval.start[2]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[2] = split_time
    val_interval = Interval(start=train_interval.end[2:3], end=epoch.end[2:3])

    return train_interval, val_interval, test_interval


def split_more_than_five_epochs(
    epoch: Interval,
) -> Tuple[Interval, Interval, Interval]:
    assert len(epoch) > 5

    train_interval, val_interval, test_interval = epoch.split(
        [0.6, 0.1, 0.3], shuffle=False
    )
    return train_interval, val_interval, test_interval


def generate_train_valid_test_splits(
    epoch_dict: Dict[str, Interval], grid: Interval
) -> Tuple[Interval, Interval, Interval]:
    train_intervals = Interval(np.array([]), np.array([]))
    valid_intervals = Interval(np.array([]), np.array([]))
    test_intervals = Interval(np.array([]), np.array([]))

    for name, epoch in epoch_dict.items():
        if name == "invalid_presentation_epochs":
            warnings.warn(
                "Found invalid presentation epochs, which will be excluded.",
                stacklevel=2,
            )
            continue
        if len(epoch) == 1:
            train, valid, test = split_one_epoch(epoch, grid)
        elif len(epoch) == 2:
            train, valid, test = split_two_epochs(epoch, grid)
        elif len(epoch) == 3:
            train, valid, test = split_three_epochs(epoch, grid)
        elif len(epoch) == 4:
            train, valid, test = split_four_epochs(epoch, grid)
        elif len(epoch) == 5:
            train, valid, test = split_five_epochs(epoch, grid)
        else:
            train, valid, test = split_more_than_five_epochs(epoch)

        train_intervals = train_intervals | train
        valid_intervals = valid_intervals | valid
        test_intervals = test_intervals | test

    return train_intervals, valid_intervals, test_intervals


def chop_intervals(
    intervals: Interval, duration: float, check_no_overlap: bool = False
) -> Interval:
    """
    Subdivides intervals into fixed-length epochs using Interval.arange().

    If some intervals are shorter than the duration, keep them as they are.
    If an interval is not a perfect multiple of the duration, the last chunk will be shorter.

    Args:
        intervals: The original intervals to chop.
        duration: The duration of each chopped interval in seconds.
        check_no_overlap: If True, verify the resulting intervals don't overlap.

    Returns:
        Interval: A new Interval object containing the chopped segments.
                  Metadata from the original intervals is preserved and repeated for each segment.

    Raises:
        ValueError: If check_no_overlap is True and intervals overlap.
    """
    if len(intervals) == 0:
        return Interval(start=np.array([]), end=np.array([]))

    chopped_intervals = []
    original_indices = []

    for i, (start, end) in enumerate(zip(intervals.start, intervals.end)):
        if end - start <= duration:
            chopped = Interval(start=np.array([start]), end=np.array([end]))
        else:
            chopped = Interval.arange(start, end, step=duration, include_end=True)

        chopped_intervals.append(chopped)
        original_indices.extend([i] * len(chopped))

    all_starts = np.concatenate([c.start for c in chopped_intervals])
    all_ends = np.concatenate([c.end for c in chopped_intervals])

    kwargs = {}
    if hasattr(intervals, "keys"):
        for key in intervals.keys():
            if key in ["start", "end"]:
                continue
            val = getattr(intervals, key)
            if isinstance(val, np.ndarray) and len(val) == len(intervals):
                kwargs[key] = val[original_indices]

    result = Interval(start=all_starts, end=all_ends, **kwargs)

    if check_no_overlap:
        if not result.is_disjoint():
            raise ValueError("Intervals overlap after chopping")

    return result


def _create_interval_split(intervals: Interval, indices: np.ndarray) -> Interval:
    """Create an Interval subset from indices and sort it."""
    mask = np.zeros(len(intervals), dtype=bool)
    mask[indices] = True
    split = intervals.select_by_mask(mask)
    split.sort()
    return split


def generate_stratified_folds(
    intervals: Interval,
    stratify_by: str,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Data]:
    """
    Generates stratified train/valid/test splits using a two-stage splitting process.

    The splitting is performed in two stages:
        1. Outer split (StratifiedKFold): The intervals are divided into n_folds,
           where each fold uses one partition as the test set and the remaining
           partitions as train+valid. Stratification ensures each fold maintains
           the class distribution of the original data.
        2. Inner split (StratifiedShuffleSplit): The train+valid portion of each fold
           is further split into train and valid sets using val_ratio, while preserving
           the class distribution.

    Args:
        intervals: The intervals to split.
        n_folds: Number of folds for cross-validation.
        val_ratio: Ratio of validation set relative to train+valid combined.
        seed: Random seed.
        stratify_by: The attribute name to use for stratification (e.g., "id", "label",
            "class"). The intervals must have this attribute.

    Returns:
        List of Data objects, one for each fold.

    Raises:
        ValueError: If the intervals don't have the specified stratify_by attribute.
        ValueError: If there are fewer samples than n_folds.
    """
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "This function requires the scikit-learn library which you can install with "
            "`pip install scikit-learn`"
        )

    if not hasattr(intervals, stratify_by):
        raise ValueError(
            f"Intervals must have a '{stratify_by}' attribute for stratification."
        )

    class_labels = getattr(intervals, stratify_by)
    if len(class_labels) < n_folds:
        raise ValueError(
            f"Not enough samples ({len(class_labels)}) for {n_folds} folds."
        )

    outer_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    sample_indices = np.arange(len(intervals))

    for fold_idx, (train_val_indices, test_indices) in enumerate(
        outer_splitter.split(sample_indices, class_labels)
    ):
        test_split = _create_interval_split(intervals, test_indices)

        train_val_labels = class_labels[train_val_indices]
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=seed + fold_idx
        )

        for train_indices, val_indices in inner_splitter.split(
            train_val_indices, train_val_labels
        ):
            train_original_indices = train_val_indices[train_indices]
            val_original_indices = train_val_indices[val_indices]

            train_split = _create_interval_split(intervals, train_original_indices)
            val_split = _create_interval_split(intervals, val_original_indices)

            combined_domain = train_split | val_split | test_split

            fold_data = Data(
                train=train_split,
                valid=val_split,
                test=test_split,
                domain=combined_domain,
            )

            folds.append(fold_data)

    return folds


def generate_train_valid_splits_one_epoch(
    epoch: Interval, split_ratios: Optional[List[float]] = None
) -> Tuple[Interval, Interval]:
    """Split a single time interval into training and validation intervals.

    Args:
        epoch: The full time interval to split (must contain a single interval)
        split_ratios: List of two ratios [train_ratio, valid_ratio] that sum to 1.0.
            Defaults to [0.9, 0.1].

    Returns:
        Tuple of (train_intervals, valid_intervals)

    Raises:
        ValueError:
            if split_ratios is not a sequence of exactly two numbers,
            if any ratio is negative,
            if split_ratios do not sum to 1, or
            if the epoch does not contain a single interval
    """
    if split_ratios is None:
        split_ratios = [0.9, 0.1]

    if not hasattr(split_ratios, "__len__") or len(split_ratios) != 2:
        raise ValueError(
            "split_ratios must be a sequence of two numbers (train, valid)"
        )

    if any(r < 0 for r in split_ratios):
        raise ValueError("split_ratios elements must be non-negative")

    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")

    if len(epoch) != 1:
        raise ValueError("Epoch must contain a single interval")

    epoch_start = epoch.start[0]
    epoch_end = epoch.end[0]

    train_split_time = epoch_start + split_ratios[0] * (epoch_end - epoch_start)
    val_split_time = train_split_time + split_ratios[1] * (epoch_end - epoch_start)

    train_intervals = Interval(
        start=epoch_start,
        end=train_split_time,
    )
    valid_intervals = Interval(
        start=train_intervals.end[0],
        end=val_split_time,
    )

    return train_intervals, valid_intervals
