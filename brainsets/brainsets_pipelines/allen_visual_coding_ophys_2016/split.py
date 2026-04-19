import logging
import numpy as np
from temporaldata import Interval


def split_one_epoch(
    epoch: Interval,
    grid: Interval,
    split_ratios: list[float] | None = None,
) -> tuple[Interval, Interval, Interval]:
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
) -> tuple[Interval, Interval, Interval]:
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
) -> tuple[Interval, Interval, Interval]:
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
) -> tuple[Interval, Interval, Interval]:
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
) -> tuple[Interval, Interval, Interval]:
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
) -> tuple[Interval, Interval, Interval]:
    assert len(epoch) > 5

    train_interval, val_interval, test_interval = epoch.split(
        [0.6, 0.1, 0.3], shuffle=False
    )
    return train_interval, val_interval, test_interval


def generate_train_valid_test_splits(
    epoch_dict: dict[str, Interval], grid: Interval
) -> tuple[Interval, Interval, Interval]:
    train_intervals = Interval(np.array([]), np.array([]))
    valid_intervals = Interval(np.array([]), np.array([]))
    test_intervals = Interval(np.array([]), np.array([]))

    for name, epoch in epoch_dict.items():
        if name == "invalid_presentation_epochs":
            logging.warning(
                "Found invalid presentation epochs, which will be excluded."
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
