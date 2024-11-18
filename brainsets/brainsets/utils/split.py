import logging
import numpy as np
from temporaldata import Interval


def split_one_epoch(epoch, grid, split_ratios=[0.6, 0.1, 0.3]):
    assert len(epoch) == 1
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

    train_interval = Interval(start=epoch_start, end=train_val_split_time)
    val_interval = Interval(start=train_interval.end[0], end=val_test_split_time)
    test_interval = Interval(start=val_interval.end[0], end=epoch_end)

    return train_interval, val_interval, test_interval


def split_two_epochs(epoch, grid):
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
        start=first_epoch_start,
        end=split_time,
    )
    val_interval = Interval(start=train_interval.end[0], end=first_epoch_end)
    test_interval = epoch.select_by_mask(np.array([False, True]))

    return train_interval, val_interval, test_interval


def split_three_epochs(epoch, grid):
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
    val_interval = Interval(start=train_interval.end[1], end=epoch.end[1])

    return train_interval, val_interval, test_interval


def split_four_epochs(epoch, grid):
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
    val_interval = Interval(start=train_interval.end[2], end=epoch.end[2])

    return train_interval, val_interval, test_interval


def split_five_epochs(epoch, grid):
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
    val_interval = Interval(start=train_interval.end[2], end=epoch.end[2])

    return train_interval, val_interval, test_interval


def split_more_than_five_epochs(epoch):
    assert len(epoch) > 5

    train_interval, val_interval, test_interval = epoch.split(
        [0.6, 0.1, 0.3], shuffle=False
    )
    return train_interval, val_interval, test_interval


def generate_train_valid_test_splits(epoch_dict, grid):
    train_intervals = Interval(np.array([]), np.array([]))
    valid_intervals = Interval(np.array([]), np.array([]))
    test_intervals = Interval(np.array([]), np.array([]))

    for name, epoch in epoch_dict.items():
        if name == "invalid_presentation_epochs":
            logging.warn(f"Found invalid presentation epochs, which will be excluded.")
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
