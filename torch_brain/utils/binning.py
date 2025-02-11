import numpy as np
from temporaldata import IrregularTimeSeries


def bin_spikes(
    spikes: IrregularTimeSeries, num_units: int, bin_size: float, right: bool = True
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Note that we cannot infer the number of units from a chunk of spikes, hence why it
    must be provided as an argument.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        right: If True, any excess spikes are truncated from the left end of the time
            series. Otherwise, they are truncated from the right end.
    """
    start = spikes.domain.start[0]
    end = spikes.domain.end[-1]

    discard = (end - start) - np.floor((end - start) / bin_size) * bin_size
    if discard != 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)

    return binned_spikes


# Have put this here
# However I dont know if we can derive a function from it that is more generic
# compare to the one for IBL bhvr binning


def get_behavior_per_interval(
    target_times,
    target_vals,
    intervals=None,
    trials_df=None,
    allow_nans=False,
    n_workers=1,
    **kwargs,
):
    """
    Format a single session-wide array of target data into a list of interval-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    intervals :
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip intervals with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each interval
        - (list): data for each interval
        - (array-like): mask of good intervals (True) and bad intervals (False)

    """

    binsize = kwargs["binsize"]
    align_interval = kwargs["time_window"]
    interval_len = align_interval[1] - align_interval[0]

    if trials_df is not None:
        align_event = kwargs["align_time"]
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert (
            intervals is not None
        ), "Require intervals to segment the recording into chunks including trials and non-trials."
        interval_begs, interval_ends = intervals.T

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print("interval times all nan")
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_interval

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = [None for _ in range(len(target_times_og_list))]
    target_vals_list = [None for _ in range(len(target_times_og_list))]
    good_interval = [None for _ in range(len(target_times_og_list))]
    skip_reasons = [None for _ in range(len(target_times_og_list))]


import sys
import uuid

import numpy as np
from scipy.interpolate import interp1d


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def get_behavior_per_interval(
    target_times,
    target_vals,
    intervals=None,
    trials_df=None,
    allow_nans=False,
    n_workers=1,
    **kwargs,
):
    """
    Format a single session-wide array of target data into a list of interval-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    intervals :
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip intervals with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each interval
        - (list): data for each interval
        - (array-like): mask of good intervals (True) and bad intervals (False)

    """

    binsize = kwargs["binsize"]
    align_interval = kwargs["time_window"]
    interval_len = align_interval[1] - align_interval[0]

    if trials_df is not None:
        align_event = kwargs["align_time"]
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert (
            intervals is not None
        ), "Require intervals to segment the recording into chunks including trials and non-trials."
        interval_begs, interval_ends = intervals.T

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print("interval times all nan")
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_interval

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = [None for _ in range(len(target_times_og_list))]
    target_vals_list = [None for _ in range(len(target_times_og_list))]
    good_interval = [None for _ in range(len(target_times_og_list))]
    skip_reasons = [None for _ in range(len(target_times_og_list))]

    @globalize
    def interpolate_behavior(target):
        # We use interval_idx to track the interval order while working with p.imap_unordered()
        interval_idx, target_time, target_vals = target

        is_good_interval, x_interp, y_interp = False, None, None

        if len(target_vals) == 0:
            skip_reason = "target data not present"
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            skip_reason = "nans in target data"
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.isnan(interval_begs[interval_idx]) or np.isnan(
            interval_ends[interval_idx]
        ):
            skip_reason = "bad interval data"
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_begs[interval_idx] - target_time[0]) > binsize:
            skip_reason = "target data starts too late"
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_ends[interval_idx] - target_time[-1]) > binsize:
            skip_reason = "target data ends too early"
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

        is_good_interval, skip_reason = True, None
        x_interp = np.linspace(
            interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins
        )
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(
                    interp1d(
                        target_time,
                        target_vals[:, n],
                        kind="linear",
                        fill_value="extrapolate",
                    )(x_interp)
                )
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_vals, kind="linear", fill_value="extrapolate"
            )(x_interp)
        return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

    # with multiprocessing.Pool(processes=n_workers) as p:
    #     targets = list(
    #         zip(np.arange(n_intervals), target_times_og_list, target_vals_og_list)
    #     )
    #     with tqdm(total=n_intervals) as pbar:
    #         for res in p.imap_unordered(interpolate_behavior, targets):
    #             pbar.update()
    #             good_interval[res[0]] = res[1]
    #             target_times_list[res[0]] = res[2]
    #             target_vals_list[res[0]] = res[3]
    #             skip_reasons[res[0]] = res[-1]
    #     pbar.close()
    #     p.close()

    targets = list(
        zip(np.arange(n_intervals), target_times_og_list, target_vals_og_list)
    )
    for target in targets:
        res = interpolate_behavior(target)
        good_interval[res[0]] = res[1]
        target_times_list[res[0]] = res[2]
        target_vals_list[res[0]] = res[3]
        skip_reasons[res[0]] = res[-1]

    return target_times_list, target_vals_list, np.array(good_interval), skip_reasons


def bin_behaviors(
    target_times,
    target_vals,
    intervals=None,
    beh="whisker",
    mask=None,
    allow_nans=True,
    n_workers=1,
    **kwargs,
):

    behave_dict, mask_dict = {}, {}

    target_times_list, target_vals_list, target_mask, skip_reasons = (
        get_behavior_per_interval(
            target_times,
            target_vals,
            intervals=intervals,
            allow_nans=allow_nans,
            n_workers=n_workers,
            **kwargs,
        )
    )
    # behave_dict.update({beh: np.array(target_vals_list, dtype=object)})
    behave_dict.update({beh: target_vals_list[0]})
    mask_dict.update({beh: target_mask})
    behave_mask = target_mask

    if not allow_nans:
        for k, v in behave_dict.items():
            behave_dict[k] = behave_dict[beh][behave_mask]

    return behave_dict, mask_dict
