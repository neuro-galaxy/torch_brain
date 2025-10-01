import os
import sys
import uuid
import multiprocessing
from pathlib import Path
from math import ceil
import numpy as np
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import matplotlib.colors as colors
from scipy.special import gammaln
from sklearn.cluster import SpectralClustering
from sklearn.metrics import r2_score

"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: for each task variable in var_tasklists, compute PSTH
:var_name2idx: for each task variable in var_tasklists, the corresponding index of X
:var_value2label:
:aligned_tbins: reference time steps to annotate. 
"""


def plot_psth(
    X,
    y,
    y_pred,
    var_tasklist,
    var_name2idx,
    var_value2label,
    aligned_tbins=[],
    axes=None,
    legend=False,
    neuron_idx="",
    neuron_region="",
    save_plot=False,
):

    if save_plot:
        if axes is None:
            nrows = 1
            ncols = len(var_tasklist)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

        for ci, var in enumerate(var_tasklist):
            ax = axes[ci]
            psth_xy = compute_all_psth(X, y, var_name2idx[var])
            psth_pred_xy = compute_all_psth(X, y_pred, var_name2idx[var])

            for _i, _x in enumerate(psth_xy.keys()):

                psth = psth_xy[_x]
                psth_pred = psth_pred_xy[_x]
                ax.plot(
                    psth,
                    color=plt.get_cmap("tab10")(_i),
                    linewidth=3,
                    alpha=0.3,
                    label=f"{var}: {tuple(_x)[0]:.2f}",
                )
                ax.plot(psth_pred, color=plt.get_cmap("tab10")(_i), linestyle="--")
                ax.set_xlabel("Time bin")
                if ci == 0:
                    ax.set_ylabel("Neural activity")
                else:
                    ax.sharey(axes[0])
            _add_baseline(ax, aligned_tbins=aligned_tbins)
            if legend:
                ax.legend()
                ax.set_title(f"{var}")

    # compute PSTH for task_contingency
    idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
    psth_xy = compute_all_psth(X, y, idxs_psth)
    psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy, clip=False)
    r2_single_trial = compute_R2_main(
        y.reshape(-1, 1), y_pred.reshape(-1, 1), clip=False
    )[0]

    if save_plot:
        axes[0].set_ylabel(
            f"Neuron: #{neuron_idx[:4]} \n PSTH R2: {r2_psth:.2f} \n Avg_SingleTrial R2: {r2_single_trial:.2f}"
        )

        for ax in axes:
            # ax.axis('off')
            ax.spines[["right", "top"]].set_visible(False)
            # ax.set_frame_on(False)
            # ax.tick_params(bottom=False, left=False)
        plt.tight_layout()

    return r2_psth, r2_single_trial


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: variables used for computing the task-condition-averaged psth if subtract_psth=='task'
:var_name2idx:
:var_tasklist: variables to be plotted in the single-trial behavior
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:aligned_tbins: reference time steps to annotate. 
:nclus, n_neighbors: hyperparameters for spectral_clustering
:cmap, vmax_perc, vmin_perc: parameters used when plotting the activity and behavior
"""


def plot_single_trial_activity(
    X,
    y,
    y_pred,
    var_name2idx,
    var_behlist,
    var_tasklist,
    subtract_psth="task",
    aligned_tbins=[],
    n_clus=8,
    n_neighbors=5,
    n_pc=32,
    clusby="y_pred",
    cmap="bwr",
    vmax_perc=90,
    vmin_perc=10,
    axes=None,
):
    if axes is None:
        ncols = 1
        nrows = 2 + len(var_behlist) + 1 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        psth_xy = compute_all_psth(X, y, idxs_psth)
        psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
        y_psth = np.asarray(
            [psth_xy[tuple(x)] for x in X[:, 0, idxs_psth]]
        )  # (K, T) predict the neural activity with psth
        y_predpsth = np.asarray(
            [psth_pred_xy[tuple(x)] for x in X[:, 0, idxs_psth]]
        )  # (K, T) predict the neural activity with psth
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_psth = np.mean(y, 0)
        y_predpsth = np.mean(y_pred, 0)
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. ''"
    y_residual = y_pred - y  # (K, T), residuals of prediction
    idxs_behavior = (
        np.concatenate(([var_name2idx[var] for var in var_behlist]))
        if len(var_behlist) > 0
        else []
    )
    X_behs = X[:, :, idxs_behavior]

    clustering = SpectralClustering(
        n_clusters=n_clus,
        n_neighbors=n_neighbors,
        affinity="nearest_neighbors",
        assign_labels="discretize",
        random_state=0,
    )
    if clusby == "y_pred":
        clustering = clustering.fit(y_pred)
    elif clusby == "y":
        clustering = clustering.fit(y)
    else:
        assert False, "invalid clusby"
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(
        zip(
            [y, y_pred, X_behs, y_residual],
            [
                f"obs. act. \n (subtract_psth={subtract_psth})",
                f"pred. act. \n (subtract_psth={subtract_psth})",
                var_behlist,
                "residual act.",
            ],
            [axes[0], axes[1], axes[2:-2], axes[-2]],
        )
    ):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(
                toshow[t_sort],
                vmax,
                vmin,
                True,
                label,
                ax,
                cmap=cmap,
                aligned_tbins=aligned_tbins,
            )
        elif ri == 2:
            # plot behavior
            for bi in range(len(var_behlist)):
                ts_ = toshow[:, :, bi][t_sort]
                vmax = np.percentile(ts_, vmax_perc)
                vmin = np.percentile(ts_, vmin_perc)
                raster_plot(
                    ts_,
                    vmax,
                    vmin,
                    True,
                    label[bi],
                    ax[bi],
                    cmap=cmap,
                    aligned_tbins=aligned_tbins,
                )
        elif ri == 3:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(
                toshow[t_sort],
                vmax,
                vmin,
                True,
                label,
                ax,
                cmap=cmap,
                aligned_tbins=aligned_tbins,
            )

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(
        n_clusters=n_clus,
        n_neighbors=n_neighbors,
        affinity="nearest_neighbors",
        assign_labels="discretize",
        random_state=0,
    ).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(
        y_residual[t_sort_rd],
        np.percentile(y_residual, vmax_perc),
        np.percentile(y_residual, vmin_perc),
        True,
        "residual act. (re-clustered)",
        axes[-1],
    )

    plt.tight_layout()


"""
This script generates a plot to examine the (single-trial) fitting of a single neuron.
:X: behavior matrix of the shape [n_trials, n_timesteps, n_variables]. 
:y: true neural activity matrix of the shape [n_trials, n_timesteps] 
:ypred: predicted activity matrix of the shape [n_trials, n_timesteps] 
:var_name2idx: dictionary mapping feature names to their corresponding index of the 3-rd axis of the behavior matrix X. e.g.: {"choice": [0], "wheel": [1]}
:var_tasklist: *static* task variables used to form the task condition and compute the psth. e.g.: ["choice"]
:var_value2label: dictionary mapping values in X to their corresponding readable labels (only required for static task variables). e.g.: {"choice": {1.: "left", -1.: "right"}}
:var_behlist: *dynamic* behavior variables. e.g., ["wheel"]
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:algined_tbins: reference time steps to annotate in the plot. 
"""


def viz_single_cell(
    X,
    y,
    y_pred,
    var_name2idx,
    var_tasklist,
    var_value2label,
    var_behlist,
    subtract_psth="task",
    aligned_tbins=[],
    clusby="y_pred",
    neuron_idx="",
    neuron_region="",
    method="",
    save_path="figs",
    save_plot=False,
):

    if save_plot:
        nrows = 8
        plt.figure(figsize=(8, 2 * nrows))
        axes_psth = [
            plt.subplot(nrows, len(var_tasklist), k + 1)
            for k in range(len(var_tasklist))
        ]
        axes_single = [
            plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)
        ]
    else:
        axes_psth = None
        axes_single = None

    ### plot psth
    r2_psth, r2_trial = plot_psth(
        X,
        y,
        y_pred,
        var_tasklist=var_tasklist,
        var_name2idx=var_name2idx,
        var_value2label=var_value2label,
        aligned_tbins=aligned_tbins,
        axes=axes_psth,
        legend=True,
        neuron_idx=neuron_idx,
        neuron_region=neuron_region,
        save_plot=save_plot,
    )

    ### plot the psth-subtracted activity
    if save_plot:
        plot_single_trial_activity(
            X,
            y,
            y_pred,
            var_name2idx,
            var_behlist,
            var_tasklist,
            subtract_psth=subtract_psth,
            aligned_tbins=aligned_tbins,
            clusby=clusby,
            axes=axes_single,
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_plot:
        plt.savefig(
            os.path.join(
                save_path,
                f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2_trial:.2f}_{method}.png",
            )
        )
        plt.tight_layout()

    return r2_psth, r2_trial


def viz_single_cell_unaligned(
    gt,
    pred,
    neuron_idx,
    neuron_region,
    method,
    save_path,
    n_clus=8,
    n_neighbors=5,
    save_plot=False,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    r2 = 0
    for _ in range(len(gt)):
        r2 += r2_score(gt, pred)
    r2 /= len(gt)

    if save_plot:
        y = gt - gt.mean(0)
        y_pred = pred - pred.mean(0)
        y_resid = y - y_pred

        clustering = SpectralClustering(
            n_clusters=n_clus,
            n_neighbors=n_neighbors,
            affinity="nearest_neighbors",
            assign_labels="discretize",
            random_state=0,
        )

        clustering = clustering.fit(y_pred)
        t_sort = np.argsort(clustering.labels_)

        vmin_perc, vmax_perc = 10, 90
        vmax = np.percentile(y_pred, vmax_perc)
        vmin = np.percentile(y_pred, vmin_perc)

        toshow = [y, y_pred, y_resid]
        resid_vmax = np.percentile(toshow, vmax_perc)
        resid_vmin = np.percentile(toshow, vmin_perc)

        N = len(y)
        y_labels = ["obs.", "pred.", "resid."]

        fig, axes = plt.subplots(3, 1, figsize=(8, 7))
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im1 = axes[0].imshow(y[t_sort], aspect="auto", cmap="bwr", norm=norm)
        cbar = plt.colorbar(im1, pad=0.02, shrink=0.6)
        cbar.ax.tick_params(rotation=90)
        axes[0].set_title(f" R2: {r2:.3f}")
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im2 = axes[1].imshow(y_pred[t_sort], aspect="auto", cmap="bwr", norm=norm)
        cbar = plt.colorbar(im2, pad=0.02, shrink=0.6)
        cbar.ax.tick_params(rotation=90)
        norm = colors.TwoSlopeNorm(vmin=resid_vmin, vcenter=0, vmax=resid_vmax)
        im3 = axes[2].imshow(y_resid[t_sort], aspect="auto", cmap="bwr", norm=norm)
        cbar = plt.colorbar(im3, pad=0.02, shrink=0.6)
        cbar.ax.tick_params(rotation=90)

        for i, ax in enumerate(axes):
            ax.set_ylabel(f"{y_labels[i]}" + f"\n(#trials={N})")
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.spines[["left", "bottom", "right", "top"]].set_visible(False)

        plt.savefig(
            os.path.join(
                save_path,
                f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2:.2f}_{method}.png",
            )
        )
        plt.tight_layout()

    return r2


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin - 1, c="k", alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(
    ts_, vmax, vmin, whether_cbar, ylabel, ax, cmap="bwr", aligned_tbins=[40]
):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect="auto", cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate(
            "",
            xy=(tbin - 1, N),
            xytext=(tbin - 1, N + 10),
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "r"},
        )
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=0.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}" + f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
        pass
    else:
        ax.axis("off")


"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list
- return: nparray [T, N] or [T]
"""


def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)


def compute_all_psth(X, y, idxs_psth):
    uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
    psth_vs = {}
    for v in uni_vs:
        # compute separately for true y and predicted y
        _psth = compute_PSTH(X, y, axis=idxs_psth, value=v)  # (T)
        psth_vs[tuple(v)] = _psth
    return psth_vs


"""
psth_xy/ psth_pred_xy: {tuple(x): (T) or (T,N)}
return a float or (N) array
"""


def compute_R2_psth(psth_xy, psth_pred_xy, clip=True):
    psth_xy_array = np.array([psth_xy[x] for x in psth_xy])
    psth_pred_xy_array = np.array([psth_pred_xy[x] for x in psth_xy])
    K, T = psth_xy_array.shape[:2]
    psth_xy_array = psth_xy_array.reshape((K * T, -1))
    psth_pred_xy_array = psth_pred_xy_array.reshape((K * T, -1))
    r2s = [
        r2_score(psth_xy_array[:, ni], psth_pred_xy_array[:, ni])
        for ni in range(psth_xy_array.shape[1])
    ]
    r2s = np.array(r2s)
    # # compute r2 along dim 0
    # r2s = [r2_score(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s, 0.0, 1.0)
    # r2s = np.mean(r2s, 0)
    if len(r2s) == 1:
        r2s = r2s[0]
    return r2s


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray(
        [r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)]
    )
    if clip:
        return np.clip(r2s, 0.0, 1.0)
    else:
        return r2s


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
