import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


class Resampler:
    def __init__(
        self,
        args: List[Dict[str, Any]],
        *,
        target_sampling_rate: float,
        method: Optional[str] = "decimate",
    ):
        r"""
        args=[{"target_key": "data_key", "target_sampling_rate": ..., "original_sampling_rate": ..., "method": "decimate"}, ...]

        Args:
            args (List[Dict[str, Any]]): The arguments for the resampling.
            target_sampling_rate (float): The target sampling rate.
            method (str): The method to use for resampling.
        """
        self.args = args
        for arg in self.args:
            # Set default values if not provided in the argument
            if "target_sampling_rate" not in arg:
                arg["target_sampling_rate"] = target_sampling_rate

            if "method" not in arg:
                arg["method"] = method

            if arg["method"] not in ["decimate", "fft"]:
                raise ValueError(
                    f"method {arg['method']} is not supported, only decimate and fft are supported"
                )

            # To avoid aliasing, we add a buffer of 10 * sampling_rate / target_sampling_rate
            # points on left/right side, which is equivalent to 10 / target_sampling_rate seconds
            arg["anti_aliasing_buffer"] = 10.0 / arg["target_sampling_rate"]

    def __call__(self, data: Data, target_domain: Interval):
        if len(target_domain) != 1:
            raise ValueError(
                f"target_domain must contain a single interval, got {len(target_domain)}"
            )

        for arg in self.args:
            target_key = arg["target_key"]
            target_sampling_rate = arg["target_sampling_rate"]
            anti_aliasing_buffer = arg["anti_aliasing_buffer"]
            method = arg["method"]

            timeseries = data.get_nested_attribute(target_key)
            if not isinstance(timeseries, (RegularTimeSeries, IrregularTimeSeries)):
                raise ValueError(f"target_key {target_key} is not a timeseries")

            start, end = target_domain.start[0], target_domain.end[0]

            # check that target_domain is included in data domain
            _overlap = timeseries.domain & target_domain
            if (
                len(_overlap) != 1
                or _overlap.start[0] != start
                or _overlap.end[0] != end
            ):
                raise ValueError(
                    f"target_domain {target_domain} is not included in timeseries domain {timeseries.domain}"
                )

            # add buffer to deal with anti-aliasing
            target_domain_with_buffer = target_domain.dilate(anti_aliasing_buffer)
            target_domain_with_buffer = target_domain_with_buffer & timeseries.domain

            if len(target_domain_with_buffer) > 1:
                idx = np.where(
                    np.logical_and(
                        start >= target_domain_with_buffer.start,
                        end <= target_domain_with_buffer.end,
                    )
                )[0]
                target_domain_with_buffer = Interval(
                    target_domain_with_buffer.start[idx],
                    target_domain_with_buffer.end[idx],
                )

            start_with_buffer = target_domain_with_buffer.start[0]
            end_with_buffer = target_domain_with_buffer.end[0]
            start_with_buffer = start_with_buffer + (start - start_with_buffer) % (
                1.0 / self.target_sampling_rate
            )

            # slice data around the target domain
            timeseries = timeseries.slice(
                start_with_buffer, end_with_buffer, reset_origin=False
            )

            if isinstance(timeseries, IrregularTimeSeries):
                timeseries = irregular_to_regular_timeseries(
                    timeseries,
                    target_domain=target_domain,
                    target_sampling_rate=arg.get("original_sampling_rate", None),
                )

            # resample data
            if method == "decimate":
                timeseries_resampled = _decimate(
                    timeseries,
                    Interval(start_with_buffer, end_with_buffer),
                    target_sampling_rate,
                )
            elif method == "fft":
                timeseries_resampled = _resample_fft(
                    timeseries,
                    Interval(start_with_buffer, end_with_buffer),
                    target_sampling_rate,
                )

            setattr(data, target_key, timeseries_resampled)

        return data


def _decimate(timeseries: RegularTimeSeries, target_sampling_rate: float):
    # compute the downsampling factor
    downsample_factor = timeseries.sampling_rate / target_sampling_rate

    if downsample_factor <= 1:
        raise ValueError(f"Upsampling is not supported")

    if not downsample_factor.is_integer():
        raise ValueError(f"Downsample factor {downsample_factor} is not an integer")

    downsample_factor = int(downsample_factor)

    # create a new RegularTimeSeries with the resampled data
    out = RegularTimeSeries(
        sampling_rate=target_sampling_rate, domain=timeseries.domain
    )

    # resample each attribute
    for att_key in timeseries.keys():
        x_in = getattr(timeseries, att_key)
        x_out = scipy.signal.decimate(x_in, downsample_factor, axis=0, ftype="iir")
        setattr(out, att_key, x_out)

    return out


def _resample_fft(timeseries: RegularTimeSeries, target_sampling_rate: float):
    downsample_factor = timeseries.sampling_rate / target_sampling_rate
    new_size = int(len(timeseries) * downsample_factor)

    # create a new RegularTimeSeries with the resampled data
    out = RegularTimeSeries(
        sampling_rate=target_sampling_rate,
        domain=Interval(
            timeseries.domain.start[0],
            timeseries.domain.start[0] + (new_size - 1) / target_sampling_rate,
        ),
    )

    # resample each attribute
    for att_key in timeseries.keys():
        x_in = getattr(timeseries, att_key)
        sos = scipy.signal.cheby1(
            N=8, rp=0.05, Wn=0.8 / downsample_factor, output="sos"
        )
        x_out = scipy.signal.sosfiltfilt(sos, x_in, axis=-1)
        x_out = scipy.signal.resample(x_out, new_size)
        setattr(out, att_key, x_out)

    return out


def irregular_to_regular_timeseries(
    timeseries: IrregularTimeSeries,
    target_sampling_rate: float = None,
    target_domain: Interval = None,
):
    r"""Resample an irregular time series to a regular time series.
    This assumes that the irregular time series is almost sampled at a regular interval.

    Args:
        timeseries (IrregularTimeSeries): The irregular time series to resample.
        target_sampling_rate (float): The target sampling rate.
        target_domain (Interval): The target domain.
    """
    dt = np.diff(timeseries.timestamps)

    if target_sampling_rate is None:
        target_sampling_rate = np.round(1.0 / np.mean(dt))

    dt_target = 1.0 / target_sampling_rate

    if np.any(np.abs(dt - dt_target) / dt_target > 1e-2):
        logging.warning("t_irregular must be sampled at a regular interval")

    if target_domain is None:
        domain = Interval(timeseries.timestamps[0], timeseries.timestamps[-1])
    else:
        if not isinstance(target_domain, Interval):
            raise ValueError(
                f"target_domain must be an Interval, got {type(target_domain)}"
            )
        if len(target_domain) != 1:
            raise ValueError(
                f"target_domain must contain a single interval, got {len(target_domain)}"
            )

        # TODO could be too restrictive when irregular has an auto domain
        # Check the interval is not outside the timeseries domain
        if target_domain.start[0] < timeseries.domain.start[0]:
            raise ValueError(
                f"target_domain.start {target_domain.start[0]} is outside the data domain {timeseries.domain.start}"
            )
        if target_domain.end[0] > timeseries.domain.end[-1]:
            raise ValueError(
                f"target_domain.end {target_domain.end[-1]} is outside the data domain {timeseries.domain.end}"
            )

        domain = target_domain

    out = RegularTimeSeries(sampling_rate=target_sampling_rate, domain=domain)

    for attr_key in timeseries.keys():
        interpolator = interp1d(
            timeseries.timestamps,
            getattr(timeseries, attr_key),
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated_values = interpolator(out.timestamps)
        setattr(out, attr_key, interpolated_values)

    return out
