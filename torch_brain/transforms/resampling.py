import copy
import logging
from typing import List

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import decimate
from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


class Resampler:
    def __init__(
        self,
        target_sampling_rate: float,
        target_keys: List[str],
    ):
        self.target_sampling_rate = target_sampling_rate
        self.target_keys = target_keys

        # To avoid aliasing, we add a buffer of 10 * sampling_rate / target_sampling_rate
        # points on left/right side, which is equivalent to 10 / target_sampling_rate seconds
        self._anti_aliasing_buffer = 10.0 / self.target_sampling_rate

        self.needs_target_domain = True

    def __call__(self, data: Data, target_domain: Interval):
        if len(target_domain.start) != 1 or len(target_domain.end) != 1:
            raise ValueError(
                f"target_domain must be a 1D Interval, got {target_domain}"
            )

        start, end = target_domain.start[0], target_domain.end[0]

        if start < data.domain.start[0]:
            raise ValueError(
                f"target_domain.start {start} is outside the data domain {data.domain}"
            )
        if end > data.domain.end[-1]:
            raise ValueError(
                f"target_domain.end {end} is outside the data domain {data.domain}"
            )

        # TODO be sure we are taking an offset when the data is not aligned

        # Ensure the buffer_start and buffer_end are inside the data domain
        buffer_start = max(start - self._anti_aliasing_buffer, data.domain.start[0])
        buffer_end = min(end + self._anti_aliasing_buffer, data.domain.end[-1])

        sliced_buffered_data = data.slice(buffer_start, buffer_end, reset_origin=False)

        resampled_data = self.resample(sliced_buffered_data, target_domain)

        return resampled_data.slice(start, end, reset_origin=False)

    def resample(self, data: Data, target_domain: Interval):
        r"""Resample the data.

        Args:
            data (Data): The data to resample.
        """
        for key in self.target_keys:
            timeseries = data.get_nested_attribute(key)

            if isinstance(timeseries, IrregularTimeSeries):
                timeseries = irregular_to_regular_timeseries(
                    timeseries, target_domain=target_domain
                )
                print(timeseries.timestamps)

            # downsample
            downsample_factor = timeseries.sampling_rate / self.target_sampling_rate

            if downsample_factor <= 1:
                raise ValueError(f"Upsampling is not supported")

            if not downsample_factor.is_integer():
                raise ValueError(
                    f"Downsample factor {downsample_factor} is not an integer"
                )

            downsample_factor = int(downsample_factor)

            resampled_dic = {}
            for attr_key in timeseries.keys():
                x_in = getattr(timeseries, attr_key)
                x_out = decimate(x_in, downsample_factor, axis=0, ftype="iir")
                resampled_dic[attr_key] = x_out

            # Create a new RegularTimeSeries with the resampled data
            resampled_timeseries = RegularTimeSeries(
                sampling_rate=self.target_sampling_rate,
                **resampled_dic,
                domain=copy.copy(timeseries.domain),
            )
            setattr(data, key, resampled_timeseries)

        return data


def irregular_to_regular_timeseries(
    timeseries: IrregularTimeSeries,
    target_sampling_rate: float = None,
    target_domain: Interval = None,
):
    dt = np.diff(timeseries.timestamps)

    if target_sampling_rate is None:
        target_sampling_rate = np.round(1.0 / np.mean(dt))

    dt_target = 1.0 / target_sampling_rate

    if np.any(np.abs(dt - dt_target) / dt_target > 1e-2):
        logging.warning("t_irregular must be sampled at a regular interval")

    if target_domain is None:
        domain = Interval(timeseries.timestamps[0], timeseries.timestamps[-1])
    else:
        # Check that target_domain is a valid interval
        if not isinstance(target_domain, Interval):
            raise ValueError(
                f"target_domain must be an Interval, got {type(target_domain)}"
            )
        if len(target_domain.start) != 1 or len(target_domain.end) != 1:
            raise ValueError(
                f"target_domain must be a 1D Interval, got {target_domain}"
            )

        # Check the interval is not outside the timeseries domain
        if target_domain.start[0] < timeseries.timestamps[0]:
            raise ValueError(
                f"target_domain.start {target_domain.start[0]} is outside the data domain {timeseries.domain.start}"
            )
        if target_domain.end[0] > timeseries.timestamps[-1]:
            raise ValueError(
                f"target_domain.end {target_domain.end[-1]} is outside the data domain {timeseries.domain.end}"
            )

        domain = target_domain

    out = RegularTimeSeries(sampling_rate=target_sampling_rate, domain=domain)

    timestamps_regular = np.arange(domain.start[0], domain.end[0], dt_target)

    print(target_domain.end)
    for attr_key in timeseries.keys():
        if attr_key == "timestamps":
            continue
        interpolator = interp1d(
            timeseries.timestamps,
            getattr(timeseries, attr_key),
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated_values = interpolator(timestamps_regular)
        setattr(out, attr_key, interpolated_values)

    return out
