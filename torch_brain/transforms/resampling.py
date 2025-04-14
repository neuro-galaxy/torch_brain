from typing import List

import logging
from scipy.interpolate import interp1d
from scipy.signal import decimate
import numpy as np
from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries, Interval


class Resampler:
    def __init__(
        self,
        target_sampling_rate: float,
        target_keys: List[str],
    ):
        self.target_sampling_rate = target_sampling_rate
        self.target_keys = target_keys

        # To avoid aliasing, we add a buffer of 10 * sampling_rate / target_sampling_rate
        # points, which is equivalent to 10 / target_sampling_rate seconds
        self._anti_aliasing_buffer = 10.0 / self.target_sampling_rate

    def __call__(self, data: Data, start: float, end: float):
        start = start - self._anti_aliasing_buffer
        end = end + self._anti_aliasing_buffer

        data = data.slice(start, end, reset_origin=False)

        return self.resample(data)

    def resample(self, data: Data, start: float, end: float):
        r"""Resample the data to a new time interval.

        Args:
            data (Data): The data to resample.
            start (float): The start time of the new interval.
            end (float): The end time of the new interval.
        """
        for key in self.target_keys:
            timeseries = data.get_nested_attribute(key)

            if isinstance(timeseries, IrregularTimeSeries):
                timeseries = irregular_to_regular_timeseries(
                    timeseries, target_domain=Interval(start, end)
                )

            # downsample
            downsample_factor = timeseries.sampling_rate / self.target_sampling_rate

            if downsample_factor <= 1:
                raise ValueError(f"Upsampling is not supported")

            if not downsample_factor.is_integer():
                raise ValueError(
                    f"Downsample factor {downsample_factor} is not an integer"
                )

            downsample_factor = int(downsample_factor)
            for attr_key in timeseries.keys():
                x_in = getattr(timeseries, attr_key)
                x_out = decimate(x_in, downsample_factor, axis=0, ftype="iir")
                setattr(timeseries, attr_key, x_out)

            timeseries.sampling_rate = self.target_sampling_rate

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
        target_domain = Interval(timeseries.timestamps[0], timeseries.timestamps[-1])

    timestamps_regular = np.arange(
        target_domain.start[0], target_domain.end[0], dt_target
    )
    domain = Interval(timestamps_regular[0], timestamps_regular[-1])
    out = RegularTimeSeries(sampling_rate=target_sampling_rate, domain=domain)

    for key in timeseries.keys():
        interpolator = interp1d(
            timeseries.timestamps,
            timeseries.get_nested_attribute(key),
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated_values = interpolator(timestamps_regular)
        setattr(out, key, interpolated_values)

    return out
