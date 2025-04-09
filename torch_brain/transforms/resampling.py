import copy

import numpy as np
from scipy.signal import resample
from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries


class Resampler:
    def __init__(self, base_frequency: float, resample_frequency: float):
        self.base_frequency = base_frequency
        self.resample_frequency = resample_frequency
        self.extra_window_length = 10 * self.resample_frequency / self.base_frequency

    def __call__(self, data: Data, start: float, end: float):
        return self.resample(data, start, end)

    def resample(self, data: Data, start: float, end: float):
        r"""Resample the data to a new time interval.

        Args:
            data (Data): The data to resample.
            start (float): The start time of the new interval.
            end (float): The end time of the new interval.
        """
        # TODO center the interval around the start and end time
        out = data.__class__.__new__(data.__class__)
        data = copy.deepcopy(data)
        new_data = data.slice(start, end + self.extra_window_length)
        for key, value in new_data.__dict__.items():
            if isinstance(value, IrregularTimeSeries):
                val = copy.copy(value)
                timestamps = np.arange(
                    start,
                    end + self.extra_window_length,
                    1 / self.resample_frequency,
                    dtype=np.float64,
                )
                num = len(timestamps)
                tmp = {}

                for timekey in val._timekeys:
                    if timekey == "timestamps":
                        continue
                    tmp[timekey] = resample(getattr(val, timekey), num)

                out.__dict__[key] = IrregularTimeSeries(
                    timestamps=timestamps,
                    **tmp,
                    domain="auto",
                )

            elif isinstance(value, RegularTimeSeries):
                val = copy.copy(value)
                val._sampling_rate = self.resample_frequency
                val._domain = copy.copy(value._domain)
                for timekey in value.get("_timekeys", []):
                    if timekey == "timestamps":
                        continue
                    val.__dict__[timekey] = resample(val.timekey, num)
                out.__dict__[key] = val

            else:
                out.__dict__[key] = copy.copy(value)

        return out
