import copy

import numpy as np
from scipy.signal import resample
from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries


class Resampler:
    def __init__(
        self,
        base_frequency: float,
        resample_frequency: float,
        extra_window_factor: int = 10,
    ):
        self.base_frequency = base_frequency
        self.resample_frequency = resample_frequency
        # TODO add the ref from where this come from
        if extra_window_factor < 4 or 10 < extra_window_factor:
            raise ValueError("extra_window_factor must be between 4 and 10, inclusive.")
        self.extra_window_factor = extra_window_factor
        # TODO not sure of this check
        if self.base_frequency / self.resample_frequency % 1 != 0:
            raise ValueError(
                "The base frequency must be a multiple of the resample frequency."
            )
        self.extra_window_length = (
            extra_window_factor * self.base_frequency / self.resample_frequency
        )

    def __call__(self, data: Data, start: float, end: float):
        return self.resample(data, start, end)

    def resample(self, data: Data, start: float, end: float):
        r"""Resample the data to a new time interval.

        Args:
            data (Data): The data to resample.
            start (float): The start time of the new interval.
            end (float): The end time of the new interval.
        """
        out = data.__class__.__new__(data.__class__)
        data = copy.deepcopy(data)

        # TODO: intersting in you opinion about this
        # When we are in a center window, because we divide by two the extra_window_length we could
        # be benefical to ensure we are using an offset to keep start and end in the sampled point in the new window
        offsett = self.extra_window_length / 2 % self.resample_frequency
        if (
            data.domain.start <= start - self.extra_window_length / 2 - offsett
            and end + self.extra_window_length / 2 - offsett < data.domain.end
        ):
            window_start = start - self.extra_window_length / 2 - offsett
            window_end = end + self.extra_window_length / 2 - offsett
        elif end + self.extra_window_length < data.domain.end:
            window_start = start
            window_end = end + self.extra_window_length
        elif data.domain.start <= start - self.extra_window_length:
            window_start = start - self.extra_window_length
            window_end = end
        else:
            raise ValueError(
                f"Data domain {data.domain} does not contain the requested interval [{start}, {end}]"
            )

        new_data = data.slice(window_start, window_end, reset_origin=False)
        for key, value in new_data.__dict__.items():
            if isinstance(value, IrregularTimeSeries):
                val = copy.copy(value)
                timestamps = np.arange(
                    window_start,
                    window_end,
                    1 / self.resample_frequency,
                    dtype=np.float64,
                )
                num = len(timestamps)
                tmp = {}
                for timekey in val._timekeys:
                    if timekey == "timestamps":
                        continue

                    tmp[timekey] = resample(getattr(val, timekey), num)

                val_unsliced = IrregularTimeSeries(
                    timestamps=timestamps,
                    **tmp,
                    domain="auto",
                )

                # TODO the slicing will be process after anyway and could be ommited
                # However it could be clean to have it already sliced (and optimized memory in case of big window_length)
                # Intersted about feedback here
                out.__dict__[key] = val_unsliced.slice(start, end, reset_origin=False)

            elif isinstance(value, RegularTimeSeries):
                val = copy.copy(value)
                window_length = window_start - window_start
                num = int(window_length * self.resample_frequency)
                tmp = {}
                for timekey in val.keys():
                    tmp[timekey] = resample(getattr(val, timekey), num)

                val_unsliced = RegularTimeSeries(
                    sampling_rate=self.resample_frequency,
                    **tmp,
                    domain=copy.copy(val._domain),
                )
                out.__dict__[key] = val_unsliced.slice(start, end, reset_origin=False)

            else:
                out.__dict__[key] = copy.copy(value)

        return out
