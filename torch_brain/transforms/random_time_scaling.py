import copy
import torch

from temporaldata import IrregularTimeSeries, RegularTimeSeries, Interval, Data


def rescale(data: Data, scale: float, offset: float):
    r"""Rescale the time axis of the data by a factor and offset.

    Args:
        data (Data): The data to rescale.
        scale (float): The scaling factor.
        offset (float): The offset.
    """
    out = data.__class__.__new__(data.__class__)

    for key, value in data.__dict__.items():
        # todo update domain
        if key != "_domain" and isinstance(value, IrregularTimeSeries):
            val = copy.copy(value)
            val.timestamps = val.timestamps * scale + offset
            val._domain = copy.copy(value._domain)
            val._domain.start = val._domain.start * scale + offset
            val._domain.end = val._domain.end * scale + offset
            out.__dict__[key] = val
        elif key != "_domain" and isinstance(value, RegularTimeSeries):
            val = copy.copy(value)
            val._sampling_rate = val._sampling_rate / scale
            val._domain = copy.copy(value._domain)
            val._domain.start = val._domain.start * scale + offset
            val._domain.end = val._domain.end * scale + offset
            out.__dict__[key] = val
        elif key != "_domain" and isinstance(value, Interval):
            val = copy.copy(value)
            val.start = val.start * scale + offset
            val.end = val.end * scale + offset
            out.__dict__[key] = val
        else:
            out.__dict__[key] = copy.copy(value)

    # update domain
    out._domain = copy.copy(data._domain)
    out._domain.start = out._domain.start * scale + offset
    out._domain.end = out._domain.end * scale + offset

    # update slice start time
    out._absolute_start = data._absolute_start

    return out


class RandomTimeScaling:
    def __init__(self, min_scale, max_scale, min_offset=0, max_offset=0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_offset = min_offset
        self.max_offset = max_offset

    def __call__(self, data):

        scale = (
            torch.rand(1).item() * (self.max_scale - self.min_scale) + self.min_scale
        )
        offset = (
            torch.rand(1).item() * (self.max_offset - self.min_offset) + self.min_offset
        )

        return rescale(data, scale, offset)
