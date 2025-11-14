import numpy as np
from temporaldata import IrregularTimeSeries, Data


class TemporalJitter:
    """
    Add temporal jitter to IrregularTimeSeries timestamps.

    Disclaimer: This transform is only compatible with IrregularTimeSeries.

    Args:
        field (str, optional): Path to field, e.g. "lfp.data", "rates.data".
        jitter_std (float): Standard deviation of Gaussian time jitter.
        max_jitter (float or None): Optional clipping bound; if given, jitter is
                                    clipped to [-max_jitter, max_jitter].
        seed (int): Random seed.
    """

    def __init__(
        self,
        field: str = "spikes",
        loc: float = 0.0,
        scale: float = 0.001,
        max_jitter: float = None,
        seed: int = None,
    ):
        self.field = field
        self.loc = loc
        self.scale = scale
        self.max_jitter = max_jitter
        self.rng = np.random.default_rng(seed)

    def __call__(self, data: Data):

        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        if not isinstance(obj, IrregularTimeSeries):
            raise ValueError(
                f"TemporalJitter only works on IrregularTimeSeries, got {type(obj)}"
            )

        timestamps = obj.timestamps

        jitter = self.rng.normal(loc=self.loc, scale=self.scale, size=timestamps.shape)

        if self.max_jitter is not None:
            jitter = np.clip(jitter, -self.max_jitter, self.max_jitter)

        new_timestamps = timestamps + jitter

        # Enforce non-negative times
        new_timestamps = np.maximum(new_timestamps, 0)

        # Preserve event order
        sort_idx = np.argsort(new_timestamps)
        values = getattr(obj, nested[1])
        obj.timestamps = new_timestamps[sort_idx]
        setattr(obj, nested[1], values[sort_idx])

        return data
