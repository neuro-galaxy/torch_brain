import numpy as np

from torch_brain.data import Interval


def isin_interval(timestamps: np.ndarray, interval: Interval) -> np.ndarray:
    r"""Check if timestamps are in any of the intervals in the `Interval` object.

    Args:
        timestamps: Timestamps to check.
        interval: Interval to check against.

    Returns:
        Boolean mask of the same shape as `timestamps`.
    """
    if len(interval) == 0:
        return np.zeros_like(timestamps, dtype=bool)

    timestamps_expanded = timestamps[:, None]
    mask = np.any(
        (timestamps_expanded >= interval.start) & (timestamps_expanded < interval.end),
        axis=1,
    )
    return mask
