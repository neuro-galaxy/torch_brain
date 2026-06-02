from packaging import version
import numpy as np


def np_string_prefix(prefix: str, array: np.ndarray) -> np.ndarray:
    """
    Adds a string prefix to each element of a numpy string array.

    Args:
        prefix: The string to prepend to each element.
        array: An array of strings or string-like objects.

    Returns:
        np.ndarray: New array with the prefix added to each element.
    """
    if version.parse(np.__version__) >= version.parse("2.0"):
        return np.strings.add(prefix, array)
    else:
        return np.core.defchararray.add(prefix, array)


def calculate_sampling_rate(timestamps: np.ndarray, rtol: float = 1e-3) -> float:
    """Calculates median sampling rate from an array of timestamps.

    Args:
        timestamps: 1D array of timestamps in seconds, expected to be monotonically increasing.
        rtol: Maximum allowed relative variation in sampling interval, defined as
            (max_diff - min_diff) / median_diff. Defaults to 1e-3.

    Returns:
        float: Sampling rate in Hz.

    Raises:
        ValueError: If fewer than 2 timestamps are provided.
        ValueError: If the timestamps are not strictly monotonically increasing.
        ValueError: If the timestamps are not uniformly sampled within the given relative tolerance.
    """

    if timestamps.ndim != 1:
        raise ValueError(
            f"Timestamps must be a 1D array, got {timestamps.ndim}D array with shape {timestamps.shape}"
        )

    if timestamps.size < 2:
        raise ValueError(
            f"Need at least 2 timestamps to compute a sampling rate, got {timestamps.size}"
        )

    diffs = np.diff(timestamps)

    if np.any(diffs <= 0):
        raise ValueError(
            "Timestamps must be strictly monotonically increasing "
            "(found duplicate or out-of-order values)"
        )

    dt = np.median(diffs)
    relative_variation = np.abs((np.max(diffs) - np.min(diffs)) / dt)
    if relative_variation > rtol:
        raise ValueError(
            f"Timestamps are not uniformly sampled (relative variation={relative_variation:.2e} >= rtol={rtol}). "
            "Use IrregularTimeSeries to store the data."
        )

    return 1.0 / dt
