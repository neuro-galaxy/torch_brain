import numpy as np
from temporaldata import IrregularTimeSeries


def bin_spikes(
    spikes: IrregularTimeSeries,
    num_units: int,
    bin_size: float,
    right: bool = True,
    eps: float = 1e-3,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Notes:
    - The number of units cannot be inferred from a subset of spikes,
      so `num_units` must be provided explicitly.
    - Floating-point roundoff can cause `(end - start) / bin_size` to be
      very close to an integer without being exact (e.g. 9.99999999).
      The `eps` parameter is added before flooring to make the bin-count
      computation numerically robust.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        right: If True, any excess spikes are truncated from the left end of the time
            series. Otherwise, they are truncated from the right end.
        eps : float, default=1e-3
            Small numerical tolerance added when computing the number of bins
            to avoid floating-point precision issues.
        dtype: Data type of the returned array.
    """
    start = spikes.domain.start[0]
    end = spikes.domain.end[-1]

    # Compute how much time must be discarded so that the duration
    # is an exact multiple of `bin_size`. The epsilon stabilizes
    # the floor operation under floating-point roundoff.
    discard = (end - start) - np.floor(((end - start) / bin_size) + eps) * bin_size
    # Note the discard should in theory always be positive except the numerical instability mention above can make it negative
    # in this case we dont want to reslice it to potentialy loose the last point
    if discard > 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins), dtype=dtype)
    bin_index = np.floor(spikes.timestamps * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)

    return binned_spikes
