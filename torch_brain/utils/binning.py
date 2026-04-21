from typing import Optional

import numpy as np
from temporaldata import IrregularTimeSeries


def bin_spikes(
    spikes: IrregularTimeSeries,
    num_units: int,
    bin_size: float,
    max_spikes: Optional[int] = None,
    right: bool = True,
    eps: float = 1e-3,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Notes:
        - The number of units cannot be inferred from a subset of spikes,
          so ``num_units`` must be provided explicitly.
        - Floating-point roundoff can cause ``(end - start) / bin_size`` to be
          very close to an integer without being exact (e.g. 9.99999999).
          The ``eps`` parameter is added before flooring to make the bin-count
          computation numerically robust.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        max_spikes: Maximum number of spikes to include per unit per
            bin. If ``None``, no clipping is applied.
        right: Decide which side gets truncated when duration is not
            a multiple of ``bin_size``. If ``True``, excess spikes are truncated from the left edge.
        eps: Small numerical margin used during bin assignment.
        dtype: Data type of the output binned array. (default ``np.int32``)

    Returns:
        Binned spike counts with shape ``(T, N)``, where ``T`` is the number of
        time bins and ``N`` is ``num_units``.
    """
    start = spikes.domain.start[0]
    end = spikes.domain.end[-1]

    # Compute how much time must be discarded so that the duration
    # is an exact multiple of `bin_size`. The epsilon stabilizes
    # the floor operation under floating-point roundoff.
    discard = (end - start) - np.floor(((end - start) / bin_size) + eps) * bin_size
    # In theory, `discard` should always be non-negative.
    # Floating-point roundoff may make it slightly negative,
    # in that case, we avoid reslicing to prevent dropping the last spike.
    if discard > 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_bins, num_units), dtype=dtype)
    # Handle timestamps when the domain start is non-zero
    ts = spikes.timestamps - spikes.domain.start[0]
    bin_index = np.floor(ts * rate).astype(int)
    np.add.at(binned_spikes, (bin_index, spikes.unit_index), 1)
    if max_spikes is not None:
        np.clip(binned_spikes, None, max_spikes, out=binned_spikes)

    return binned_spikes
