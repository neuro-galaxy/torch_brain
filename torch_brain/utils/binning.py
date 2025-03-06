import numpy as np
from temporaldata import IrregularTimeSeries


def bin_spikes(
    spikes: IrregularTimeSeries,
    num_units: int,
    bin_size: float,
    right: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Note that we cannot infer the number of units from a chunk of spikes, hence why it
    must be provided as an argument.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        right: If True, any excess spikes are truncated from the left end of the time
            series. Otherwise, they are truncated from the right end.
        dtype: Data type of the returned array.
    """
    start = spikes.domain.start[0]
    end = spikes.domain.end[-1]

    discard = (end - start) - np.floor((end - start) / bin_size) * bin_size
    if discard != 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins), dtype=dtype)
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)

    return binned_spikes
