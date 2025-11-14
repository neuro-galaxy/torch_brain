from temporaldata import Data
import numpy as np
from torch_brain.utils.binning import bin_spikes
from torch_brain.utils.spike_rates import compute_rates
from temporaldata import IrregularTimeSeries


class AddRate:
    """
    Add a spikerate to the data.

    Args:
        field (str): Path to the spikes data inside Data. E.g., "spikes".
        bin_size (float): The size of the time bins in seconds.
    """

    def __init__(self, field: str = "spikes", bin_size: float = 0.02):
        self.field = field
        self.bin_size = bin_size

    def __call__(self, data: Data):

        # Extract object along the nested field path
        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        binned = bin_spikes(obj, num_units=len(data.units), bin_size=self.bin_size)

        rates = compute_rates(
            binned,
            method="gaussian",  # or "exponential" or "none"
            sigma=2.0,  # smoothing width
            normalize=True,  # optional
        )

        start = obj.domain.start[0]
        # timestamps represent bin centers
        timestamps = start + (np.arange(rates.shape[1]) + 0.5) * self.bin_size
        rates = rates.T

        data.rates = IrregularTimeSeries(
            values=rates, timestamps=timestamps, domain="auto"
        )

        return data
