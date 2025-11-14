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
        method (str): The method to use to compute the rates. It must be one of "gaussian", "exponential", or "none".
        sigma (float): The sigma for the Gaussian smoothing.
    """

    def __init__(
        self,
        field: str = "spikes",
        bin_size: float = 0.02,
        method: str = "gaussian",
        sigma: float = 2.0,
    ):
        self.field = field
        self.bin_size = bin_size
        self.method = method
        self.sigma = sigma

    def __call__(self, data: Data):

        # Extract object along the nested field path
        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        binned = bin_spikes(
            obj, num_units=len(data.units), bin_size=self.bin_size
        )  # (num_units, num_bins)

        rates = compute_rates(
            binned,
            method=self.method,  # or "exponential" or "none"
            sigma=self.sigma,  # smoothing width
            normalize=True,  # optional
        )  # (num_units, num_bins)

        start = obj.domain.start[0]
        # timestamps represent bin centers
        timestamps = start + (np.arange(rates.shape[1]) + 0.5) * self.bin_size
        rates = rates.T  # (num_bins, num_units)

        data.rates = IrregularTimeSeries(
            values=rates, timestamps=timestamps, domain="auto"
        )

        return data
