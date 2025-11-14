import numpy as np
from temporaldata import Data

from torch_brain.utils.binning import bin_spikes
from torch_brain.utils.spike_rates import compute_rates


class RateDrift:
    """
    Add artificial mean shifts (constant offset) and random-walk drift
    to firing-rate features.

    Args:
        field (str): Path to the rate data inside Data. E.g., "rates.data".
                     The object at this field must be a numpy array of shape (T, D)
                     or (D, T). The augmentation is applied elementwise.
        offset_scale (float): Std dev of the constant offset noise.
        drift_scale (float): Std dev of each random-walk increment.
        seed (int): Random seed.
    """

    def __init__(
        self,
        field: str = "spikes",
        offset_scale: float = 0.05,
        drift_scale: float = 0.01,
        seed: int = None,
    ):
        self.field = field
        self.offset_scale = offset_scale
        self.drift_scale = drift_scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, data: Data):

        # Extract object along the nested field path
        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        binned = bin_spikes(obj, num_units=len(data.units), bin_size=0.02)

        rates = compute_rates(
            binned,
            method="gaussian",  # or "exponential" or "none"
            sigma=2.0,  # smoothing width
            normalize=True,  # optional
        )

        # ---------------------------------------------
        # 1. Constant offset noise (φ)
        # ---------------------------------------------
        # One offset per feature dimension
        offset = self.rng.normal(loc=0.0, scale=self.offset_scale, size=rates.shape[1:])

        # Broadcast to match shape (T, D)
        offset_term = offset

        # ---------------------------------------------
        # 2. Random-walk drift (sum of ν_i)
        # ---------------------------------------------
        # Independent Gaussian increment at each time step
        drift_steps = self.rng.normal(loc=0.0, scale=self.drift_scale, size=rates.shape)
        # Cumulative sum along time axis (axis 0)
        drift_term = np.cumsum(drift_steps, axis=0)

        # ---------------------------------------------
        # 3. Apply augmentation
        # ---------------------------------------------
        augmented = rates + offset_term + drift_term

        data.rates = augmented

        # # Assign back
        # setattr(data, self.field, augmented)
        return data
