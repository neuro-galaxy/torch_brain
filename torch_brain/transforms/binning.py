from typing import Optional

import numpy as np
from temporaldata import Data

from torch_brain.utils.binning import bin_spikes


class BinningTransform:
    def __init__(
        self,
        spikes_attr: str,
        units_attr: str,
        bin_size: float,
        max_spikes: Optional[int] = None,
        right: bool = True,
        eps: float = 1e-3,
        dtype: np.dtype = np.float32,
    ):
        self.spikes_attr = spikes_attr
        self.units_attr = units_attr

        self.params = {
            "bin_size": bin_size,
            "max_spikes": max_spikes,
            "right": right,
            "eps": eps,
            "dtype": dtype,
        }

    def __call__(self, data: Data):
        spikes = data.get_nested_attribute(self.spikes_attr)
        units = data.get_nested_attribute(self.units_attr)

        binned_spikes = bin_spikes(spikes, n_units=len(units), **self.params)

        data.set_nested_attribute(f"{self.spikes_attr}_binned", binned_spikes)
        return data
