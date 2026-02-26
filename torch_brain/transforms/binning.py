from typing import Optional

import numpy as np
from temporaldata import Data

from torch_brain.utils.binning import bin_spikes


class BinningTransform:
    r"""Bin spike events into fixed-width time bins.

    The transform reads spikes and units from nested attributes, applies
    :func:`torch_brain.utils.binning.bin_spikes`, and stores the result in a new
    nested attribute named ``{spikes_attr}_binned``.

    Args:
        spikes_attr (str): Nested attribute path to the spikes object.
        units_attr (str): Nested attribute path to the units object.
        bin_size (float): Bin width in seconds.
        max_spikes (int, optional): Maximum number of spikes to include per unit per
            bin. If ``None``, no clipping is applied.
        right (bool, optional): If ``True``, bins include the right edge.
        eps (float, optional): Small numerical margin used during bin assignment.
        dtype (np.dtype, optional): Data type of the output binned array.
    """

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

        binned_spikes = bin_spikes(spikes, num_units=len(units), **self.params)

        data.set_nested_attribute(f"{self.spikes_attr}_binned", binned_spikes)
        return data
