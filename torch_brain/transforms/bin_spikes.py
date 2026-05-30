from typing import Optional

import numpy as np

from torch_brain.data import Data, RegularTimeSeries
from torch_brain.utils.binning import bin_spikes


class BinSpikes:
    r"""Bin spike events into fixed-width time bins.

    The transform reads spikes and units from nested attributes, applies
    :func:`torch_brain.utils.binning.bin_spikes`, and stores the result in a new
    nested attribute named ``{spikes_attribute}_binned``.

    Args:
        bin_size: Bin width in seconds.
        spikes_attribute: Nested attribute path to the spikes ``IrregularTimeseries``.
        units_attribute: Nested attribute path to the units ``ArrayDict``.
        max_spikes: Maximum number of spikes to include per unit per
            bin. If ``None``, no clipping is applied.
        right: Decide which side gets truncated when duration is not
            a multiple of ``bin_size``. If ``True``, excess spikes are truncated from the left edge.
        eps: Small numerical margin used during bin assignment.
        dtype: Data type of the output binned array. (default ``np.int32``)
    """

    def __init__(
        self,
        bin_size: float,
        spikes_attribute: str = "spikes",
        units_attribute: str = "units",
        max_spikes: Optional[int] = None,
        right: bool = True,
        eps: float = 1e-3,
        dtype: np.dtype = np.int32,
    ):
        self.spikes_attr = spikes_attribute
        self.units_attr = units_attribute

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

        binned_counts = bin_spikes(spikes, num_units=len(units), **self.params)

        # RegularTimeSeries expects time on axis 0; bin_spikes returns (units, bins).
        binned_spikes = RegularTimeSeries(
            sampling_rate=1 / self.params["bin_size"],
            binned_counts=binned_counts,
            domain="auto",
            domain_start=spikes.domain.start[0],
        )

        data.set_nested_attribute(f"{self.spikes_attr}_binned", binned_spikes)
        return data
