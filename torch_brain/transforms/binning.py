from typing import Optional

import numpy as np
from temporaldata import Data

from torch_brain.utils.binning import bin_spikes


class BinSpikes:
    r"""Bin spike events into fixed-width time bins.

    The transform reads spikes and units from nested attributes, applies
    :func:`torch_brain.utils.binning.bin_spikes`, and stores the result in a new
    nested attribute named ``{spikes_attribute}_binned``.

    Args:
        spikes_attribute (str): Nested attribute path to the spikes object.
        units_attribute (str): Nested attribute path to the units object.
        bin_size (float): Bin width in seconds.
        max_spikes (int, optional): Maximum number of spikes to include per unit per
            bin. If ``None``, no clipping is applied.
        right (bool, optional): Decide which side gets truncated when duration is not
            a multiple of ``bin_size``. If ``True``, excess spikes are truncated from the left edge.
        eps (float, optional): Small numerical margin used during bin assignment.
        dtype (np.dtype, optional): Data type of the output binned array.
    """

    def __init__(
        self,
        bin_size: float,
        spikes_attribute: str = "spikes",
        units_attribute: str = "units",
        max_spikes: Optional[int] = None,
        right: bool = True,
        eps: float = 1e-3,
        dtype: np.dtype = np.float32,
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

        binned_spikes = bin_spikes(spikes, num_units=len(units), **self.params)

        _set_nested_attribute(data, f"{self.spikes_attr}_binned", binned_spikes)
        return data


def _set_nested_attribute(data, path: str, value):
    r"""Set a nested attribute in a :class:`temporaldata.Data` object using a dot-separated path.

    Args:
        data: The :class:`temporaldata.Data` object to modify.
        path: The dot-separated path to the nested attribute (e.g., "session.id").
        value: The value to set for the attribute.

    Returns:
        The modified data object (same instance, modified in-place).

    Raises:
        AttributeError: If any component of the path cannot be resolved.
    """
    # Split key by dots, resolve using getattr
    components = path.split(".")
    obj = data
    for c in components[:-1]:
        try:
            obj = getattr(obj, c)
        except AttributeError:
            raise AttributeError(
                f"Could not resolve {path} in data (specifically, at level {c})"
            )

    setattr(obj, components[-1], value)
    return data
