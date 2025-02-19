import re
from typing import Callable, Pattern, Union

import numpy as np

from temporaldata import ArrayDict, Data, IrregularTimeSeries, RegularTimeSeries


class UnitFilter:
    r"""
    Drop units based on the `mask_fn` given in the constructor.

    Args:
        mask_fn (Callable[[ArrayDict], np.ndarray]): A function that takes the unit ids and returns a mask to keep the units.
        field (str): The field to apply the filter.
        reset_index (bool, optional): If True, it will reset_index the unit index of the time series.
    """

    def __init__(
        self,
        mask_fn: Callable[[ArrayDict], np.ndarray],
        field: str,
        reset_index: bool = True,
    ):
        self.field = field
        self.mask_fn = mask_fn
        self.reset_index = reset_index

    def __call__(self, data: Data) -> Data:
        # convention: True means keep the unit
        unit_mask = self.mask_fn(data.units)

        original_num_units = len(data.units.id)
        if self.reset_index:
            data.units = data.units.select_by_mask(unit_mask)

        target_obj = getattr(data, self.field)
        if isinstance(target_obj, IrregularTimeSeries):
            target_mask = np.isin(target_obj.unit_index, np.where(unit_mask)[0])
            setattr(data, self.field, target_obj.select_by_mask(target_mask))

            if self.reset_index:
                # hack to have the lookup array that remaps the unit index
                relabel_map = np.zeros(original_num_units, dtype=int)
                relabel_map[unit_mask] = np.arange(unit_mask.sum())
                target_obj = getattr(data, self.field)
                target_obj.unit_index = relabel_map[target_obj.unit_index]

        elif isinstance(target_obj, RegularTimeSeries):
            raise NotImplementedError("RegularTimeSeries is not supported yet.")
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")
        return data


class UnitFilterById(UnitFilter):
    r"""
    Keep/drop units based on the keyword/regex given in the constructor.
    Filtering is done based on the unit ids.
    Whether to keep or drop is based on the keep_matches argument.


    Args:
        pattern (Union[str, Pattern]): The regex pattern to match against the unit ids.
        field (str): The field to apply the filter.
        reset_index (bool, optional): If True, it will reset_index the unit index of the time series.
        keep_matches (bool, optional): If True, units matching the pattern will be kept.
    """

    def __init__(
        self,
        pattern: Union[str, Pattern],
        field: str,
        reset_index: bool = True,
        keep_matches: bool = True,
    ):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.keep_matches = keep_matches
        super().__init__(self._generate_unit_mask, field, reset_index)

    def _generate_unit_mask(self, units: ArrayDict) -> np.ndarray:
        unit_mask = np.array([bool(self.pattern.search(uid)) for uid in units.id])
        if not self.keep_matches:
            unit_mask = ~unit_mask
        return unit_mask
