import numpy as np
from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries


class FilterUnit:
    r"""
    Drop (or keep) units whose ids has the `keyword` given in the constructor.
    By default drop but can keep if `keep` is set to True
    """

    def __init__(self, keyword: str = "unsorted", field="spikes", keep: bool = False):
        self.keyword = keyword
        self.field = field
        self.keep = keep

    def __call__(self, data: Data) -> Data:
        unit_ids = data.units.id
        num_units = len(unit_ids)

        no_keywork_unit = np.char.find(unit_ids, self.keyword) == -1
        if self.keep == True:
            keep_unit_mask = ~no_keywork_unit
        else:
            keep_unit_mask = no_keywork_unit

        if keep_unit_mask.all():
            return data

        keep_indices = np.where(keep_unit_mask)[0]
        data.units = data.units.select_by_mask(keep_unit_mask)

        nested_attr = self.field.split(".")
        target_obj = getattr(data, nested_attr[0])
        if isinstance(target_obj, IrregularTimeSeries):
            # make a mask to select spikes that are from the units we want to keep
            spike_mask = np.isin(target_obj.unit_index, keep_indices)

            # using lazy masking, we will apply the mask for all attributes from spikes and units
            setattr(data, self.field, target_obj.select_by_mask(spike_mask))

            relabel_map = np.zeros(num_units, dtype=int)
            relabel_map[keep_unit_mask] = np.arange(keep_unit_mask.sum())

            target_obj = getattr(data, self.field)
            target_obj.unit_index = relabel_map[target_obj.unit_index]
        elif isinstance(target_obj, RegularTimeSeries):
            assert len(nested_attr) == 2
            setattr(
                target_obj,
                nested_attr[1],
                getattr(target_obj, nested_attr[1])[:, keep_unit_mask],
            )
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")

        return data
