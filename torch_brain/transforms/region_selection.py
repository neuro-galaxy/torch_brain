import logging
from typing import List, Optional

import numpy as np

from temporaldata import Data, IrregularTimeSeries


class RandomRegionSelection:
    r"""Augmentation that randomly selects one region from the available regions in the data.
    
    This transform assumes that the data has a `units` object with a `region` attribute.
    It works for :class:`IrregularTimeSeries` data, keeping only spikes from units in the selected region.

    Args:
        field (str, optional): Field to apply the region selection. Defaults to "spikes".
        exclude_regions (List[str], optional): List of regions to exclude from selection.
            Defaults to ["void"].
        min_units (int, optional): Minimum number of units required for a region to be selected.
            Defaults to 1.
        seed (int, optional): Seed for the random number generator.
    """

    def __init__(
        self, 
        field: str = "spikes", 
        exclude_regions: Optional[List[str]] = None,
        min_units: int = 1,
        reset_index: bool = True,
        seed: Optional[int] = None
    ):
        self.field = field
        self.reset_index = reset_index
        self.exclude_regions = exclude_regions if exclude_regions is not None else ["void"]
        self.min_units = min_units
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, data: Data):
        # get regions from data
        regions = data.units.region
        unique_regions, region_counts = np.unique(regions, return_counts=True)
        
        # filter out excluded regions and regions with insufficient units
        available_regions = []
        for region, count in zip(unique_regions, region_counts):
            if region not in self.exclude_regions and count >= self.min_units:
                available_regions.append(region)
        
        if not available_regions:
            raise ValueError(
                f"No regions have at least {self.min_units} units after excluding {self.exclude_regions}. "
                f"Available regions and their unit counts: {list(zip(unique_regions, region_counts))}"
            )
        
        # randomly select one region
        selected_region = self.rng.choice(available_regions)
        logging.info(f"Selected region: {selected_region}")
        
        # create mask for units in the selected region
        unit_mask = regions == selected_region
        
        if self.reset_index:
            data.units = data.units.select_by_mask(unit_mask)

        target_obj = getattr(data, self.field)
        
        if not isinstance(target_obj, IrregularTimeSeries):
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}. Only IrregularTimeSeries is supported.")
        
        # make a mask to select spikes that are from units in the selected region
        spike_mask = unit_mask[target_obj.unit_index]

        # using lazy masking, we will apply the mask for all attributes from spikes
        # and units.
        setattr(data, self.field, target_obj.select_by_mask(spike_mask))

        if self.reset_index:
            # relabel unit indices to be consecutive
            relabel_map = np.zeros(len(regions), dtype=int)
            relabel_map[unit_mask] = np.arange(unit_mask.sum())

            target_obj = getattr(data, self.field)
            target_obj.unit_index = relabel_map[target_obj.unit_index]

        return data
