from typing import Callable
from temporaldata import Data

from brainsets.datasets import (
    ChurchlandShenoyNeural2012,
    FlintSlutzkyAccurate2012,
    OdohertySabesNonhuman2017,
)
from torch_brain.dataset import SpikingDatasetMixin, NestedDataset
from datasets.poyo_mp import PoyoMPDataset
from datasets.wrapper import PoyoReadoutConfig


class Poyo1Dataset(SpikingDatasetMixin, NestedDataset):
    dim_target = 2

    def __init__(self, root, transform: Callable | None = None):
        ds_mp = PoyoMPDataset(root)
        ds_flint = PoyoFlintDataset(root)
        ds_odoherty = PoyoOdohertyDataset(root)
        ds_churchland = PoyoChurchlandDataset(root)
        super().__init__(
            datasets={
                "mp": ds_mp,
                "flint": ds_flint,
                "odoherty": ds_odoherty,
                "churchland": ds_churchland,
            },
            transform=transform,
        )


class PoyoChurchlandDataset(ChurchlandShenoyNeural2012):
    dim_target = 2

    def get_recording_hook(self, data: Data):
        """Adds ``readout_config`` attribute to the data object.

        This will be used by the ``PoyoDatasetWrapper`` to apply normalization
        and extract loss weights.
        """
        super().get_recording_hook(data)
        data.readout_config = PoyoReadoutConfig(
            value_key="cursor.vel",
            timestamp_key="cursor.timestamps",
            normalize_mean=0.0,
            normalize_std=800.0,
            weights={
                "movement_phases.hold_period": 0.1,
                "movement_phases.reach_period": 5.0,
                "movement_phases.return_period": 1.0,
                "cursor_outlier_segments": 1.0,
            },
        )


class PoyoOdohertyDataset(OdohertySabesNonhuman2017):
    dim_target = 2

    def get_recording_hook(self, data: Data):
        """Adds ``readout_config`` attribute to the data object.

        This will be used by the ``PoyoDatasetWrapper`` to apply normalization
        and extract loss weights.
        """
        super().get_recording_hook(data)
        data.readout_config = PoyoReadoutConfig(
            value_key="cursor.vel",
            timestamp_key="cursor.timestamps",
            normalize_mean=0.0,
            normalize_std=200.0,
        )


class PoyoFlintDataset(FlintSlutzkyAccurate2012):
    dim_target = 2

    def get_recording_hook(self, data: Data):
        """Adds ``readout_config`` attribute to the data object.

        This will be used by the ``PoyoDatasetWrapper`` to apply normalization
        and extract loss weights.
        """
        super().get_recording_hook(data)
        data.readout_config = PoyoReadoutConfig(
            timestamp_key="hand.timestamps",
            value_key="hand.vel",
            normalize_mean=0.0,
            normalize_std=0.4,
        )
