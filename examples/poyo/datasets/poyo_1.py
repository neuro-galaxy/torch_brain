import torchmetrics
from temporaldata import Data

from brainsets.datasets import (
    ChurchlandShenoyNeural2012,
    FlintSlutzkyAccurate2012,
    OdohertySabesNonhuman2017,
)
from torch_brain.dataset import NestedSpikingDataset
from datasets.poyo_mp import PoyoMPDataset


def Poyo1Dataset(root, transform=None):
    ds_mp = PoyoMPDataset(root)
    ds_flint = PoyoFlintDataset(root)
    ds_odoherty = PoyoOdohertyDataset(root)
    ds_churchland = PoyoChurchlandDataset(root)

    ds = NestedSpikingDataset(
        datasets={
            "mp": ds_mp,
            "flint": ds_flint,
            "odoherty": ds_odoherty,
            "churchland": ds_churchland,
        },
        transform=transform,
    )
    return ds


class PoyoChurchlandDataset(ChurchlandShenoyNeural2012):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "normalize_mean": 0.0,
            "normalize_std": 800.0,
            "weights": {
                "movement_phases.hold_period": 0.1,
                "movement_phases.reach_period": 5.0,
                "movement_phases.return_period": 1.0,
                "cursor_outlier_segments": 1.0,
            },
            "metrics": [{"metric": torchmetrics.R2Score()}],
        }
    }

    def get_recording_hook(self, data: Data):
        data.config = self.READOUT_CONFIG
        return super().get_recording_hook(data)


class PoyoOdohertyDataset(OdohertySabesNonhuman2017):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "normalize_mean": 0.0,
            "normalize_std": 200.0,
            "metrics": [{"metric": torchmetrics.R2Score()}],
        }
    }

    def get_recording_hook(self, data: Data):
        data.config = self.READOUT_CONFIG
        return super().get_recording_hook(data)


class PoyoFlintDataset(FlintSlutzkyAccurate2012):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            # we will use the hand velocity for this dataset
            "timestamp_key": "hand.timestamps",
            "value_key": "hand.vel",
            "normalize_mean": 0.0,
            "normalize_std": 0.4,
            "metrics": [{"metric": torchmetrics.R2Score()}],
        }
    }

    def get_recording_hook(self, data: Data):
        data.config = self.READOUT_CONFIG
        return super().get_recording_hook(data)
