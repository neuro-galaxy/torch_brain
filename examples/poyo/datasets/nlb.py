from typing import Literal
import torchmetrics
import brainsets.datasets as datasets
from temporaldata import Data


class POYONLBDataset(datasets.PeiPandarinathNLB2021):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "timestamp_key": "hand.timestamps",
            "value_key": "hand.vel",
            "normalize_mean": 0.0,
            "normalize_std": 100.0,
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "nlb_eval_intervals",
        }
    }

    def __init__(self, root, transform=None, **kwargs):
        super().__init__(
            root,
            recording_ids=["jenkins_maze_train"],
            transform=transform,
            **kwargs,
        )

    def get_recording_hook(self, data: Data):
        data.config = self.READOUT_CONFIG
        super().get_recording_hook(data)
