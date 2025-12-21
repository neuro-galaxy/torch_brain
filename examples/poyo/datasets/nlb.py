from typing import Literal
import torchmetrics
from brainsets.datasets import PeiPandarinathNLB2021
from temporaldata import Data

from torch_brain.registry import MODALITY_REGISTRY


def nlb_dataset(root: str):
    ds = PeiPandarinathNLB2021(
        root=root,
        recording_ids=["jenkins_maze_train"],
        transform=NLBReadoutConfigTransform(),
    )
    readout_spec = MODALITY_REGISTRY["cursor_velocity_2d"]
    return ds, readout_spec


class NLBReadoutConfigTransform:
    CONFIG = {
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

    def __call__(self, data: Data) -> Data:
        data.config = self.CONFIG
        return data
