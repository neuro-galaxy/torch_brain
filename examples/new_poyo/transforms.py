from typing import Literal
import torchmetrics
from temporaldata import Data


class MPReadoutConfigTransform:
    CO_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "normalize_mean": 0.0,
            "normalize_std": 20.0,
            "weights": {
                "movement_phases.random_period": 1.0,
                "movement_phases.hold_period": 0.1,
                "movement_phases.reach_period": 5.0,
                "movement_phases.return_period": 1.0,
                "movement_phases.invalid": 0.1,
                "cursor_outlier_segments": 0.0,
            },
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "movement_phases.reach_period",
        }
    }

    RT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "normalize_mean": 0.0,
            "normalize_std": 20.0,
            "weights": {
                "movement_phases.random_period": 1.0,
                "movement_phases.hold_period": 0.1,
                "cursor_outlier_segments": 0.0,
            },
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "movement_phases.random_period",
        }
    }

    def __call__(self, data: Data) -> Data:
        if data.session.id.endswith("center_out_reaching"):
            data.config = self.CO_CONFIG
        elif data.session.id.endswith("random_target_reaching"):
            data.config = self.RT_CONFIG
        return data


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
