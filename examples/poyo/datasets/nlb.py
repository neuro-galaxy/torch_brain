from copy import deepcopy
from brainsets.datasets import PeiPandarinathNLB2021
from temporaldata import Data

from .wrapper import PoyoReadoutConfig


class PoyoNLBDataset(PeiPandarinathNLB2021):
    dim_target = 2

    def __init__(self, root, transform=None, **kwargs):
        super().__init__(
            root,
            recording_ids=["jenkins_maze_train"],
            transform=transform,
            **kwargs,
        )

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
            normalize_std=100.0,
            eval_interval="nlb_eval_intervals",
        )
