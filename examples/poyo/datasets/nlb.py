from typing import Callable, Literal
import numpy as np
from torch_brain.dataset import DatasetIndex
from torch_brain.data.collate import pad8
from brainsets.datasets import PeiPandarinathNLB2021
from temporaldata import Data


class PoyoNLBDataset(PeiPandarinathNLB2021):
    dim_target = 2
    tokenizer: Callable

    def __init__(self, root, transform=None, **kwargs):
        super().__init__(
            root,
            recording_ids=["jenkins_maze_train"],
            transform=transform,
            **kwargs,
        )

    def get_sampling_intervals(self, split: Literal["train", "valid", "test"]):
        ans = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            if split == "train":
                ans[rid] = rec.train_domain
            else:
                ans[rid] = rec.nlb_eval_intervals
                if split == "valid":
                    ans[rid] = ans[rid] & rec.valid_domain
                else:
                    ans[rid] = ans[rid] & rec.test_domain

        return ans

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        # Prepare encoder input
        X = self.tokenizer(data)

        # Prepare target
        timestamps = data.hand.timestamps
        values = data.hand.vel / 100.0  # To keep values in [-1, +1] approx.

        Y = dict(
            timestamps=pad8(timestamps.astype(np.float32)),
            values=pad8(values.astype(np.float32)),
            output_mask=pad8(np.ones(len(timestamps), dtype=bool)),
            session_id=data.session.id,
        )

        return X, Y
