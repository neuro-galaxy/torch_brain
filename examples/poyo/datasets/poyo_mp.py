from typing import Literal, Callable
import numpy as np
from temporaldata import Data
from torch_brain.dataset import DatasetIndex
from torch_brain.data.collate import pad8
from brainsets.datasets import PerichMillerPopulation2018


class PoyoMPDataset(PerichMillerPopulation2018):
    dim_target = 2
    tokenizer: Callable

    def __init__(self, root, transform=None):
        super().__init__(
            root,
            recording_ids=TRAIN_RECORDING_IDS,
            transform=transform,
        )

    def get_sampling_intervals(self, split: Literal["train", "valid", "test"]):
        if split == "train":
            ans = {}
            for rid in self.recording_ids:
                ans[rid] = self.get_recording(rid).train_domain
            return ans
        else:
            eval_intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                if rec.session.id.endswith("center_out_reaching"):
                    intrvl = rec.movement_phases.reach_period
                else:
                    intrvl = rec.movement_phases.random_period

                if split == "valid":
                    intrvl = intrvl & rec.valid_domain
                else:
                    intrvl = intrvl & rec.test_domain

                eval_intervals[rid] = intrvl

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        # Prepare encoder input
        X = self.tokenizer(data)

        # Prepare target
        timestamps = data.cursor.timestamps
        values = data.cursor.vel / 20.0  # To keep values in [-1, +1] approx.

        Y = dict(
            timestamps=pad8(timestamps.astype(np.float32)),
            values=pad8(values.astype(np.float32)),
            output_mask=pad8(np.ones(len(timestamps), dtype=bool)),
            session_id=data.session.id,
        )

        return X, Y


TRAIN_RECORDING_IDS = [
    "c_20131003_center_out_reaching",
    "c_20131022_center_out_reaching",
    "c_20131023_center_out_reaching",
    "c_20131031_center_out_reaching",
    "c_20131101_center_out_reaching",
    "c_20131203_center_out_reaching",
    "c_20131204_center_out_reaching",
    "c_20131219_center_out_reaching",
    "c_20131220_center_out_reaching",
    "c_20150309_center_out_reaching",
    "c_20150311_center_out_reaching",
    "c_20150312_center_out_reaching",
    "c_20150313_center_out_reaching",
    "c_20150319_center_out_reaching",
    "c_20150629_center_out_reaching",
    "c_20150630_center_out_reaching",
    "c_20150701_center_out_reaching",
    "c_20150703_center_out_reaching",
    "c_20150706_center_out_reaching",
    "c_20150707_center_out_reaching",
    "c_20150708_center_out_reaching",
    "c_20150709_center_out_reaching",
    "c_20150710_center_out_reaching",
    "c_20150713_center_out_reaching",
    "c_20150714_center_out_reaching",
    "c_20150715_center_out_reaching",
    "c_20150716_center_out_reaching",
    "c_20151103_center_out_reaching",
    "c_20151104_center_out_reaching",
    "c_20151106_center_out_reaching",
    "c_20151109_center_out_reaching",
    "c_20151110_center_out_reaching",
    "c_20151112_center_out_reaching",
    "c_20151113_center_out_reaching",
    "c_20151116_center_out_reaching",
    "c_20151117_center_out_reaching",
    "c_20151119_center_out_reaching",
    "c_20151120_center_out_reaching",
    "c_20151201_center_out_reaching",
    "c_20160909_center_out_reaching",
    "c_20160912_center_out_reaching",
    "c_20160914_center_out_reaching",
    "c_20160915_center_out_reaching",
    "c_20160919_center_out_reaching",
    "c_20160921_center_out_reaching",
    "c_20160923_center_out_reaching",
    "c_20160929_center_out_reaching",
    "c_20161005_center_out_reaching",
    "c_20161006_center_out_reaching",
    "c_20161007_center_out_reaching",
    "c_20161011_center_out_reaching",
    "c_20161013_center_out_reaching",
    "c_20161021_center_out_reaching",
    "j_20160405_center_out_reaching",
    "j_20160406_center_out_reaching",
    "j_20160407_center_out_reaching",
    "m_20140203_center_out_reaching",
    "m_20140217_center_out_reaching",
    "m_20140218_center_out_reaching",
    "m_20140303_center_out_reaching",
    "m_20140304_center_out_reaching",
    "m_20140306_center_out_reaching",
    "m_20140307_center_out_reaching",
    "m_20140626_center_out_reaching",
    "m_20140627_center_out_reaching",
    "m_20140929_center_out_reaching",
    "m_20141203_center_out_reaching",
    "m_20150511_center_out_reaching",
    "m_20150512_center_out_reaching",
    "m_20150610_center_out_reaching",
    "m_20150611_center_out_reaching",
    "m_20150612_center_out_reaching",
    "m_20150615_center_out_reaching",
    "m_20150616_center_out_reaching",
    "m_20150617_center_out_reaching",
    "m_20150623_center_out_reaching",
    "m_20150625_center_out_reaching",
    "m_20150626_center_out_reaching",
    "c_20131009_random_target_reaching",
    "c_20131010_random_target_reaching",
    "c_20131011_random_target_reaching",
    "c_20131028_random_target_reaching",
    "c_20131029_random_target_reaching",
    "c_20131209_random_target_reaching",
    "c_20131210_random_target_reaching",
    "c_20131212_random_target_reaching",
    "c_20131213_random_target_reaching",
    "c_20131217_random_target_reaching",
    "c_20131218_random_target_reaching",
    "c_20150316_random_target_reaching",
    "c_20150317_random_target_reaching",
    "c_20150318_random_target_reaching",
    "c_20150320_random_target_reaching",
    "m_20140114_random_target_reaching",
    "m_20140115_random_target_reaching",
    "m_20140116_random_target_reaching",
    "m_20140214_random_target_reaching",
    "m_20140221_random_target_reaching",
    "m_20140224_random_target_reaching",
]
