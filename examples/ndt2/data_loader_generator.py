from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transforms import DropUnit, NDT2Tokenizer
from utils import custom_sampling_intervals

from brainsets.taxonomy import decoder_registry
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.transforms import Compose


class DataLoaderGenerator:
    def __init__(self, cfg, dataset_cfg, train_wrapper, is_ssl):
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_wrapper = train_wrapper
        self.is_ssl = is_ssl

        session_tokenizer = self.train_wrapper.patchifier.sess_emb.tokenizer

        drop_unit = DropUnit(keyword="sorted")
        tokenizer = NDT2Tokenizer(
            ctx_time=cfg.ctx_time,
            bin_time=cfg.bin_time,
            patch_size=cfg.patch_size,
            decoder_registry=decoder_registry,
            mask_ratio=cfg.mask_ratio,
            pad_val=cfg.pad_val,
            sess_emb_space_idx=cfg.model.max_space_patches - 1,
            sess_emb_time_idx=cfg.model.max_time_patches - 1,
            session_tokenizer=session_tokenizer,
            inc_behavior=not self.is_ssl,
            inc_mask=self.is_ssl,
        )
        transforms = Compose([drop_unit, tokenizer])

        self.dataset = Dataset(
            root=cfg.data_root,
            split="train",
            include=self.dataset_cfg,
            transform=transforms,
        )

        self.session_ids = self.dataset.get_session_ids()
        res = custom_sampling_intervals(
            self.dataset, cfg.ctx_time, train_ratio=0.8, seed=0
        )
        self.train_intervals = res[0]
        self.eval_intervals = res[1]

    def __call__(self, split: str) -> DataLoader:
        cfg = self.cfg
        if split == "train":
            train_intervals = self.train_intervals
        else:
            train_intervals = self.eval_intervals

        sampler = SequentialFixedWindowSampler(
            interval_dict=train_intervals,
            window_length=cfg.ctx_time,
            step=cfg.ctx_time,
            drop_short=True,
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            drop_last=split == "train",
        )
        return loader

    def get_session_ids(self) -> List[str]:
        assert self.session_ids is not None
        return self.session_ids
