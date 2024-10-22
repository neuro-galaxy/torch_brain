from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transforms import FilterUnit, NDT2Tokenizer
from utils import ndt2_custom_sampling_intervals

from brainsets.taxonomy import decoder_registry
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.transforms import Compose


class DataLoaderGenerator:
    """
    DataLoaderGenerator is responsible for creating and configuring a data loader for training and evaluation.
    Attributes:
        cfg: Configuration object containing various settings.
        dataset_cfg: Configuration specific to the dataset.
        train_wrapper: An instance of TrainWrapper that provides access to the encoder and other components.
        is_ssl: Boolean indicating whether to use self-supervised learning (SSL) mode.
    Methods:
        __init__(self, cfg, dataset_cfg, train_wrapper: TrainWrapper, is_ssl: bool = True):
            Initializes the DataLoaderGenerator with the given configurations and train wrapper.
        __call__(self, split: str) -> DataLoader:
            Creates and returns a DataLoader for the specified split ('train' or 'eval').
    """

    def __init__(
        self, cfg, dataset_cfg, train_wrapper, is_ssl: bool = True, unsorted=True
    ):
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_wrapper = train_wrapper
        self.is_ssl = is_ssl
        self.unsorted = unsorted
        keep_M1_unit = FilterUnit("/M1", keep=True)

        session_tokenizer = train_wrapper.ctx_manager.session_emb.tokenizer
        subject_tokenizer = train_wrapper.ctx_manager.subject_emb.tokenizer

        tokenizer = NDT2Tokenizer(
            ctx_time=cfg.ctx_time,
            bin_time=cfg.bin_time,
            patch_size=cfg.patch_size,
            pad_val=cfg.pad_val,
            decoder_registry=decoder_registry,
            mask_ratio=cfg.mask_ratio,
            session_tokenizer=session_tokenizer,
            subject_tokenizer=subject_tokenizer,
            inc_behavior=not self.is_ssl,
            inc_mask=self.is_ssl,
            unsorted=self.unsorted,
        )

        transforms = Compose([keep_M1_unit, tokenizer])

        # do not use split for dataset because is handle at sampler level
        self.dataset = Dataset(
            root=cfg.data_root,
            split=None,
            include=self.dataset_cfg,
            transform=transforms,
        )
        self.dataset.disable_data_leakage_check()

        self.session_ids: List[str] = self.dataset.get_session_ids()
        self.subject_ids: List[str] = self.dataset.get_subject_ids()

        self.train_intervals: Dict[str, List[Tuple[float, float]]]
        self.eval_intervals: Dict[str, List[Tuple[float, float]]]
        intervals = ndt2_custom_sampling_intervals(
            self.dataset, cfg.ctx_time, cfg.train_ratio, cfg.split_seed
        )
        self.train_intervals, self.eval_intervals = intervals

    def __call__(self, split: str) -> DataLoader:
        """
        Generates a DataLoader for the specified data split.
        Args:
            split (str): The data split to load. Should be either "train" or "eval".
        Returns:
            DataLoader: A DataLoader instance for the specified data split.
        """

        cfg = self.cfg

        if split == "train":
            sampler = RandomFixedWindowSampler(
                interval_dict=self.train_intervals,
                window_length=cfg.ctx_time,
                generator=torch.Generator(),
            )
        else:
            sampler = SequentialFixedWindowSampler(
                interval_dict=self.eval_intervals,
                window_length=cfg.ctx_time,
                drop_short=True,
            )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        return DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )

    def get_vocab(self) -> Dict[str, List[str]]:
        return {
            "session": self.session_ids,
            "subject": self.subject_ids,
        }
