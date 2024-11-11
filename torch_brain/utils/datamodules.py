import copy
import logging

import hydra
import lightning
import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowBatchSampler,
    RandomFixedWindowSampler,
)
from torch_brain.models import POYOPlusTokenizer
from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.transforms import Compose


class DataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig, unit_tokenizer, session_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.sequence_length = 1.0
        self.train_dataset = None
        self.val_dataset = None
        self.log = logging.getLogger(__name__)

    def setup(self, stage=None):
        # prepare tokenizer and transforms
        transforms = hydra.utils.instantiate(
            self.cfg.train_transforms, sequence_length=self.sequence_length
        )

        # build tokenizer
        tokenizer = POYOPlusTokenizer(
            self.unit_tokenizer,
            self.session_tokenizer,
            decoder_registry=MODALITIY_REGISTRY,
            latent_step=1 / 8,
            num_latents_per_step=self.cfg.model.num_latents,
        )

        transform = Compose([*transforms, tokenizer])

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=transform,
        )
        self.train_dataset.disable_data_leakage_check()

        val_tokenizer = copy.copy(tokenizer)
        val_tokenizer.eval = True
        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=val_tokenizer,
        )
        self.val_dataset.disable_data_leakage_check()

        test_tokenizer = copy.copy(tokenizer)
        test_tokenizer.eval = True
        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=test_tokenizer,
        )
        self.test_dataset.disable_data_leakage_check()

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def get_multitask_readout_registry(self):
        config_dict = self.train_dataset.get_recording_config_dict()

        custum_readout_registry = {}
        for recording_id in config_dict.keys():
            config = config_dict[recording_id]
            multitask_readout = config["multitask_readout"]

            for readout_config in multitask_readout:
                readout_id = readout_config["readout_id"]
                if readout_id not in MODALITIY_REGISTRY:
                    raise ValueError(
                        f"Readout {readout_id} not found in modality registry, please register it "
                        "using torch_brain.register_modality()"
                    )
                custum_readout_registry[readout_id] = MODALITIY_REGISTRY[readout_id]
        return custum_readout_registry

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            interval_dict=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(
            f"Training on {len(self.train_dataset.get_session_ids())} sessions"
        )

        return train_loader

    def val_dataloader(self):
        val_sampler = DistributedStitchingFixedWindowBatchSampler(
            interval_dict=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=self.cfg.eval_batch_size or self.cfg.batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        self.val_sequence_index = val_sampler.sequence_index

        return val_loader

    def test_dataloader(self):
        test_sampler = DistributedStitchingFixedWindowBatchSampler(
            interval_dict=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=self.cfg.eval_batch_size or self.cfg.batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate,
            num_workers=0,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader
