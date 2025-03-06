import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch import optim
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.models import NDT2
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything


class NDT2TrainWrapper(L.LightningModule):
    def __init__(self, cfg, model: NDT2):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.is_ssl = cfg.is_ssl
        self.val_loss_smoothing = False
        if cfg.callbacks.get("monitor_avg", False):
            self.val_loss_smoothing = True
            self.window_size = 10
            self.loss_queue = deque(maxlen=self.window_size)

    def configure_optimizers(self):
        cfg = self.cfg.optimizer

        params = self.parameters()
        if cfg.get("accelerate_factor", 1) > 1:
            params = self.split_params(self.named_parameters())
        if cfg.get("freeze_encoder", False):
            for _, param in self.model.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.model.spikes_patchifier.named_parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if not cfg.scheduler:
            return {"optimizer": optimizer}

        linearLR = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.start_factor, total_iters=cfg.warmup_steps
        )
        cosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.decay_steps, eta_min=cfg.lr_min
        )
        scheduler = optim.lr_scheduler.ChainedScheduler([linearLR, cosineAnnealingLR])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    def training_step(self, batch, batch_idx):
        model_inputs = batch["model_inputs"]
        decoder_out = self.model(
            model_inputs["units_bincount"],
            model_inputs["time_idx"],
            model_inputs["space_idx"],
            model_inputs["input_mask"],
            model_inputs["encoder_attn_mask"],
            model_inputs["session_idx"],
            model_inputs["subject_idx"],
            model_inputs["task_idx"],
        )

        loss = decoder_out["loss"]
        if self.is_ssl:
            self.log("train_shuffle_infill_loss", loss)
        else:
            self.log("train_kinematic_decoding_loss", loss)

        #     task = self.cfg.model.bhv_decoder.get("task", "regression")
        #     if task == "regression":
        #         self.log("train_kinematic_r2", decoder_out["r2"].mean())
        #     elif task == "classification":
        #         self.log(
        #             f"train_acc",
        #             decoder_out["acc"].mean(),
        #             add_dataloader_idx=False,
        #         )
        #         self.log(
        #             f"train_balanced_acc",
        #             decoder_out["balanced_acc"].mean(),
        #             add_dataloader_idx=False,
        #         )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ssl_loss = 0.0
        superv_loss = 0.0

        prefix = "val_"
        if dataloader_idx == 1:
            prefix = "eval_"

        if self.is_ssl:
            decoder_out = self.model(batch, "ssl")
            ssl_loss = decoder_out["loss"]
            self.log(
                f"{prefix}shuffle_infill_loss",
                decoder_out["loss"],
                add_dataloader_idx=False,
            )

        else:
            decoder_out = self.model(batch, "bhv")
            superv_loss = decoder_out["loss"]
            self.log(
                f"{prefix}kinematic_decoding_loss",
                decoder_out["loss"],
                add_dataloader_idx=False,
            )

            task = self.cfg.model.bhv_decoder.get("task", "regression")
            if task == "regression":
                self.log(
                    f"{prefix}kinematic_r2",
                    decoder_out["r2"].mean(),
                    add_dataloader_idx=False,
                )
            elif task == "classification":
                self.log(
                    f"{prefix}acc",
                    decoder_out["acc"].mean(),
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{prefix}balanced_acc",
                    decoder_out["balanced_acc"].mean(),
                    add_dataloader_idx=False,
                )
        loss = ssl_loss + superv_loss
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if self.val_loss_smoothing:
            avg_loss = self.moving_average(loss)
            self.log(
                f"{prefix}loss_avg",
                avg_loss,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return loss

    # TODO not being used but could be implemented
    # def test_step(self, batch, batch_idx):

    # TODO move somewhere else
    def split_params(self, params):
        cfg = self.cfg.optimizer
        accel_flag = lambda n: "decoder" in n or "ctx_manager" in n and "_emb" in n

        accelerate_params = [p for n, p in params if accel_flag(n)]
        regular_params = [p for n, p in params if not accel_flag(n)]
        return [
            {
                "params": accelerate_params,
                "lr": cfg.lr * cfg.accelerate_factor,
            },
            {
                "params": regular_params,
                "lr": cfg.lr,
            },
        ]

    # TODO move somewhere else
    # def on_save_checkpoint(self, ckpt):
    #     ckpt["context_manager_state_dict"] = self.model.ctx_manager.state_dict()
    #     ckpt["spikes_patchifier_state_dict"] = self.model.spikes_patchifier.state_dict()
    #     ckpt["encoder_state_dict"] = self.model.encoder.state_dict()
    #     ckpt["decoder_state_dict"] = self.model.decoder.state_dict()

    # TODO move somewhere else
    def moving_average(self, x):
        """
        Computes a simple moving average over the last 'window_size' losses.
        """
        self.loss_queue.append(x.item())
        return sum(self.loss_queue) / len(self.loss_queue)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return self.model.mae_masking(batch)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg, is_ssl: bool = True, unsorted: bool = True):
        super().__init__()

        self.cfg = cfg
        self.is_ssl = is_ssl
        self.dataset_cfg = cfg.dataset

    # TODO Use FilterUnit("/M1", keep=True)
    # train_transforms:
    #   - _target_: torch_brain.transforms.UnitDropout
    #     max_units: 1000
    #     min_units: 60
    #     mode_units: 300
    #     peak: 4

    def setup_dataset_and_link_model(self, model: NDT2):
        cfg = self.cfg

        #  Do not use split for dataset because is handle at sampler level
        transforms = []
        if cfg.get("transforms"):
            transforms = hydra.utils.instantiate(self.cfg.transforms)

        self.dataset = Dataset(
            root=cfg.data_root,
            split=None,
            config=self.dataset_cfg,
            transform=Compose([*transforms, model.tokenize]),
        )

        # TODO change this design here the call to a train/eval datset is made just to create the datasplit
        if not cfg.get("custom_ndt2_data_spliter", True):
            train_transforms = []
            if cfg.get("train_transforms"):
                train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
            self.train_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="train",
                transform=Compose([*train_transforms, model.tokenize]),
            )

            self.train_intervals = self.train_dataset.get_sampling_intervals()

            self._init_model_vocab(model)

            eval_transforms = []
            if cfg.get("eval_transforms"):
                eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

            self.val_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="valid",
                transform=Compose([*eval_transforms, model.tokenize]),
            )
            self.val_intervals = self.val_dataset.get_sampling_intervals()

            self.test_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="test",
                transform=Compose([*eval_transforms, model.tokenize]),
            )
            self.eval_intervals = self.test_dataset.get_sampling_intervals()

        # else:
        #     self.dataset.disable_data_leakage_check()
        #     self.train_intervals: Dict[str, List[Tuple[float, float]]]
        #     self.val_intervals: Dict[str, List[Tuple[float, float]]]
        #     self.eval_intervals: Optional[Dict[str, List[Tuple[float, float]]]]
        #     intervals = self.ndt2_custom_sampling_intervals()
        #     self.train_intervals, self.val_intervals, self.eval_intervals = intervals

    def _init_model_vocab(self, model: NDT2):
        if model.session_emb is not None:
            model.session_emb.initialize_vocab(self.get_session_ids())
        if model.subject_emb is not None:
            model.subject_emb.initialize_vocab(self.get_subject_ids())
        if model.task_emb is not None:
            model.task_emb.initialize_vocab(self.get_task_ids())

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_subject_ids(self):
        return self.train_dataset.get_subject_ids()

    def get_task_ids(self):
        return self.train_dataset.get_brainset_ids()

    def train_dataloader(self):
        cfg = self.cfg
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_intervals,
            window_length=cfg.ctx_time,
            generator=torch.Generator(),
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )

        return train_loader

    def val_dataloader(self):
        cfg = self.cfg

        val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.val_intervals,
            window_length=cfg.ctx_time,
            drop_short=True,
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        val_loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=val_sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )
        if self.eval_intervals is None:
            return val_loader

        eval_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.eval_intervals,
            window_length=cfg.ctx_time,
            drop_short=True,
        )
        eval_loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=eval_sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )

        return [val_loader, eval_loader]

    def test_dataloader(self):
        return None


def get_ckpt(cfg):
    if cfg.get("fragment_checkpoint"):
        ses = cfg.dataset[0].selection[0]["sessions"][0]
        checkpoint_path = f"{cfg.checkpoint_path}{cfg.checkpoint_prefix}-{ses}.ckpt"
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.load(cfg.checkpoint_path)
    return ckpt


def run_training(cfg):
    # fix random seed, skipped if cfg.seed is None
    L.seed_everything(cfg.seed)
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    log.info("NDT2!")
    wandb_logger = None
    if cfg.wandb.enable:
        # TODO can be reworked
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        wandb_logger = WandbLogger(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            save_dir=cfg.log_dir,
            log_model=False,
        )
        log.info(f"Using wandb logger: {wandb_logger.version}")

    # TODO check if needed
    with open_dict(cfg):
        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        log.info(f"Number of GPUs: {num_gpus}")
        log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    dim = cfg.model.dim

    # Set up data module
    data_module = DataModule(cfg, cfg.is_ssl)

    model = NDT2(
        is_ssl=cfg.is_ssl,
        dim=dim,
        units_per_patch=cfg.units_per_patch,
        max_units_bincount=cfg.max_units_bincount,
        spike_pad=cfg.spike_pad,
        pad_value=cfg.pad_val,
        max_time_patches=cfg.model.max_time_patches,
        max_space_patches=cfg.model.max_space_patches,
        bin_time=cfg.bin_time,
        ctx_time=cfg.ctx_time,
        mask_ratio=cfg.mask_ratio,
        tokenize_session=cfg.tokenize_session,
        tokenize_subject=cfg.tokenize_subject,
        tokenize_task=cfg.tokenize_task,
        depth=cfg.model.encoder.depth,
        heads=cfg.model.encoder.heads,
        dropout=cfg.model.encoder.dropout,
        ffn_mult=cfg.model.encoder.ffn_mult,
    )

    # # Load from checkpoint TODO update
    # if cfg.get("load_from_checkpoint", False):
    #     ckpt = get_ckpt(cfg)
    #     model.ctx_manager.load_state_dict(ckpt["context_manager_state_dict"])
    #     model.spikes_patchifier.load_state_dict(ckpt["spikes_patchifier_state_dict"])
    #     model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    #     if not cfg.get("new_decoder", False):
    #         model.decoder.load_state_dict(ckpt["decoder_state_dict"])

    # TODO add an link_model
    data_module.setup_dataset_and_link_model(model)

    # if cfg.get("load_from_checkpoint", False):
    #     # Register new context
    #     ctx_manager.extend_vocab(data_module.get_ctx_vocab(ctx_manager.keys))

    # Train wrapper
    train_wrapper = NDT2TrainWrapper(cfg, model)

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    monitor = "val_loss"
    if cfg.callbacks.checkpoint:
        if cfg.callbacks.get("monitor_avg", False):
            monitor = "val_loss_avg"

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.callbacks.checkpoint_path,
            filename=f"{cfg.wandb.run_name}",
            monitor=monitor,
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)

    if cfg.callbacks.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode="min",
                strict=False,
                check_finite=False,
                patience=cfg.callbacks.patience,
            )
        )

    # Set up trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        accelerator="gpu",
        precision=cfg.precision,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        strategy="ddp_find_unused_parameters_true",
    )

    if wandb_logger:
        wandb_logger.watch(train_wrapper, log="all")

    # Train model
    trainer.fit(train_wrapper, data_module)

    # finish wandb
    if wandb_logger:
        wandb_logger.finalize(status="success")
        wandb.finish()


@hydra.main(version_base="1.3", config_path="./ibl_configs", config_name="pretrain")
def main(cfg):
    if cfg.get("fragment_dataset", False):
        run_name = cfg.wandb.run_name
        sessions = cfg.dataset[0].selection[0]["sessions"].copy()
        for ses in sessions:
            cfg.dataset[0].selection[0]["sessions"] = [ses]
            cfg.wandb.run_name = f"{run_name}-{ses}"
            run_training(cfg)

    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
