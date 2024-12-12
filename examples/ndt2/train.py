import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import wandb
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from model import (
    BhvrDecoder,
    ContextManager,
    Encoder,
    MaeMaskManager,
    NDT2Model,
    SpikesPatchifier,
    SslDecoder,
)
from omegaconf import OmegaConf, open_dict
from torch import optim
from torch.utils.data import DataLoader
from transforms import FilterUnit, Ndt2Tokenizer
import torch.functional as F
from temporaldata import Interval
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from collections import deque

from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything

log = logging.getLogger(__name__)

import torch.nn as nn


class TrainWrapper(L.LightningModule):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.is_ssl = cfg.is_ssl
        self.val_loss_smoothing = False
        if cfg.callbacks.get("monitor_avg", False):
            self.val_loss_smoothing = True
            self.window_size = 10
            self.loss_queue = deque(maxlen=self.window_size)
                
    def moving_average(self, x):
        """
        Computes a simple moving average over the last 'window_size' losses.
        """
        self.loss_queue.append(x.item())
        return sum(self.loss_queue) / len(self.loss_queue) 

    def training_step(self, batch, batch_idx):
        ssl_loss = 0.0
        superv_loss = 0.0

        if self.is_ssl:
            decoder_out = self.model(batch, "ssl")
            ssl_loss = decoder_out["loss"]
            self.log("train_shuffle_infill_loss", decoder_out["loss"])
        else:
            decoder_out = self.model(batch, "bhv")
            superv_loss = decoder_out["loss"]
            self.log("train_kinematic_decoding_loss", decoder_out["loss"])

            task = self.cfg.model.bhv_decoder.get("task", "regression")
            if task == "regression":
                self.log("train_kinematic_r2", decoder_out["r2"].mean())
            elif task == "classification":
                self.log(
                    f"train_acc",
                    decoder_out["acc"].mean(),
                    add_dataloader_idx=False,
                )
                self.log(
                    f"train_balanced_acc",
                    decoder_out["balanced_acc"].mean(),
                    add_dataloader_idx=False,
                )

        loss = ssl_loss + superv_loss
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

    def on_save_checkpoint(self, ckpt):
        ckpt["context_manager_state_dict"] = self.model.ctx_manager.state_dict()
        ckpt["spikes_patchifier_state_dict"] = self.model.spikes_patchifier.state_dict()
        ckpt["encoder_state_dict"] = self.model.encoder.state_dict()
        ckpt["decoder_state_dict"] = self.model.decoder.state_dict()


class DataModule(L.LightningDataModule):
    def __init__(
        self, cfg, tokenizer: Ndt2Tokenizer, is_ssl: bool = True, unsorted: bool = True
    ):
        super().__init__()

        self.cfg = cfg
        self.is_ssl = is_ssl
        self.dataset_cfg = cfg.dataset

        if cfg.keep_M1_units:
            keep_M1_unit = FilterUnit("/M1", keep=True)
            self.transforms = Compose([keep_M1_unit, tokenizer])
        else:
            self.transforms = tokenizer

    def setup(self, stage=None):
        cfg = self.cfg

        #  Do not use split for dataset because is handle at sampler level
        self.dataset = Dataset(
            root=cfg.data_root,
            split=None,
            config=self.dataset_cfg,
            transform=self.transforms,
        )

        if not cfg.get("custom_ndt2_data_spliter", True):

            self.train_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="train",
                transform=self.transforms,
            )
            self.train_intervals = self.train_dataset.get_sampling_intervals()

            self.val_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="valid",
                transform=self.transforms,
            )
            self.val_intervals = self.val_dataset.get_sampling_intervals()

            self.test_dataset = Dataset(
                root=cfg.data_root,
                config=cfg.dataset,
                split="test",
                transform=self.transforms,
            )

            self.eval_intervals = self.test_dataset.get_sampling_intervals()

        else:
            self.dataset.disable_data_leakage_check()
            self.train_intervals: Dict[str, List[Tuple[float, float]]]
            self.val_intervals: Dict[str, List[Tuple[float, float]]]
            self.eval_intervals: Optional[Dict[str, List[Tuple[float, float]]]]
            intervals = self.ndt2_custom_sampling_intervals()
            self.train_intervals, self.val_intervals, self.eval_intervals = intervals

    def get_ctx_vocab(self, ctx_keys):
        return {k: getattr(self.dataset, f"get_{k}_ids")() for k in ctx_keys}

    def train_dataloader(self):
        cfg = self.cfg
        train_sampler = RandomFixedWindowSampler(
            interval_dict=self.train_intervals,
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
            interval_dict=self.val_intervals,
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
            interval_dict=self.eval_intervals,
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

    # The next function are utils for ndt2_custom_sampling_intervals
    def sort_sessions(self, res):
        ind = np.argsort([int(e.split("-")[1]) for e in res])
        return [res[i] for i in ind]

    def ndt2_eval_split(self, ses_keys):
        cfg = self.cfg
        nb_sessions = len(ses_keys)
        df = pd.DataFrame([0] * nb_sessions)
        eval_subset = df.sample(frac=cfg.eval_ratio, random_state=cfg.eval_seed)
        eval_keys = [ses_keys[i] for i in eval_subset.index]
        non_eval_keys = [ses_keys[i] for i in df.index.difference(eval_subset.index)]
        return self.sort_sessions(eval_keys), self.sort_sessions(non_eval_keys)

    def ndt2_limit_per_session(self, ses_keys):
        cfg = self.cfg
        nb_sessions = len(ses_keys)
        df = pd.DataFrame([0] * nb_sessions)
        subset = df.sample(cfg.limit_per_eval_session)
        ses_keys = [ses_keys[i] for i in subset.index]
        return self.sort_sessions(ses_keys)

    def ndt2_custom_sampling_intervals(self) -> Tuple[Dict, Dict]:
        """
        Custom sampling intervals for NDT2.
        It splits the dataset into training and validation sets.
        Note: Used at the sampling level and not at the session level.
        This is because ndt2 split at the dataset object level and not at session level.
        """
        ses_keys = []
        dataset = self.dataset
        ctx_time = self.cfg.ctx_time
        train_ratio = self.cfg.train_ratio
        seed = self.cfg.split_seed

        for ses_id, ses in dataset._data_objects.items():
            nb_trials = int(ses.domain.end[-1] - ses.domain.start[0])
            for i in range(nb_trials):
                ses_keys.append(f"{ses_id}-{i}")

        if self.cfg.get("is_eval", False):
            ses_keys = self.sort_sessions(ses_keys)
            eval_keys, ses_keys = self.ndt2_eval_split(ses_keys)
            ses_keys = self.ndt2_limit_per_session(ses_keys)

        L.seed_everything(seed)
        np.random.shuffle(ses_keys)
        tv_cut = int(train_ratio * len(ses_keys))
        train_keys, val_keys = ses_keys[:tv_cut], ses_keys[tv_cut:]

        def get_dict(keys):
            d = defaultdict(list)
            for k in keys:
                # ses_id, trial = k.split("-")
                trial = k.split("-")[-1]
                ses_id = "-".join(k.split("-")[:-1])
                ses = dataset._data_objects[ses_id]
                ses_start = ses.domain.start[0]
                offset = ctx_time * int(trial)
                start = ses_start + offset
                end = start + ctx_time
                d[ses_id].append((start, end))
            return dict(d)

        train_sampling_intervals = get_dict(train_keys)
        val_sampling_intervals = get_dict(val_keys)

        # val will be deterministic and need to be sorted
        for v in val_sampling_intervals.values():
            v.sort()
        val_sampling_intervals = dict(sorted(val_sampling_intervals.items()))

        # TODO this is very dirty code should be cleaned
        def list_to_inter(l):
            start = np.array([e[0] for e in l])
            end = np.array([e[1] for e in l])
            return Interval(start, end)

        def to_inter(d):
            return {k: list_to_inter(v) for k, v in d.items()}

        train_sampling_intervals = to_inter(train_sampling_intervals)
        val_sampling_intervals = to_inter(val_sampling_intervals)

        eval_sampling_intervals = None
        if self.cfg.get("is_eval", False):
            eval_sampling_intervals = get_dict(eval_keys)
            eval_sampling_intervals = to_inter(eval_sampling_intervals)

        return train_sampling_intervals, val_sampling_intervals, eval_sampling_intervals


def set_wandb(cfg, log) -> Optional[WandbLogger]:
    if not cfg.wandb.enable:
        return None

    # Initialize wandb before anything else
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

    return wandb_logger


def get_ckpt(cfg):
    if cfg.get("fragment_checkpoint"):
        ses = cfg.dataset[0].selection[0]["sessions"][0]
        checkpoint_path = f"{cfg.checkpoint_path}{cfg.checkpoint_prefix}-{ses}.ckpt"
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.load(cfg.checkpoint_path)
    return ckpt


def run_training(cfg):
    L.seed_everything(cfg.seed)
    seed_everything(cfg.seed)

    if cfg.fast_dev_run:
        cfg.wandb.enable = False
        cfg.num_workers = 0

    with open_dict(cfg):
        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        log.info(f"Number of GPUs: {num_gpus}")
        log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    wandb_logger = set_wandb(cfg, log)
    dim = cfg.model.dim

    # Mask manager (for MAE SSL)
    mae_mask_manager = None
    if cfg.is_ssl:
        mae_mask_manager = MaeMaskManager(cfg.mask_ratio)

    # Context manager
    ctx_manager = ContextManager(dim, cfg.ctx_keys)

    # Spikes patchifier
    spikes_patchifier = SpikesPatchifier(dim, cfg.patch_size)

    # Encoder
    encoder = Encoder(
        dim=dim,
        max_time_patches=cfg.model.max_time_patches,
        max_space_patches=cfg.model.max_space_patches,
        **cfg.model.encoder,
    )

    # Decoder
    if cfg.is_ssl:
        decoder = SslDecoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            patch_size=cfg.patch_size,
            **cfg.model.predictor,
        )
    else:
        decoder = BhvrDecoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            bin_time=cfg.bin_time,
            **cfg.model.bhv_decoder,
        )

    # Model wrap everithing
    model = NDT2Model(
        mae_mask_manager, ctx_manager, spikes_patchifier, encoder, decoder
    )

    # Train wrapper
    train_wrapper = TrainWrapper(cfg, model)

    # Tokenizer
    bhvr_dim = None
    if not cfg.is_ssl:
        bhvr_dim = cfg.model.bhv_decoder["behavior_dim"]

    # Load from checkpoint
    if cfg.get("load_from_checkpoint", False):
        ckpt = get_ckpt(cfg)
        model.ctx_manager.load_state_dict(ckpt["context_manager_state_dict"])
        model.spikes_patchifier.load_state_dict(ckpt["spikes_patchifier_state_dict"])
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        if not cfg.get("new_decoder", False):
            model.decoder.load_state_dict(ckpt["decoder_state_dict"])

    ctx_tokenizer = ctx_manager.get_ctx_tokenizer()
    tokenizer = Ndt2Tokenizer(
        ctx_time=cfg.ctx_time,
        bin_time=cfg.bin_time,
        patch_size=cfg.patch_size,
        pad_val=cfg.pad_val,
        ctx_tokenizer=ctx_tokenizer,
        unsorted=cfg.unsorted,
        is_ssl=cfg.is_ssl,
        bhvr_key=cfg.get("bhvr_key"),
        bhvr_dim=bhvr_dim,
        ibl_binning=cfg.get("ibl_binning", False),
    )

    # Set up data module
    data_module = DataModule(cfg, tokenizer, cfg.is_ssl)
    data_module.setup()

    if cfg.get("load_from_checkpoint", False):
        # Register new context
        ctx_manager.extend_vocab(data_module.get_ctx_vocab(ctx_manager.keys))
    else:
        # Register context
        ctx_manager.init_vocab(data_module.get_ctx_vocab(ctx_manager.keys))

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.callbacks.checkpoint:
        monitor = "val_loss"
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
        fast_dev_run=cfg.fast_dev_run,
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
