import logging
from collections import deque
from typing import Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import open_dict
from torch import optim
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.models import NDT2
from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything

logger = logging.getLogger(__name__)

ACCEL_PREFIXES = (
    "model.decoder.",
    "model.head.",
    "model.ctx_embedder.task_emb.weight",
    "model.ctx_embedder.session_emb.weight",
    "model.ctx_embedder.subject_emb.weight",
)

FREEZE_MAP_PREFIXES = {
    "freeze_ctx_embedder": ("model.ctx_embedder"),
    "freeze_encoder": (
        "model.encoder.encoder",
        "model.encoder.time_emb.weight",
        "model.encoder.space_emb.weight",
    ),
    "freeze_all": ("model.encoder", "model.decoder", "model.head"),
}

NEW_DECODER_PREFIXES = ("decoder", "head")


class NDT2TrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg,
        model: NDT2,
        modality_spec: Optional[ModalitySpec] = None,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.is_ssl = cfg.is_ssl

        if cfg.is_ssl:
            self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=True)
        else:
            self.loss_fn = modality_spec.loss_fn

    def configure_optimizers(self):
        cfg = self.cfg.optimizer

        params = self.parameters()

        # TODO discuss the finetuning technique (in the paper end-to-end)
        if cfg.get("accelerate_factor", 1) > 1:
            params = self._split_params()

        freeze_strategy = cfg.get("freeze_strategy", None)
        if freeze_strategy is not None:
            self._freeze_components(freeze_strategy)

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
        model_out = self.model(**model_inputs)

        if self.cfg.is_ssl:
            rates = model_out["output"]
            target = batch["target"]
            extra_units_mask = batch["extra_units_mask"]
            loss = self.loss_fn(rates, target)[extra_units_mask].mean()
        else:
            output = rearrange(model_out["output"], "b t bhvr_dim -> b (t bhvr_dim)")
            target = rearrange(batch["target"], "b t bhvr_dim -> b (t bhvr_dim)")

            loss = self.loss_fn(output, target)

            # TODO manage metrics

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, prefix="val_"):
        model_out = self.model(**batch["model_inputs"])

        if self.cfg.is_ssl:
            rates = model_out["output"]
            target = batch["target"]
            extra_units_mask = batch["extra_units_mask"]
            loss = self.loss_fn(rates, target)[extra_units_mask].mean()
        else:
            output = rearrange(model_out["output"], "b t bhvr_dim -> b (t bhvr_dim)")
            target = rearrange(batch["target"], "b t bhvr_dim -> b (t bhvr_dim)")

            loss = self.loss_fn(output, target)

            # TODO manage metrics

        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, prefix="test_")

    def _freeze_components(self, freeze_strategy):
        if freeze_strategy not in FREEZE_MAP_PREFIXES:
            logger.info(
                f"Dreeze_strategy {freeze_strategy} was not recognized, curetly supporting {FREEZE_MAP_PREFIXES.keys()}"
            )
            return

        freeze_prefixes = FREEZE_MAP_PREFIXES[freeze_strategy]

        frozen_count = 0
        for name, param in self.named_parameters():
            if any(prefix in name for prefix in freeze_prefixes):
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Froze {frozen_count} from prefixes: {freeze_prefixes}")

    def _split_params(self):
        cfg = self.cfg.optimizer
        accel_factor = cfg.get("accelerate_factor", 1.0)

        accelerate_params, regular_params = [], []

        # Iterate through named_parameters of the Wrapper
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if the parameter name contains any of our target prefixes
            if any(prefix in name for prefix in ACCEL_PREFIXES):
                accelerate_params.append(param)
            else:
                regular_params.append(param)

        logger.info(
            f"Param split: {len(accelerate_params)} accelerated, {len(regular_params)} regular"
        )

        # Safety check: ensure we didn't lose any parameters
        total_split = len(accelerate_params) + len(regular_params)
        total_trainable = len([p for p in self.parameters() if p.requires_grad])

        assert (
            total_split == total_trainable
        ), f"Parameter mismatch! Split: {total_split}, Total Trainable: {total_trainable}"

        return [
            {"params": accelerate_params, "lr": cfg.lr * accel_factor},
            {"params": regular_params, "lr": cfg.lr},
        ]


class DataModule(L.LightningDataModule):
    def __init__(self, cfg, is_ssl: bool = True):
        super().__init__()

        self.cfg = cfg
        self.is_ssl = is_ssl

        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        train_ds = hydra.utils.instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            transform=Compose([*train_transforms]),
        )

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)
        eval_ds = hydra.utils.instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            transform=Compose([*eval_transforms]),
        )

        example_recording = train_ds.get_recording(train_ds.recording_ids[0])
        readout_id = example_recording.config["readout"]["readout_id"]
        for recording_id in train_ds.recording_ids:
            recording = train_ds.get_recording(recording_id)
            if readout_id != recording.config["readout"]["readout_id"]:
                raise ValueError(
                    f"Readout ID mismatch: expected '{readout_id}' but got "
                    f"'{recording.config['readout']['readout_id']}' for recording '{recording_id}'."
                    f"POYO only supports a single readout"
                )
        self.readout_spec = MODALITY_REGISTRY[readout_id]

        self.train_dataset = train_ds
        self.eval_dataset = eval_ds

    def link_model(self, model: NDT2):
        self.train_dataset.transform.transforms.append(model.tokenize)
        self.eval_dataset.transform.transforms.append(model.tokenize)

        if self.cfg.get("load_from_checkpoint", False):
            self._extend_model_vocab(model)
        else:
            self._init_model_vocab(model)

    def _init_model_vocab(self, model: NDT2):
        ctx_embedder = model.ctx_embedder
        if ctx_embedder.tokenize_session:
            ctx_embedder.session_emb.initialize_vocab(self.get_session_ids())
        if ctx_embedder.tokenize_subject:
            ctx_embedder.subject_emb.initialize_vocab(self.get_subject_ids())
        if ctx_embedder.tokenize_task:
            ctx_embedder.task_emb.initialize_vocab(self.get_task_ids())

    def _extend_model_vocab(self, model: NDT2):
        ctx_embedder = model.ctx_embedder

        if ctx_embedder.tokenize_session:
            session_emb = ctx_embedder.session_emb
            existing = list(session_emb.vocab.keys()) if session_emb.vocab else []
            new_sessions = [s for s in self.get_session_ids() if s not in existing]
            if len(new_sessions) > 0:
                logger.info(
                    f"Extending session vocab with {len(new_sessions)} new IDs: {new_sessions}"
                )
                ctx_embedder.session_emb.extend_vocab(new_sessions, exist_ok=True)
            else:
                logger.info("Session vocab already includes all session IDs.")

        if ctx_embedder.tokenize_subject:
            subject_emb = ctx_embedder.subject_emb
            existing = list(subject_emb.vocab.keys()) if subject_emb.vocab else []
            new_subjects = [s for s in self.get_subject_ids() if s not in existing]
            if len(new_subjects) > 0:
                logger.info(
                    f"Extending subject vocab with {len(new_subjects)} new IDs: {new_subjects}"
                )
                subject_emb.extend_vocab(new_subjects, exist_ok=True)
            else:
                logger.info("Subject vocab already includes all subject IDs.")

        if ctx_embedder.tokenize_task:
            task_emb = ctx_embedder.task_emb
            existing = list(task_emb.vocab.keys()) if task_emb.vocab else []
            new_tasks = [t for t in self.get_task_ids() if t not in existing]
            if len(new_tasks) > 0:
                logger.info(
                    f"Extending task vocab with {len(new_tasks)} new IDs: {new_tasks}"
                )
                task_emb.extend_vocab(new_tasks, exist_ok=True)
            else:
                logger.info("Task vocab already includes all task IDs.")

    # TODO update
    def get_session_ids(self):
        return self.train_dataset.recording_ids

    def get_subject_ids(self):
        return self.train_dataset.get_subject_ids()

    def get_task_ids(self):
        return self.train_dataset.get_brainset_ids()

    def train_dataloader(self):
        cfg = self.cfg

        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals("train"),
            window_length=cfg.model.ctx_time,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=bs,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

        logger.info(f"Training on {len(train_sampler)} samples")
        logger.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        logger.info(f"Training on {len(self.get_session_ids())} sessions")

        return train_loader

    def val_dataloader(self):
        cfg = self.cfg

        val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.eval_dataset.get_sampling_intervals("valid"),
            window_length=cfg.model.ctx_time,
            drop_short=True,
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        val_loader = DataLoader(
            self.eval_dataset,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate,
            batch_size=bs,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

        logger.info(f"Expecting {len(val_sampler)} validation steps")

        return val_loader

    def test_dataloader(self):
        cfg = self.cfg
        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu

        test_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.eval_dataset.get_sampling_intervals("test"),
            window_length=cfg.model.ctx_time,
            drop_short=True,
        )
        test_loader = DataLoader(
            self.eval_dataset,
            sampler=test_sampler,
            shuffle=False,
            collate_fn=collate,
            batch_size=bs,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

        logger.info(f"Testing on {len(test_sampler)} samples")

        return test_loader


def _load_from_checkpoint(cfg, model):
    logger.info(f"Loading checkpoint from {cfg.checkpoint_path}")

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    if cfg.get("new_decoder", False):
        logger.info("Loading pretrained weights (everything except decoder)")

        filtered_state_dict = {
            k: v
            for k, v in clean_state_dict.items()
            if not k.startswith(NEW_DECODER_PREFIXES)
        }

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

        logger.info(f"Loaded pretrained model (excluding decoder).")
        logger.info(
            f"Filtered out {len(clean_state_dict) - len(filtered_state_dict)} tensors."
        )
        logger.info(f"Missing keys (should be decoder/head): {len(missing)}")

        for m in missing:
            logger.debug(f"Missing expected key: {m}")

    else:
        logger.info("Loading full model weights")
        model.load_state_dict(clean_state_dict, strict=False)


@hydra.main(version_base="1.3", config_path="./configs", config_name="defaults")
def main(cfg):
    logger.info("NDT2!")

    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # TODO check if needed (need to be better)
    with open_dict(cfg):
        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        logger.info(f"Number of GPUs: {num_gpus}")
        logger.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        logger.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    # make model amd data module
    data_module = DataModule(cfg, cfg.is_ssl)
    readout_spec = data_module.readout_spec

    if cfg.is_ssl:
        model = hydra.utils.instantiate(cfg.model, is_ssl=cfg.is_ssl)
    else:
        model = hydra.utils.instantiate(
            cfg.model, is_ssl=cfg.is_ssl, readout_spec=readout_spec
        )

    # Load from checkpoint
    if cfg.get("load_from_checkpoint", False):
        _load_from_checkpoint(cfg, model)

    data_module.link_model(model)

    # Train wrapper
    wrapper = NDT2TrainWrapper(cfg=cfg, model=model, modality_spec=readout_spec)

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    monitor = "val_loss"
    if cfg.callbacks.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.callbacks.checkpoint_path,
            filename=f"{cfg.wandb.run_name}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
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

    strategy = (
        "ddp_find_unused_parameters_true"
        if cfg.optimizer.get("freeze_encoder", False)
        else "auto"
    )
    # Set up trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu",
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        strategy=strategy,
    )

    # Train
    trainer.fit(wrapper, data_module)

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
