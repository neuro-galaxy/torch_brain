import logging
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from data_loader_generator import DataLoaderGenerator
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import CombinedLoader
from model import (
    NDT2_bhvr_Decoder,
    NDT2_ContextManager,
    NDT2_Decoder,
    NDT2_MAE_MaskManager,
    NDT2_SpikesPatchifier,
    NDT2_Transformer,
)
from omegaconf import OmegaConf, open_dict
from torch import optim

log = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        dim = cfg.model.dim

        self.mae_mask_manager = NDT2_MAE_MaskManager(cfg.mask_ratio)

        self.ctx_manager = NDT2_ContextManager(dim)

        self.spikes_patchifier = NDT2_SpikesPatchifier(
            dim=dim, patch_size=cfg.patch_size
        )

        self.encoder = NDT2_Transformer(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            **cfg.model.encoder,
        )

        self.predictor = NDT2_Decoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            patch_size=cfg.patch_size,
            **cfg.model.predictor,
        )

        self.bhv_decoder = NDT2_bhvr_Decoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            bin_time=cfg.bin_time,
            **cfg.model.bhv_decoder,
        )

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def training_step(self, batch, batch_idx):
        mae_loss = 0.0
        if "ssl" in batch:
            mae_loss = self.mae_step(batch["ssl"])
            self.log("train_shuffle_infill_loss", mae_loss)

        bhv_loss = 0.0
        if "superv" in batch:
            bhv_loss, r2 = self.decoder_step(batch["superv"])
            self.log("val_kinematic_decoding_loss", bhv_loss)
            self.log("val_kinematic_r2_x", r2[0])
            self.log("val_kinematic_r2_y", r2[1])
            self.log("val_kinematic_r2", r2)

        loss = bhv_loss + mae_loss
        self.log("train_loss", loss, prog_bar=True)

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        mae_loss = 0.0
        if "ssl" in batch:
            mae_loss = self.mae_step(batch["ssl"])
            self.log("val_shuffle_infill_loss", mae_loss)

        bhv_loss = 0.0
        if "superv" in batch:
            bhv_loss, r2 = self.decoder_step(batch["superv"])
            self.log("val_kinematic_decoding_loss", bhv_loss)
            self.log("val_kinematic_r2_x", r2[0])
            self.log("val_kinematic_r2_y", r2[1])
            self.log("val_kinematic_r2", r2)

        loss = mae_loss + bhv_loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def mae_step(self, batch, eval_mode=False):
        batch = self.mae_mask_manager(batch, eval_mode)
        encoder_input = self.spikes_patchifier(batch["spike_tokens"])
        ctx_emb = self.ctx_manager(batch, encoder_input.dtype)
        encoder_out = self.encoder.encode(encoder_input, ctx_emb, batch)
        loss = self.predictor(encoder_out, ctx_emb, batch, eval_mode)
        return loss

    def decoder_step(self, batch, eval_mode=False):
        encoder_input = self.spikes_patchifier(batch["spike_tokens"])
        ctx_emb = self.ctx_manager(batch, encoder_input.dtype)
        encoder_out = self.encoder.encode(encoder_input, ctx_emb, batch)
        loss, r2 = self.bhv_decoder(encoder_out, ctx_emb, batch)
        return loss, r2

    def configure_optimizers(self):
        # TODO update to match the superv settings (*10 lr for decoder layer and no scheduler)
        cfg = self.cfg.optimizer

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

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

    def get_data_loaders(self, cfg) -> Tuple[CombinedLoader, CombinedLoader]:
        # ssl
        ssl_cfg = OmegaConf.to_container(cfg.data_ssl.include)
        ssl_loader_generator = DataLoaderGenerator(cfg, ssl_cfg, self, True)
        ssl_train_loader = ssl_loader_generator("train")
        ssl_val_loader = ssl_loader_generator("valid")

        # superv
        superv_cfg = OmegaConf.to_container(cfg.data_superv.include)
        superv_loader_generator = DataLoaderGenerator(cfg, superv_cfg, self, True)
        superv_train_loader = superv_loader_generator("train")
        superv_val_loader = superv_loader_generator("valid")

        # combine loaders
        train_loader_dict, val_loader_dict = {}, {}
        if cfg.doing_ssl:
            train_loader_dict["ssl"] = ssl_train_loader
            val_loader_dict["ssl"] = ssl_val_loader
        if cfg.doing_superv:
            # TODO pb here because the same subject is registered 2 times
            train_loader_dict["superv"] = superv_train_loader
            val_loader_dict["superv"] = superv_val_loader
        train_loader = CombinedLoader(train_loader_dict, mode="max_size_cycle")
        val_loader = CombinedLoader(val_loader_dict, mode="max_size")

        # Set up vocab
        self.ctx_manager.init_vocab(ssl_loader_generator.get_vocab())

        cfg.encoder_finetune = True
        if cfg.doing_superv and cfg.encoder_finetune:
            self.ctx_manager.extend_vocab(superv_loader_generator.get_vocab())

        if cfg.superv_only:
            # TODO
            pass

        # TODO better logging info
        # log.info(f"SSL sessions: {len(ses_ids)}")
        # log.info(f"Superv sessions: {len(superv_ses_ids)}")
        # log.info(f"Vocab size: {len(train_wrapper.encoder.patchifier.ses_emb.vocab)}")

        return train_loader, val_loader


def set_wandb(cfg, log) -> Optional[WandbLogger]:
    if not cfg.wandb.enable:
        return None
    wandb_logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        save_dir=cfg.log_dir,
        log_model=False,
    )
    log.info(f"Using wandb logger: {wandb_logger.version}")

    return wandb_logger


def set_callbacks(cfg) -> List[Callback]:
    cfg = cfg.checkpoint
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.enable:
        callbacks.append(
            ModelCheckpoint(
                save_last=True,  # saves a checkpoint for the last epoch
                every_n_train_steps=cfg.every_n_steps,
                every_n_epochs=cfg.every_n_epochs,
                save_top_k=-1 if cfg.save_all else 1,
            )
        )
    return callbacks


def run_training(cfg):
    L.seed_everything(cfg.seed)

    if cfg.fast_dev_run:
        cfg.wandb.enable = False
        cfg.num_workers = 0

    with open_dict(cfg):
        cfg.doing_ssl = not cfg.superv_only
        cfg.doing_superv = not cfg.ssl_only

        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        log.info(f"Number of GPUs: {num_gpus}")
        log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    wandb_logger = set_wandb(cfg, log)

    # Train wrapper
    train_wrapper = TrainWrapper(cfg)

    # data loaders
    train_loader, val_loader = train_wrapper.get_data_loaders(cfg)

    L.seed_everything(cfg.seed)

    # Callbacks
    callbacks = set_callbacks(cfg)

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
    trainer.fit(train_wrapper, train_loader, val_loader)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
