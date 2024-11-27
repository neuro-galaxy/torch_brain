import logging
from typing import Callable, Dict
import copy

import hydra
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data

from torch_brain.registry import MODALITIY_REGISTRY, ModalitySpec
from torch_brain.models.poyo import POYOTokenizer, poyo_mp
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import DecodingStitchEvaluator
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.transforms import Compose

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO!")

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

    # get modality details
    modality_spec = MODALITIY_REGISTRY[cfg.modality_name]

    # make model and tokenizer
    model = poyo_mp(dim_out=modality_spec.dim)

    tokenizer = POYOTokenizer(
        unit_tokenizer=model.unit_emb.tokenizer,
        session_tokenizer=model.session_emb.tokenizer,
        latent_step=cfg.latent_step,
        num_latents_per_step=cfg.model.num_latents,
        modality_spec=modality_spec,
        sequence_length=cfg.sequence_length,
        subtask_weights=cfg.subtask_weights,
    )

    # setup data module
    data_module = DataModule(cfg=cfg, tokenizer=tokenizer)
    data_module.setup()

    # register units and sessions
    model.unit_emb.initialize_vocab(data_module.get_unit_ids())
    model.session_emb.initialize_vocab(data_module.get_session_ids())

    # Lightning train wrapper
    wrapper = POYOTrainWrapper(
        cfg=cfg,
        model=model,
        modality_spec=modality_spec,
    )

    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        modality_spec=modality_spec,
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,  # Ensure no limit on validation batches
        num_sanity_val_steps=cfg.num_sanity_val_steps,
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    # Test
    trainer.test(wrapper, data_module)


class POYOTrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        modality_spec: ModalitySpec,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        optimizer = Lamb(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        output_mask = batch.pop("output_mask")

        # forward pass
        output_values = self.model(**batch)

        # compute loss
        output_values = output_values[output_mask]
        target_values = target_values[output_mask]
        target_weights = target_weights[output_mask]

        if self.modality_spec.loss_fn == "mse":
            loss = torch.nn.functional.mse_loss(
                output_values, target_values, reduction="none"
            )
            loss = loss * target_weights[:, None]
            loss = loss.mean()
        else:
            raise NotImplementedError("Only MSE loss is supported for now.")

        self.log("train_loss", loss, prog_bar=True)

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        unit_index = batch["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):
        target_values = batch.pop("target_values")
        batch.pop("target_weights")
        absolute_starts = batch.pop("absolute_start")
        session_ids = batch.pop("session_id")
        output_mask = batch.pop("output_mask")

        # forward pass
        output_values = self.model(**batch)

        # add removed elements back to batch
        batch["target_values"] = target_values
        batch["absolute_start"] = absolute_starts
        batch["session_id"] = session_ids
        batch["output_mask"] = output_mask

        return output_values

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer: Callable[[Data], Dict]):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.train_dataset = None
        self.val_dataset = None
        self.sequence_length = self.cfg.sequence_length
        self.log = logging.getLogger(__name__)

    def setup(self, stage=None):
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=Compose([*train_transforms, self.tokenizer]),
        )
        self.train_dataset.disable_data_leakage_check()

        eval_tokenizer = copy.copy(self.tokenizer)
        eval_tokenizer.eval = True
        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=eval_tokenizer,
        )
        self.val_dataset.disable_data_leakage_check()

        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=eval_tokenizer,
        )
        self.test_dataset.disable_data_leakage_check()

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

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
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            interval_dict=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")

        return val_loader

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            interval_dict=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")

        return test_loader


if __name__ == "__main__":
    main()
