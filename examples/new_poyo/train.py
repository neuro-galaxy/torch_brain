import logging

import hydra
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data

from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.optim import SparseLamb
from torch_brain.models.poyo import POYO
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    DecodingStitchEvaluator,
    DataForDecodingStitchEvaluator,
)
from torch_brain.data import collate
from torch_brain.datasets import Dataset
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.transforms import Compose

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
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

        special_emb_params = list(self.model.unit_emb.parameters()) + list(
            self.model.session_emb.parameters()
        )

        remaining_params = [
            p
            for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n
        ]

        optimizer = SparseLamb(
            [
                {"params": special_emb_params, "sparse": True},
                {"params": remaining_params},
            ],
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

        # forward pass
        output_values = self.model(**batch["model_inputs"])

        # compute loss
        mask = batch["model_inputs"]["output_mask"]
        output_values = output_values[mask]
        target_values = batch["target_values"][mask]
        target_weights = batch["target_weights"][mask]

        loss = self.modality_spec.loss_fn(output_values, target_values, target_weights)

        self.log("train_loss", loss, prog_bar=True)

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        unit_index = batch["model_inputs"]["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"])

        # prepare data for evaluator
        # (goes to DecodingStitchEvaluator.on_validation_batch_end)
        data_for_eval = DataForDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

        return data_for_eval

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        self.train_dataset, self.readout_spec = hydra.utils.instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            split="train",
        )
        self.train_dataset.transform = Compose(
            [self.train_dataset.transform, *train_transforms]
        )

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)
        self.val_dataset, _ = hydra.utils.instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            split="valid",
        )
        self.val_dataset.transform = Compose(
            [self.val_dataset.transform, *eval_transforms]
        )

        self.test_dataset, _ = hydra.utils.instantiate(
            self.cfg.dataset,
            root=self.cfg.data_root,
            split="test",
        )
        self.test_dataset.transform = Compose(
            [self.test_dataset.transform, *eval_transforms]
        )

    def link_model(self, model: POYO):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """

        model.unit_emb.initialize_vocab(self.train_dataset.get_unit_ids())
        model.session_emb.initialize_vocab(self.get_session_ids())

        self.sequence_length = model.sequence_length

        self.train_dataset.transform.append(model.tokenize)
        self.val_dataset.transform.append(model.tokenize)
        self.test_dataset.transform.append(model.tokenize)

    def get_session_ids(self):
        return self.train_dataset.recording_ids

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
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
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        return train_loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
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
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
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

    # make model and data module
    data_module = DataModule(cfg=cfg)
    readout_spec = data_module.readout_spec
    model = hydra.utils.instantiate(cfg.model, readout_spec=readout_spec)
    data_module.link_model(model)

    # Lightning train wrapper
    wrapper = TrainWrapper(
        cfg=cfg,
        model=model,
        modality_spec=readout_spec,
    )

    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        modality_spec=readout_spec,
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
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
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,  # Ensure no limit on validation batches
        num_sanity_val_steps=-1 if cfg.sanity_check_validation else 0,
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
