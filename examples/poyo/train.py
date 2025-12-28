import logging
from typing import Dict
from collections import defaultdict

import hydra
import lightning as L
import pandas as pd
from rich import print as rprint
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
import wandb

from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.optim import SparseLamb
from torch_brain.models.poyo import POYO
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.metrics import MetricGroupWithStitcher
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    SequentialFixedWindowSampler,
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
        model: POYO,
        validation_metric_group: MetricGroupWithStitcher,
        test_metric_group: MetricGroupWithStitcher,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = model.readout_spec

        self.save_hyperparameters(OmegaConf.to_container(cfg))

        self.validation_metric_group = validation_metric_group
        self.test_metric_group = test_metric_group

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
        self._evaluation_step(batch, self.validation_metric_group)

    def on_validation_epoch_end(self):
        metric_dict = self.validation_metric_group.compute(device=self.device)
        self.validation_metric_group.reset()
        self._log_metric_dict(metric_dict, prefix="val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, self.test_metric_group)

    def on_test_epoch_end(self):
        metric_dict = self.test_metric_group.compute(device=self.device)
        self.test_metric_group.reset()
        self._log_metric_dict(metric_dict, prefix="test")

    def _evaluation_step(self, batch: Dict, metric_group: MetricGroupWithStitcher):
        # forward pass
        output_values = self.model(**batch["model_inputs"])

        batch_size = len(batch["session_id"])
        for sample_idx in range(batch_size):
            mask = batch["eval_mask"][sample_idx]
            preds = output_values[sample_idx, mask]
            targets = batch["target_values"][sample_idx, mask]
            timestamps = (
                batch["model_inputs"]["output_timestamps"][sample_idx, mask]
                + batch["absolute_start"][sample_idx]
            )

            metric_group.update(
                timestamps=timestamps.detach().cpu(),
                preds=preds.detach().cpu(),
                targets=targets.detach().cpu(),
                recording_id=batch["session_id"][sample_idx],
            )

    def _log_metric_dict(self, metric_dict: Dict[str, float], prefix: str):
        metric_dict["average_metric"] = sum(metric_dict.values()) / len(metric_dict)
        metric_dict_with_prefix = {f"{prefix}/{k}": v for k, v in metric_dict.items()}
        self.log_dict(metric_dict_with_prefix)

        metric_df = pd.DataFrame(
            {
                "metric": metric_dict_with_prefix.keys(),
                "value": metric_dict_with_prefix.values(),
            }
        )

        if self.trainer.is_global_zero:
            logging.info(f"Logged {len(metric_dict_with_prefix)} {prefix} metrics.")
            rprint(metric_df)

            for logger in self.trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metric_df)}
                    )


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def setup_dataset_and_link_model(self, model: POYO):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """
        self.sequence_length = model.sequence_length

        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=Compose([*train_transforms, model.tokenize]),
        )
        self.train_dataset.disable_data_leakage_check()

        self._init_model_vocab(model)

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.val_dataset.disable_data_leakage_check()

        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.test_dataset.disable_data_leakage_check()

        self.val_metric_group = self.build_metric_group()
        self.test_metric_group = self.build_metric_group()

    def _init_model_vocab(self, model: POYO):
        # TODO: Add code for finetuning situation (when model already has a vocab)
        model.unit_emb.initialize_vocab(self.get_unit_ids())
        model.session_emb.initialize_vocab(self.get_session_ids())

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def build_metric_group(self):
        dataset_config_dict = self.get_recording_config_dict()
        metrics = defaultdict(list)
        # setup the metrics
        for recording_id, recording_config in dataset_config_dict.items():
            metrics[recording_id] = []
            for metric_config in recording_config["readout"]["metrics"]:
                metric = hydra.utils.instantiate(metric_config["metric"])
                metrics[recording_id].append(metric)

        return MetricGroupWithStitcher(metrics=metrics)

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

        val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
        )

        val_sampler = self.val_metric_group.convert_to_stitcher_sampler(
            val_sampler,
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

        test_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
        )

        test_sampler = self.test_metric_group.convert_to_stitcher_sampler(
            test_sampler,
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

    # get modality details
    # TODO: add test to verify that all recordings have the same readout
    readout_id = cfg.dataset[0].config.readout.readout_id
    readout_spec = MODALITY_REGISTRY[readout_id]

    # make model and data module
    model = hydra.utils.instantiate(cfg.model, readout_spec=readout_spec)
    data_module = DataModule(cfg=cfg)
    data_module.setup_dataset_and_link_model(model)

    # Lightning train wrapper
    wrapper = TrainWrapper(
        cfg=cfg,
        model=model,
        validation_metric_group=data_module.val_metric_group,
        test_metric_group=data_module.test_metric_group,
    )

    callbacks = [
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
