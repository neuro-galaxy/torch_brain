import copy
import logging
from collections import defaultdict
from typing import Callable, Dict

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data
from torch.utils.data import DataLoader
from torch_optimizer import Lamb

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.models import POYOPlusTokenizer
from torch_brain.nn import compute_loss_or_metric
from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.transforms import Compose
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator


# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
    ):
        super().__init__()

        self.model = model
        self.cfg = cfg
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

        # forward pass
        output_values = self.model(**batch, unpack_output=False)

        # compute loss
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        taskwise_loss = {}
        for readout_id in output_values.keys():
            output = output_values[readout_id]
            target = target_values[readout_id]

            spec = self.model.readout.readout_specs[readout_id]

            weights = 1.0
            if readout_id in target_weights and target_weights[readout_id] is not None:
                weights = target_weights[readout_id]

            taskwise_loss[readout_id] = compute_loss_or_metric(
                spec.loss_fn, spec.type, output, target, weights
            )

            # count the number of sequences in the batch that have the current task
            num_sequences_with_current_task = torch.any(
                batch["output_decoder_index"] == MODALITIY_REGISTRY[readout_id].id,
                dim=1,
            ).sum()
            loss = loss + taskwise_loss[readout_id] * num_sequences_with_current_task

        batch_size = batch["input_unit_index"].shape[0]
        # TODO change batch_size when POYOPlusEfficient is used
        loss = loss / batch_size

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

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
        eval_mask = batch.pop("eval_mask")

        # forward pass
        output_values = self.model(**batch, unpack_output=True)

        # add removed elements back to batch
        batch["target_values"] = target_values
        batch["absolute_start"] = absolute_starts
        batch["session_id"] = session_ids
        batch["eval_mask"] = eval_mask

        return output_values

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: Callable[[Data], Dict],
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.train_dataset = None
        self.log = logging.getLogger(__name__)

    def setup(self, stage=None):
        # prepare transforms
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)

        # compose transforms, tokenizer is always the last transform
        train_transform = Compose([*train_transforms, self.tokenizer])

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=train_transform,
        )
        self.train_dataset.disable_data_leakage_check()

        # validation and test datasets require a tokenizer that is in eval mode
        eval_tokenizer = copy.copy(self.tokenizer)
        eval_tokenizer.eval = True

        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
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

    def get_metrics(self):
        dataset_config_dict = self.get_recording_config_dict()
        metrics = defaultdict(lambda: defaultdict(dict))
        # setup the metrics
        for recording_id, recording_config in dataset_config_dict.items():
            for readout_config in recording_config["multitask_readout"]:
                readout_id = readout_config["readout_id"]
                for metric_config in readout_config["metrics"]:
                    metric = hydra.utils.instantiate(metric_config["metric"])
                    metrics[recording_id][readout_id][str(metric)] = metric
        return metrics

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            interval_dict=self.train_dataset.get_sampling_intervals(),
            window_length=self.cfg.sequence_length,
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
            window_length=self.cfg.sequence_length,
            step=self.cfg.sequence_length / 2,
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
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        self.val_sequence_index = val_sampler.sequence_index

        return val_loader

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            interval_dict=self.test_dataset.get_sampling_intervals(),
            window_length=self.cfg.sequence_length,
            step=self.cfg.sequence_length / 2,
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
            num_workers=0,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO+!")
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # make model
    # TODO: resolve the readout_id from dataset, only build readouts needed
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITIY_REGISTRY)

    tokenizer = POYOPlusTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=MODALITIY_REGISTRY,
        latent_step=cfg.latent_step,
        num_latents_per_step=cfg.model.num_latents,
    )

    # setup data module
    data_module = DataModule(cfg, tokenizer)
    data_module.setup()

    # register units and sessions
    model.unit_emb.initialize_vocab(data_module.get_unit_ids())
    model.session_emb.initialize_vocab(data_module.get_session_ids())

    # Lightning train wrapper
    wrapper = TrainWrapper(cfg=cfg, model=model)

    evaluator = MultiTaskDecodingStitchEvaluator(metrics=data_module.get_metrics())

    callbacks = [
        evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
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
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        limit_val_batches=None,  # Ensure no limit on validation batches
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    # Test
    trainer.test(wrapper, data_module)


if __name__ == "__main__":
    main()
