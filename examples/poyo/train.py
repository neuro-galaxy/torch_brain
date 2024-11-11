import logging

import hydra
import lightning as L
import torch
import torch.nn as nn
from torch_optimizer import Lamb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf

from torch_brain.nn import compute_loss_or_metric
from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything, DataModule
from torch_brain.utils.stitcher import StitchEvaluator

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


class POYOTrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        dataset_config_dict: dict = None,
        steps_per_epoch: int = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.dataset_config_dict = dataset_config_dict
        self.steps_per_epoch = steps_per_epoch
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
            epochs=self.cfg.epochs,
            steps_per_epoch=self.steps_per_epoch,
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

            loss = loss + taskwise_loss[readout_id] * len(target)

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
        output_subtask_index = batch.pop("output_subtask_index")

        # forward pass
        output_values = self.model(**batch, unpack_output=True)

        # add removed elements back to batch
        batch["target_values"] = target_values
        batch["absolute_start"] = absolute_starts
        batch["session_id"] = session_ids
        batch["output_subtask_index"] = output_subtask_index

        return output_values

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix="test")


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
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
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITIY_REGISTRY)

    # setup data module
    data_module = DataModule(cfg, model.unit_emb.tokenizer, model.session_emb.tokenizer)
    data_module.setup()

    # register units and sessions
    model.unit_emb.initialize_vocab(data_module.get_unit_ids())
    model.session_emb.initialize_vocab(data_module.get_session_ids())

    # Lightning train wrapper
    wrapper = POYOTrainWrapper(
        cfg=cfg,
        model=model,
        dataset_config_dict=data_module.get_recording_config_dict(),
        steps_per_epoch=len(data_module.train_dataloader()),
    )

    evaluator = StitchEvaluator(
        dataset_config_dict=data_module.get_recording_config_dict()
    )

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
        num_sanity_val_steps=0,  # Disable sanity validation
        limit_val_batches=None,  # Ensure no limit on validation batches
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train
    trainer.fit(
        wrapper,
        data_module,
        ckpt_path=cfg.ckpt_path if not cfg.finetune else None,
    )


if __name__ == "__main__":
    main()
