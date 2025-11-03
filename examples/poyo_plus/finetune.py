import logging
from typing import List, Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig

from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.datamodules import DataModule
from torch_brain.utils.stitcher import StitchEvaluator

from train import POYOTrainWrapper

from torch_brain.models.poyo_plus import POYOPlus, CAPOYO

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


class GradualUnfreezing(L.Callback):
    r"""A Lightning callback to handle freezing and unfreezing of the model for the
    purpose of finetuning the model to new sessions. If this callback is used,
    most of the model weights will be frozen initially.
    The only parts of the model that will be left unforzen are the unit, and session embeddings.
    One we reach the specified epoch (`unfreeze_at_epoch`), the entire model will be unfrozen.
    """

    _has_been_frozen: bool = False
    frozen_params: Optional[List[nn.Parameter]] = None

    def __init__(self, unfreeze_at_epoch: int):
        self.enabled = unfreeze_at_epoch != 0
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.cli_log = logging.getLogger(__name__)

    @classmethod
    def freeze(cls, model):
        r"""Freeze the model weights, except for the unit and session embeddings, and
        return the list of frozen parameters.
        """
        if isinstance(model, POYOPlus):
            layers_to_freeze = [
                model.enc_atn,
                model.enc_ffn,
                model.proc_layers,
                model.dec_atn,
                model.dec_ffn,
                model.readout,
                model.token_type_emb,
                model.task_emb,
            ]
        elif isinstance(model, CAPOYO):
            layers_to_freeze = [
                model.enc_atn,
                model.enc_ffn,
                model.proc_layers,
                model.dec_atn,
                model.dec_ffn,
                model.readout,
                model.task_emb,
                model.input_value_map,  # for calcium value map
            ]
        else:
            raise ValueError(
                f"Model {type(model)} is not a supported model for finetuning."
            )

        frozen_params = []
        for layer in layers_to_freeze:
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params.append(param)

        return frozen_params

    def on_train_start(self, trainer, pl_module):
        if self.enabled:
            self.frozen_params = self.freeze(pl_module.model)
            self._has_been_frozen = True
            self.cli_log.info(
                f"POYO+ Perceiver frozen at epoch 0. "
                f"Will stay frozen until epoch {self.unfreeze_at_epoch}."
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled and (trainer.current_epoch == self.unfreeze_at_epoch):
            if not self._has_been_frozen:
                raise RuntimeError("Model has not been frozen yet.")

            for param in self.frozen_params:
                param.requires_grad = True

            self.frozen_params = None
            self.cli_log.info(
                f"POYO+ Perceiver unfrozen at epoch {trainer.current_epoch}"
            )


def load_model_from_ckpt(model: nn.Module, ckpt_path: str) -> None:
    if ckpt_path is None:
        raise ValueError("Must provide a checkpoint path to finetune the model.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)


@hydra.main(version_base="1.3", config_path="./configs", config_name="finetune.yaml")
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
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)
    load_model_from_ckpt(model, cfg.ckpt_path)
    log.info(f"Loaded model weights from {cfg.ckpt_path}")

    # setup data module
    data_module = DataModule(cfg, model.unit_emb.tokenizer, model.session_emb.tokenizer)
    data_module.setup()

    # register units and sessions
    unit_ids, session_ids = data_module.get_unit_ids(), data_module.get_session_ids()
    model.unit_emb.extend_vocab(unit_ids, exist_ok=True)
    model.unit_emb.subset_vocab(unit_ids)
    model.session_emb.extend_vocab(session_ids, exist_ok=True)
    model.session_emb.subset_vocab(session_ids)

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
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
        GradualUnfreezing(cfg.freeze_perceiver_until_epoch),
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
        num_sanity_val_steps=0,
        limit_val_batches=None,  # Ensure no limit on validation batches
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train
    trainer.fit(wrapper, data_module)

    # Test
    trainer.test(wrapper, data_module, "best")


if __name__ == "__main__":
    main()
