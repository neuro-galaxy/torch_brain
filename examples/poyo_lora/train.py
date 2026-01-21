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

from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.optim import SparseLamb
from torch_brain.models.poyo import POYO
from torch_brain.nn import apply_lora_to_model, LoRAModelWrapper
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    DecodingStitchEvaluator,
    DataForDecodingStitchEvaluator,
)
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.transforms import Compose

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


logger = logging.getLogger(__name__)


class LoRATrainWrapper(L.LightningModule):
    """Lightning wrapper for LoRA finetuning of POYO models."""

    def __init__(
        self,
        cfg: DictConfig,
        model: LoRAModelWrapper,
        modality_spec: ModalitySpec,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        # Get trainable parameters (LoRA + embeddings + readout)
        trainable_params = self.model.get_trainable_parameters()

        # Separate sparse embedding params from other trainable params
        sparse_param_names = ["unit_emb", "session_emb"]
        sparse_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(sp_name in name for sp_name in sparse_param_names):
                sparse_params.append(param)
            else:
                other_params.append(param)

        optimizer = SparseLamb(
            [
                {"params": sparse_params, "sparse": True},
                {"params": other_params},
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

        unit_index = batch["model_inputs"]["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):
        # forward pass
        output_values = self.model(**batch["model_inputs"])

        # prepare data for evaluator
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
    """Data module for LoRA finetuning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def setup_dataset_and_link_model(self, model: LoRAModelWrapper):
        """Setup Dataset objects and update the model's embedding vocabs."""
        # Access the base model's sequence_length through the wrapper
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

    def _init_model_vocab(self, model: LoRAModelWrapper):
        """Initialize or extend model vocabulary for finetuning.

        For finetuning with a pretrained model, we extend the existing vocabulary.
        For training from scratch, we initialize the vocabulary.
        """
        unit_ids = self.get_unit_ids()
        session_ids = self.get_session_ids()

        # Check if vocab is already initialized (from checkpoint)
        # is_lazy() returns True if NOT initialized
        if not model.unit_emb.is_lazy():
            # Extend vocab with new units/sessions (exist_ok=True allows existing IDs)
            model.unit_emb.extend_vocab(unit_ids, exist_ok=True)
            model.unit_emb.subset_vocab(unit_ids)
            model.session_emb.extend_vocab(session_ids, exist_ok=True)
            model.session_emb.subset_vocab(session_ids)
        else:
            # Initialize vocab from scratch (no checkpoint provided)
            model.unit_emb.initialize_vocab(unit_ids)
            model.session_emb.initialize_vocab(session_ids)

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

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


def load_model_from_ckpt(model: nn.Module, ckpt_path: str) -> bool:
    """Load model weights from a checkpoint file.
    
    Returns:
        True if weights were loaded, False if no checkpoint was provided.
    """
    if ckpt_path is None:
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Remove "model." prefix from keys if present (from Lightning wrapper)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return True


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="finetune_lora.yaml"
)
def main(cfg: DictConfig):
    logger.info("POYO LoRA Finetuning!")

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
    readout_id = cfg.dataset[0].config.readout.readout_id
    readout_spec = MODALITY_REGISTRY[readout_id]

    # Create base model
    base_model = hydra.utils.instantiate(cfg.model, readout_spec=readout_spec)

    # Load pretrained weights if checkpoint is provided
    if load_model_from_ckpt(base_model, cfg.ckpt_path):
        logger.info(f"Loaded pretrained model from {cfg.ckpt_path}")
    else:
        logger.info("No checkpoint provided - using randomly initialized model")

    # Apply LoRA to the model
    lora_model = apply_lora_to_model(
        model=base_model,
        target_modules=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
        init_scale=cfg.lora.init_scale,
        target_projections=cfg.lora.target_projections,
    )

    # Freeze base model weights (keep LoRA, embeddings, and readout trainable)
    lora_model.freeze_base_model()

    # Setup data module (this initializes the embedding vocabularies)
    data_module = DataModule(cfg=cfg)
    data_module.setup_dataset_and_link_model(lora_model)

    # Print parameter summary (after vocab initialization)
    lora_model.print_parameter_summary()

    # Lightning train wrapper
    wrapper = LoRATrainWrapper(
        cfg=cfg,
        model=lora_model,
        modality_spec=readout_spec,
    )

    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        modality_spec=readout_spec,
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),
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
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,
        num_sanity_val_steps=-1 if cfg.sanity_check_validation else 0,
    )

    # Train
    trainer.fit(wrapper, data_module)

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
