import logging
import math

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf

# Allow numpy scalar types in torch.load (required for PyTorch 2.6+)
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
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

        # Get learning rate multipliers from config (default to 1.0 if not specified)
        embedding_lr_mult = getattr(self.cfg.optim, "embedding_lr_multiplier", 1.0)
        lora_lr_mult = getattr(self.cfg.optim, "lora_lr_multiplier", 1.0)

        # Separate parameters into groups for different learning rates
        embedding_params = []
        lora_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if it's an embedding parameter
            if any(emb_name in name for emb_name in ["unit_emb", "session_emb"]):
                embedding_params.append(param)
            # Check if it's a LoRA parameter (typically named lora_A, lora_B, or similar)
            elif "lora" in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if embedding_params:
            param_groups.append(
                {
                    "params": embedding_params,
                    "lr": max_lr * embedding_lr_mult,
                    "name": "embeddings",
                }
            )
            logger.info(
                f"Embedding params: {len(embedding_params)} tensors, lr_mult={embedding_lr_mult}"
            )

        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": max_lr * lora_lr_mult, "name": "lora"}
            )
            logger.info(
                f"LoRA params: {len(lora_params)} tensors, lr_mult={lora_lr_mult}"
            )

        if other_params:
            param_groups.append({"params": other_params, "lr": max_lr, "name": "other"})
            logger.info(f"Other params: {len(other_params)} tensors, lr_mult=1.0")

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=max_lr,  # default lr (used if param group doesn't specify one)
            weight_decay=self.cfg.optim.weight_decay,
        )

        # Build lr_lambdas list matching the param_groups order
        lr_lambdas = []
        if embedding_params:
            lr_lambdas.append(
                lambda step, mult=embedding_lr_mult: self.lr_lambda(step) * mult
            )
        if lora_params:
            lr_lambdas.append(
                lambda step, mult=lora_lr_mult: self.lr_lambda(step) * mult
            )
        if other_params:
            lr_lambdas.append(lambda step: self.lr_lambda(step))

        # Use per-group lr_lambdas to maintain the multipliers through the schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def lr_lambda(self, current_step):
        """
        Computes a multiplicative factor for the learning rate based on the current training step.

        The learning rate schedule is divided into three phases:

        - warmup_steps: Number of steps to linearly increase the learning rate from 0 to max_lr.
        - hold_steps: Number of steps to keep the learning rate constant at max_lr after warmup.
        - decay_steps: Number of steps to decay the learning rate from max_lr to a minimum value
          (min_lr_factor * max_lr) using a cosine schedule.

        After these phases, the learning rate remains at the minimum value.

        Args:
            current_step (int): The current training step.

        Returns:
            float: The learning rate multiplier (to be multiplied with max_lr).
        """
        warmup_steps = self.cfg.optim.warmup_steps
        hold_steps = self.cfg.optim.hold_steps
        decay_steps = self.cfg.optim.decay_steps
        min_lr_factor = 0.1  # Decay to 10% of max_lr

        if current_step < warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + hold_steps or decay_steps == 0:
            # Hold phase. If decay_steps is 0, we don't decay.
            return 1.0
        elif current_step < warmup_steps + hold_steps + decay_steps:
            # Cosine decay phase
            progress = float(current_step - warmup_steps - hold_steps) / float(
                decay_steps
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Decay from 1.0 down to min_lr_factor
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
        else:
            # After decay, hold at minimum
            return min_lr_factor

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
    # trainer.fit(wrapper, data_module)

    # Test
    trainer.test(wrapper, data_module)  # , ckpt_path="best")


if __name__ == "__main__":
    main()
