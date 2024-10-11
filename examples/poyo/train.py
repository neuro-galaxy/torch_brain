import lightning
import logging

from collections import OrderedDict
import copy

import hydra
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

# Flags are absorbed by Hydra.
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_optimizer import Lamb

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from brainsets.taxonomy import decoder_registry
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything, train_wrapper
from torch_brain.models import POYOPlusTokenizer


def run_training(cfg: DictConfig):
    # Fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # Higher speed on machines with tensor cores.
    torch.set_float32_matmul_precision("medium")

    log = logging.getLogger(__name__)

    # Device setup is managed by PyTorch Lightning.

    # make model
    model = hydra.utils.instantiate(
        cfg.model,
        task_specs=decoder_registry,
        backend_config=cfg.backend_config,
        _convert_="object",
    )

    # prepare tokenizer and transforms

    # The transform list is defined in the config file.
    sequence_length = 1.0
    transforms = hydra.utils.instantiate(
        cfg.train_transforms, sequence_length=sequence_length
    )

    # build tokenizer
    tokenizer = POYOPlusTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=decoder_registry,
        latent_step=1 / 8,
        num_latents_per_step=cfg.model.num_latents,
        batch_type=model.batch_type,
    )

    transform = Compose([*transforms, tokenizer])

    log.info("Data root: {}".format(cfg.data_root))
    train_dataset = Dataset(
        cfg.data_root,
        "train",
        include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
        transform=transform,
    )
    train_dataset.disable_data_leakage_check()

    # In Lightning, testing only happens once, at the end of training. To get the
    # intended behavior, we need to specify a validation set instead.
    val_tokenizer = copy.copy(tokenizer)
    val_tokenizer.eval = True
    val_dataset = Dataset(
        cfg.data_root,
        "test",
        include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
        transform=val_tokenizer,
    )
    val_dataset.disable_data_leakage_check()

    if not cfg.finetune:
        # Register units and sessions
        model.unit_emb.initialize_vocab(train_dataset.get_unit_ids())
        model.session_emb.initialize_vocab(train_dataset.get_session_ids())
    else:
        assert (
            cfg.ckpt_path is not None
        ), "Missing `ckpt_path`. Checkpoint is required finetuning."
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Remove 'model.' prefix at the front of the state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix("model.")
            new_state_dict[new_key] = state_dict[key]

        model.load_state_dict(new_state_dict)
        log.info(f"Loaded model state dict from {cfg.ckpt_path}")

        # Optionally freeze parameters for Unit Identification
        if cfg.freeze_perceiver_until_epoch != 0:
            model.freeze_middle()
            log.info(f"Froze perceiver")

        # Register new units and sessions, and delete old ones
        model.unit_emb.extend_vocab(train_dataset.get_unit_ids(), exist_ok=False)
        model.unit_emb.subset_vocab(train_dataset.get_unit_ids())

        model.session_emb.extend_vocab(train_dataset.session_ids, exist_ok=False)
        model.session_emb.subset_vocab(train_dataset.session_ids)

    # sampler and dataloader
    train_sampler = RandomFixedWindowSampler(
        interval_dict=train_dataset.get_sampling_intervals(),
        window_length=sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
        # For debugging. we allow the user to set num_workers to 0.
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    log.info(f"Training on {len(train_sampler)} samples")
    log.info(f"Training on {len(train_dataset.get_unit_ids())} units")
    log.info(f"Training on {len(train_dataset.get_session_ids())} sessions")

    val_sampler = SequentialFixedWindowSampler(
        interval_dict=val_dataset.get_sampling_intervals(),
        window_length=sequence_length,
        step=sequence_length / 2,
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=collate,
        batch_size=cfg.get(
            "eval_batch_size", cfg.batch_size
        ),  # Default to training batch size, but allow override in config.
        num_workers=2,
    )

    # No need to explicitly use DDP with the model, lightning does this for us.
    max_lr = cfg.base_lr * cfg.batch_size

    if cfg.epochs > 0 and cfg.steps == 0:
        epochs = cfg.epochs
    elif cfg.steps > 0 and cfg.epochs == 0:
        epochs = cfg.steps // len(train_loader) + 1
    else:
        raise ValueError("Must specify either epochs or steps")

    log.info(f"Epochs: {epochs}")

    optimizer = Lamb(
        model.parameters(),  # filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
        div_factor=1,
    )

    # Now we create the model wrapper. It's a simple shim that contains the train and
    # test code.
    wrapper = train_wrapper.TrainWrapper(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    tb = lightning.pytorch.loggers.tensorboard.TensorBoardLogger(
        save_dir=cfg.log_dir,
    )

    wandb = lightning.pytorch.loggers.WandbLogger(
        save_dir=cfg.log_dir,
        entity=cfg.get("wandb_entity", None),
        name=cfg.name,
        project=cfg.get("wandb_project", "poyo"),
        log_model=cfg.get("wandb_log_model", False),
    )
    print(f"Wandb ID: {wandb.version}")

    callbacks = [
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            dirpath=f"logs/lightning_logs/{wandb.version}",
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        train_wrapper.CustomValidator(val_loader),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
    ]

    if cfg.finetune:
        if cfg.freeze_perceiver_until_epoch > 0:
            callbacks.append(
                train_wrapper.UnfreezeAtEpoch(cfg.freeze_perceiver_until_epoch)
            )

    trainer = lightning.Trainer(
        logger=[tb, wandb],
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        num_sanity_val_steps=0,
        precision=cfg.precision,
        reload_dataloaders_every_n_epochs=5,
        accelerator="gpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: {trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    for logger in trainer.loggers:
        # OmegaConf.to_container converts the config object to a dictionary.
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # To resume from a checkpoint rather than training from scratch,
    # set ckpt_path on the command line.
    trainer.fit(
        wrapper,
        train_loader,
        [0],
        ckpt_path=cfg.ckpt_path if not cfg.finetune else None,
    )
    # [0] is a hack to force the validation callback to be called.


# This loads the config file using Hydra, similar to Flags, but composable.
@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # Train the whole thing.
    # This inner function is unnecessary, but I keep it here to maintain
    # a parallel to the original code.
    run_training(cfg)


if __name__ == "__main__":
    main()
