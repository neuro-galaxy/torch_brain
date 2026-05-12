import rich
import torchmetrics
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import torch_brain
from torch_brain.data import collate
from torch_brain.dataset import Dataset
from torch_brain.transforms import Compose
from torch_brain.registry import ModalitySpec, DataType
from torch_brain.models import POYO
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.optim import SparseLamb

from utils import seed_everything, move_to_device, BehaviorStitcher

log = logging.getLogger(__name__)

READOUT_SPEC = ModalitySpec(
    id=0,
    dim=2,
    type=DataType.CONTINUOUS,
    timestamp_key="cursor.timestamps",
    value_key="cursor.vel",
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

device = torch.device("cuda")


def loss_fn(pred, target, weights):
    """Simple sample-weighted MSE loss"""
    loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
    loss = loss.flatten(1).mean(1) * weights
    loss = loss.sum() / weights.sum()
    return loss


metric_fn = torchmetrics.functional.r2_score


def create_optim(model: POYO, steps_per_epoch: int, cfg: DictConfig):
    emb_params = [
        p for n, p in model.named_parameters() if "unit_emb" in n or "session_emb" in n
    ]
    nonemb_params = [
        p
        for n, p in model.named_parameters()
        if "unit_emb" not in n and "session_emb" not in n
    ]

    max_lr = cfg.optim.base_lr * cfg.batch_size
    optim = SparseLamb(
        [
            {"params": emb_params, "sparse": True},
            {"params": nonemb_params},
        ],
        lr=max_lr,  # linear scaling rule
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr,
        steps_per_epoch * cfg.epochs,
        pct_start=cfg.optim.lr_decay_start,
        div_factor=1,
    )
    log.info(
        f"Optim: max_lr={max_lr}, "
        f"# Embedding params={sum(p.numel() for p in emb_params):,}, "
        f"# Non-Embedding params={sum(p.numel()for p in nonemb_params):,}"
    )
    return optim, scheduler


@hydra.main(version_base="1.3", config_path="./configs")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Setup dataset
    train_ds = instantiate(cfg.dataset, root=cfg.data_root)
    train_ds.transform = Compose(instantiate(cfg.train_transforms))

    eval_ds = instantiate(cfg.dataset, root=cfg.data_root)
    eval_ds.transform = Compose(instantiate(cfg.eval_transforms))

    log.info(
        f"Dataset: num_recordings={len(train_ds.recording_ids)}, "
        f"num_units={len(train_ds.get_unit_ids())}"
    )

    # Setup model
    model: POYO = instantiate(cfg.model, readout_spec=READOUT_SPEC)
    model.init_vocabs(train_ds)
    model = model.to(device)
    train_ds.transform.transforms.append(model.tokenize)
    eval_ds.transform.transforms.append(model.tokenize)

    # Samplers
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_ds.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )
    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=eval_ds.get_sampling_intervals("valid"),
        window_length=model.sequence_length,
        step=model.sequence_length / 2.0,
    )

    # Loaders
    loader_args = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        collate_fn=collate,
    )
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        drop_last=True,
        **loader_args,  # type: ignore
    )
    val_loader = DataLoader(eval_ds, sampler=val_sampler, **loader_args)  # type: ignore

    # Optimizer
    optim, scheduler = create_optim(model, len(train_loader), cfg)

    # Train loop
    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        train_epoch(train_loader, model, optim, scheduler)
        eval_epoch(val_loader, model)


def train_epoch(loader, model, optim, scheduler):
    model.train()

    loader_pbar = tqdm(loader, leave=False)
    for batch in loader_pbar:
        batch = move_to_device(batch, device)
        mask = batch["model_inputs"]["output_mask"]
        pred = model(**batch["model_inputs"])[mask]

        target = batch["target_values"][mask]
        loss_weights = batch["target_weights"][mask]
        loss = loss_fn(pred, target, loss_weights)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        loader_pbar.set_description(f"Loss: {loss.item():.3f}")


@torch.no_grad()
def eval_epoch(loader, model):
    model.eval()

    stitchers = {rid: BehaviorStitcher() for rid in loader.dataset.recording_ids}

    for batch in tqdm(loader, leave=False):
        batch = move_to_device(batch, device)
        pred = model(**batch["model_inputs"])

        for i in range(len(pred)):
            _mask = batch["model_inputs"]["output_mask"][i]
            _abs_start = batch["absolute_start"][i]
            _rid = batch["session_id"][i]
            _timestamps = batch["model_inputs"]["output_timestamps"][i][_mask]
            stitchers[_rid].update(
                preds=pred[i][_mask],
                targets=batch["target_values"][i][_mask],
                timestamps=_timestamps + _abs_start,
            )

    metrics = {}
    for rid, stitcher in stitchers.items():
        pred, target = stitcher.compute()
        metrics[rid] = metric_fn(pred, target)

    rich.print(metrics)


if __name__ == "__main__":
    main()
