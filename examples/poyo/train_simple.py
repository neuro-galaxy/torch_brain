import rich
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import torch
import torchmetrics

import torch_brain
from torch_brain.data import collate
from torch_brain.models import POYO
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)

from utils import seed_everything, move_to_device, BehaviorStitcher, create_optim
from datasets.wrapper import PoyoDatasetWrapper

log = logging.getLogger(__name__)

device = torch.device("cuda")


def loss_fn(pred, target, weights):
    """Simple sample-weighted MSE loss"""
    loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
    loss = loss.flatten(1).mean(1) * weights
    loss = loss.sum() / weights.sum()
    return loss


metric_fn = torchmetrics.functional.r2_score


@hydra.main(version_base="1.3", config_path="./configs")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Datasets
    train_ds = instantiate(cfg.dataset, root=cfg.data_root)
    train_ds.transform = instantiate(cfg.train_transform)

    eval_ds = instantiate(cfg.dataset, root=cfg.data_root)
    eval_ds.transform = instantiate(cfg.eval_transform)

    log.info(
        f"Dataset: num_recordings={len(train_ds.recording_ids)}, "
        f"num_units={len(train_ds.get_unit_ids())}"
    )

    # Model
    model: POYO = instantiate(cfg.model, dim_out=train_ds.dim_target)
    model.init_vocabs(train_ds)
    model = model.to(device)
    train_ds = PoyoDatasetWrapper(train_ds, model.tokenize)
    eval_ds = PoyoDatasetWrapper(eval_ds, model.tokenize)

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
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        drop_last=True,
        **loader_args,  # type: ignore
    )
    val_loader = torch.utils.data.DataLoader(eval_ds, sampler=val_sampler, **loader_args)  # type: ignore

    # Optimizer
    optim, scheduler = create_optim(model, len(train_loader), cfg)

    # Train loop
    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        train_epoch(train_loader, model, optim, scheduler)
        eval_epoch(val_loader, model)


def train_epoch(loader, model, optim, scheduler):
    model.train()

    loader_pbar = tqdm(loader, leave=False)
    for X, Y in loader_pbar:
        X, Y = move_to_device((X, Y), device)
        mask = Y["output_mask"]
        pred = model(**X, output_timestamps=Y["timestamps"])[mask]

        target = Y["values"][mask]
        loss_weights = Y["weights"][mask]
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

    for X, Y in tqdm(loader, leave=False):
        X, Y = move_to_device((X, Y), device)
        pred = model(**X, output_timestamps=Y["timestamps"])

        for i in range(len(pred)):
            _mask = Y["eval_mask"][i]
            _rid = Y["session_id"][i]
            stitchers[_rid].update(
                preds=pred[i][_mask],
                targets=Y["values"][i][_mask],
                timestamps=Y["timestamps"][i][_mask] + Y["absolute_start"][i],
            )

    metrics = {}
    for rid, stitcher in stitchers.items():
        pred, target = stitcher.compute()
        metrics[rid] = metric_fn(pred, target).item()

    rich.print(metrics)


if __name__ == "__main__":
    main()
