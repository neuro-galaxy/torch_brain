import logging
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import pandas as pd
import numpy as np
import torch
import torchmetrics
import wandb

from torch_brain.data import collate
from torch_brain.models import POYO
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)

from utils import (
    seed_everything,
    move_to_device,
    BehaviorStitcher,
    create_optim,
    weighted_mse_loss_fn,
)
from datasets.wrapper import PoyoDatasetWrapper


@hydra.main(version_base="1.3", config_path="./configs")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    run = wandb.init(**cfg.wandb)
    seed_everything(cfg.seed)

    # Datasets
    train_ds = instantiate(cfg.dataset, root=cfg.data_root)
    train_ds.transform = instantiate(cfg.train_transform)

    val_ds = instantiate(cfg.dataset, root=cfg.data_root)
    val_ds.transform = instantiate(cfg.eval_transform)

    logger.info(
        f"Dataset: num_recordings={len(train_ds.recording_ids)}, "
        f"num_units={len(train_ds.get_unit_ids())}"
    )

    # Model
    model: POYO = instantiate(cfg.model, dim_out=train_ds.dim_target)
    model.init_vocabs(train_ds)
    model = model.to(device)
    train_ds = PoyoDatasetWrapper(train_ds, model.tokenize)
    val_ds = PoyoDatasetWrapper(val_ds, model.tokenize)

    # Samplers
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_ds.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )
    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=val_ds.get_sampling_intervals("valid"),
        window_length=model.sequence_length,
        step=model.sequence_length / 2.0,
    )

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        drop_last=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        sampler=val_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate,
    )

    # Optimizer
    optim, scheduler = create_optim(model, len(train_loader), cfg)

    # Train loop
    step = 0
    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        always_log = {"train/epoch": epoch, "train/step": step}

        # Train epoch
        model.train()
        loader_pbar = tqdm(train_loader, leave=False)
        for X, Y in loader_pbar:
            X, Y = move_to_device((X, Y), device)
            mask = Y["output_mask"]
            pred = model(**X, output_timestamps=Y["timestamps"])[mask]

            target = Y["values"][mask]
            loss_weights = Y["weights"][mask]
            loss = weighted_mse_loss_fn(pred, target, loss_weights)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            loader_pbar.set_description(f"Loss: {loss.item():.3f}")
            to_wandb = {"train/loss": loss.item(), **always_log}
            for param_group in optim.param_groups:
                to_wandb[f"lr/{param_group['name']}"] = param_group["lr"]
            run.log(to_wandb)

            scheduler.step()
            step += 1

        # Validation epoch
        model.eval()
        metric_fn = torchmetrics.functional.r2_score
        with torch.no_grad():
            stitchers = {rid: BehaviorStitcher() for rid in val_ds.recording_ids}

            for X, Y in tqdm(val_loader, leave=False):
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

            # Metric computation and logging
            metrics = {}
            for rid, stitcher in stitchers.items():
                pred, target = stitcher.compute()
                metrics[f"val/{rid}"] = metric_fn(pred, target).item()
            metrics["val/avg_metric"] = np.mean(list(metrics.values()))

            print(pd.Series(metrics))
            run.log({**metrics, **always_log})


if __name__ == "__main__":
    main()
