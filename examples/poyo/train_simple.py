from torch_brain.transforms import UnitDropout
from argparse import ArgumentParser
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
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
from datasets.nlb import PoyoNLBDataset
from datasets.poyo_mp import PoyoMPDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    cfg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    run = wandb.init()
    seed_everything(cfg.seed)

    # Datasets
    unit_dropout = UnitDropout(max_units=300, min_units=30, mode_units=100, peak=4)
    train_ds = PoyoMPDataset(cfg.data_root, transform=unit_dropout)
    eval_ds = PoyoMPDataset(root=cfg.data_root)
    logger.info(
        f"Dataset: num_recordings={len(train_ds.recording_ids)}, "
        f"num_units={len(train_ds.get_unit_ids())}"
    )

    # Model
    model = POYO(
        sequence_length=1.0,
        latent_step=0.125,
        num_latents_per_step=32,
        dim=128,
        dim_out=train_ds.dim_target,
        dim_head=64,
        depth=24,
        cross_heads=4,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.2,
    )
    model.init_vocabs(train_ds)
    train_ds.tokenizer = model.tokenize
    eval_ds.tokenizer = model.tokenize

    # Samplers
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_ds.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )
    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=eval_ds.get_sampling_intervals("valid"),
        window_length=model.sequence_length,
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
        eval_ds,
        sampler=val_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate,
    )

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Train loop
    model = model.to(device)
    step = 0
    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):

        # Train epoch
        model.train()
        loader_pbar = tqdm(train_loader, leave=False)
        for X, Y in loader_pbar:
            X, Y = move_to_device((X, Y), device)
            mask = Y["output_mask"]
            pred = model(**X, output_timestamps=Y["timestamps"])[mask]

            target = Y["values"][mask]
            loss = F.mse_loss(pred, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            loader_pbar.set_description(f"Loss: {loss.item():.3f}")
            run.log(
                {"train/loss": loss.item(), "train/step": step, "train/epoch": epoch}
            )

            step += 1

        # Validation epoch
        metric_fn = torchmetrics.functional.r2_score
        model.eval()
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
            run.log(
                {
                    **metrics,
                    "train/step": step,
                    "train/epoch": epoch,
                }
            )


if __name__ == "__main__":
    main()
