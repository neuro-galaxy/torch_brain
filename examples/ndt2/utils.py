from collections import defaultdict
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger

from torch_brain.data import Dataset


def balanced_accuracy_score(y_true, y_pred):
    with torch.no_grad():
        # Convert predictions to binary classes
        y_pred_classes = (y_pred > 0.5).float()

        # Confusion matrix elements
        TP = (y_pred_classes * y_true).sum().item()
        TN = ((1 - y_pred_classes) * (1 - y_true)).sum().item()
        FP = ((1 - y_true) * y_pred_classes).sum().item()
        FN = (y_true * (1 - y_pred_classes)).sum().item()

        # Sensitivity for each class
        sensitivity_pos = TP / (TP + FN)
        sensitivity_neg = TN / (TN + FP)

        # Balanced accuracy
        balanced_acc = (sensitivity_pos + sensitivity_neg) / 2
        return balanced_acc


def set_callbacks(cfg) -> List[Callback]:
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.checkpoint.enable:
        callbacks += [
            ModelCheckpoint(
                save_last=True,  # saves a checkpoint for the last epoch
                every_n_train_steps=cfg.checkpoint.every_n_steps,
                every_n_epochs=cfg.checkpoint.every_n_epochs,
                save_top_k=-1 if cfg.checkpoint.save_all else 1,
            )
        ]
    return callbacks


def wandb_set_up(cfg, log) -> Optional[WandbLogger]:
    if not cfg.wandb.enable:
        return None
    wandb_logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        save_dir=cfg.log_dir,
        log_model=False,
    )
    log.info(f"Using wandb logger: {wandb_logger.version}")

    return wandb_logger


def custom_sampling_intervals(dataset: Dataset, ctx_time=1.0, train_ratio=0.8, seed=0):
    session_cache = {}
    ses_keys = []
    for ses_id in dataset.get_session_ids():
        ses = dataset.get_session_data(ses_id)
        session_cache[ses_id] = ses
        nb_trials = int(ses.domain.end[-1] - ses.domain.start[0])
        for i in range(nb_trials):
            ses_keys.append(f"{ses_id}-{i}")

    pl.seed_everything(seed)
    np.random.shuffle(ses_keys)
    tv_cut = int(train_ratio * len(ses_keys))
    train_keys, val_keys = ses_keys[:tv_cut], ses_keys[tv_cut:]

    def get_dict(keys):
        d = defaultdict(list)
        for k in keys:
            ses_id, trial = k.split("-")
            ses = session_cache[ses_id]
            ses_start = ses.domain.start[0]
            offset = ctx_time * int(trial)
            start = ses_start + offset
            end = start + ctx_time
            d[ses_id].append((start, end))
        return dict(d)

    return get_dict(train_keys), get_dict(val_keys)
