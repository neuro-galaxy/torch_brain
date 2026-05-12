import os
import logging
import random
from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor

from torch_brain.utils.stitcher import stitch
from torch_brain.models import POYO
from torch_brain.optim import SparseLamb

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    if seed is not None:
        log.info(f"Global seed: {seed}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, (str, int, float, bool, type(None), np.ndarray)):
        # Metadata/scalars do not need device transfer.
        return data
    else:
        raise TypeError(f"Unknown data type {type(data)}")


class BehaviorStitcher:
    def __init__(self):
        self.pred_cache = []
        self.target_cache = []
        self.timestamps_cache = []

    @torch.no_grad
    def update(self, preds: Tensor, targets: Tensor, timestamps: Tensor):
        self.pred_cache.append(preds)
        self.target_cache.append(targets)
        self.timestamps_cache.append(timestamps)

    @torch.no_grad
    def compute(self) -> tuple[Tensor, Tensor]:
        preds = torch.concat(self.pred_cache)
        targets = torch.concat(self.target_cache)
        timestamps = torch.concat(self.timestamps_cache)

        t1, preds = stitch(timestamps, preds)
        t2, targets = stitch(timestamps, targets)
        assert torch.allclose(t1, t2)
        return preds, targets

    def reset(self):
        self.pred_cache = []
        self.target_cache = []
        self.timestamps_cache = []


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
            {"params": emb_params, "sparse": True, "name": "embeddings"},
            {"params": nonemb_params, "name": "weights"},
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
