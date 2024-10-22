import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

import torch_brain.data
from temporaldata import Interval

log = logging.getLogger(__name__)


def ndt2_custom_sampling_intervals(
    dataset: torch_brain.data.Dataset,
    ctx_time: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[Dict, Dict]:
    """
    Custom sampling intervals for NDT2.
    It splits the dataset into training and validation sets.
    Note: Used at the sampling level and not at the session level.
    This is because ndt2 split at the dataset object level and not at session level.
    """
    ses_keys = []

    for ses_id, ses in dataset._data_objects.items():
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
            ses = dataset._data_objects[ses_id]
            ses_start = ses.domain.start[0]
            offset = ctx_time * int(trial)
            start = ses_start + offset
            end = start + ctx_time
            d[ses_id].append((start, end))
        return dict(d)

    train_sampling_intervals = get_dict(train_keys)
    val_sampling_intervals = get_dict(val_keys)

    # val will be deterministic and need to be sorted
    for v in val_sampling_intervals.values():
        v.sort()
    val_sampling_intervals = dict(sorted(val_sampling_intervals.items()))
    return train_sampling_intervals, val_sampling_intervals


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
