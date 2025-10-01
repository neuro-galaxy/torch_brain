from collections import defaultdict
from typing import Dict, Tuple

import lightning as L
import numpy as np
import pandas as pd
from temporaldata import Interval


# TODO move somewhere else
# The next function are utils for ndt2_custom_sampling_intervals
def sort_sessions(self, res):
    ind = np.argsort([int(e.split("-")[1]) for e in res])
    return [res[i] for i in ind]


def ndt2_eval_split(self, ses_keys):
    cfg = self.cfg
    nb_sessions = len(ses_keys)
    df = pd.DataFrame([0] * nb_sessions)
    eval_subset = df.sample(frac=cfg.eval_ratio, random_state=cfg.eval_seed)
    eval_keys = [ses_keys[i] for i in eval_subset.index]
    non_eval_keys = [ses_keys[i] for i in df.index.difference(eval_subset.index)]
    return self.sort_sessions(eval_keys), self.sort_sessions(non_eval_keys)


def ndt2_limit_per_session(self, ses_keys):
    cfg = self.cfg
    nb_sessions = len(ses_keys)
    df = pd.DataFrame([0] * nb_sessions)
    subset = df.sample(cfg.limit_per_eval_session)
    ses_keys = [ses_keys[i] for i in subset.index]
    return self.sort_sessions(ses_keys)


def ndt2_custom_sampling_intervals(self) -> Tuple[Dict, Dict]:
    """
    Custom sampling intervals for NDT2.
    It splits the dataset into training and validation sets.
    Note: Used at the sampling level and not at the session level.
    This is because ndt2 split at the dataset object level and not at session level.
    """
    ses_keys = []
    dataset = self.dataset
    ctx_time = self.cfg.ctx_time
    train_ratio = self.cfg.train_ratio
    seed = self.cfg.split_seed

    for ses_id, ses in dataset._data_objects.items():
        nb_trials = int(ses.domain.end[-1] - ses.domain.start[0])
        for i in range(nb_trials):
            ses_keys.append(f"{ses_id}-{i}")

    if self.cfg.get("is_eval", False):
        ses_keys = self.sort_sessions(ses_keys)
        eval_keys, ses_keys = self.ndt2_eval_split(ses_keys)
        ses_keys = self.ndt2_limit_per_session(ses_keys)

    L.seed_everything(seed)
    np.random.shuffle(ses_keys)
    tv_cut = int(train_ratio * len(ses_keys))
    train_keys, val_keys = ses_keys[:tv_cut], ses_keys[tv_cut:]

    def get_dict(keys):
        d = defaultdict(list)
        for k in keys:
            # ses_id, trial = k.split("-")
            trial = k.split("-")[-1]
            ses_id = "-".join(k.split("-")[:-1])
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

    # TODO this is very dirty code should be cleaned
    def list_to_inter(l):
        start = np.array([e[0] for e in l])
        end = np.array([e[1] for e in l])
        return Interval(start, end)

    def to_inter(d):
        return {k: list_to_inter(v) for k, v in d.items()}

    train_sampling_intervals = to_inter(train_sampling_intervals)
    val_sampling_intervals = to_inter(val_sampling_intervals)

    eval_sampling_intervals = None
    if self.cfg.get("is_eval", False):
        eval_sampling_intervals = get_dict(eval_keys)
        eval_sampling_intervals = to_inter(eval_sampling_intervals)

    return train_sampling_intervals, val_sampling_intervals, eval_sampling_intervals


# def extract_chan_nb(self, units: ArrayDict):
#     channel_names = units.channel_name
#     res = [int(chan_name.split(b" ")[-1]) for chan_name in channel_names]
#     return np.array(res) - 1
