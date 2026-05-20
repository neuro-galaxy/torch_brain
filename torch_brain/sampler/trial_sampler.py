from typing import List, Dict, Tuple, Optional

import torch
from torch_brain.dataset import DatasetIndex
from temporaldata import Interval


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        sampling_intervals (Dict[str, Interval): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = False,
    ):
        self.sampling_intervals = sampling_intervals
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return sum(len(intervals) for intervals in self.sampling_intervals.values())

    def __iter__(self):
        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, intervals in self.sampling_intervals.items()
            for start, end in intervals
        ]

        if self.shuffle:
            # Yield a single DatasetIndex representing the selected interval
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices
