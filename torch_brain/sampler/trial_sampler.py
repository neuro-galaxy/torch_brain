from typing import Dict, Optional

import torch
from torch_brain.dataset import DatasetIndex
from temporaldata import Interval


class TrialSampler(torch.utils.data.Sampler):
    r"""Samples complete trial intervals without windowing.

    Unlike :class:`RandomFixedWindowSampler` and :class:`SequentialFixedWindowSampler`,
    which slice continuous recordings into fixed-length windows, :class:`TrialSampler`
    treats each individual interval in :obj:`sampling_intervals` as a complete trial and
    yields one :class:`~torch_brain.dataset.DatasetIndex` per trial. This is suited for
    trial-based experimental paradigms where each trial has a well-defined start and end
    time that should be preserved.

    Args:
        sampling_intervals: Sampling intervals for each session.
            Each individual interval within the session's :class:`temporaldata.Interval`
            object is treated as one trial. Typically obtained from
            :meth:`~torch_brain.dataset.Dataset.get_sampling_intervals`.
        shuffle: If ``True``, trials are yielded in a randomly shuffled order.
            If ``False`` (default), trials are yielded in the order they appear in
            :obj:`sampling_intervals`.
        generator (Optional[torch.Generator]): Optional RNG used when
            :obj:`shuffle=True`. If ``None`` (default), uses the default global PyTorch generator.

    Example::

        >>> from temporaldata import Interval
        >>> import numpy as np
        >>> from torch_brain.sampler import TrialSampler

        >>> sampling_intervals = {
        ...     "session_1": Interval(np.array([0.0, 5.0, 10.0]), np.array([2.0, 8.0, 15.0])),
        ... }
        >>> sampler = TrialSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     shuffle=True,
        ... )
        >>> len(sampler)
        3
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ):
        self.sampling_intervals = sampling_intervals
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        r"""Returns the total number of trials across all sessions."""
        return sum(len(intervals) for intervals in self.sampling_intervals.values())

    def __iter__(self):
        r"""Yields one :class:`~torch_brain.dataset.DatasetIndex` per trial, optionally shuffled."""
        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, intervals in self.sampling_intervals.items()
            for start, end in intervals
        ]

        if self.shuffle:
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices
