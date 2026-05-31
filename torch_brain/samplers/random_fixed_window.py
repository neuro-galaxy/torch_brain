import math
import logging
from typing import Dict
from functools import cached_property

import torch

from torch_brain.data import Interval
from torch_brain.dataset import DatasetIndex


class RandomFixedWindowSampler(torch.utils.data.Sampler[DatasetIndex]):
    r"""Samples fixed-length windows randomly from a collection of time intervals.

    Given the :obj:`sampling_intervals` dictionary mapping session IDs to
    :class:`temporaldata.Interval` objects, this sampler produces
    :class:`~torch_brain.dataset.DatasetIndex` objects for indexing a
    :class:`~torch_brain.dataset.Dataset`. Each call to :meth:`__iter__` applies a
    fresh random temporal jitter and re-shuffles the windows, so every epoch explores
    slightly different positions within each interval.

    In one epoch, the number of samples generated from a single contiguous interval
    of length :math:`L` is:

    .. math::
        N = \left\lfloor\frac{L}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals: Sampling intervals for each session.
            Typically obtained from
            :meth:`~torch_brain.dataset.Dataset.get_sampling_intervals`.
        window_length: Duration of each sampled window in seconds.
        generator: Optional RNG used for jitter and
            shuffling. If ``None`` (default), uses the default global PyTorch generator.
        drop_short: If ``True`` (default), intervals shorter than
            :obj:`window_length` are silently skipped with a warning logged. If
            ``False``, a :exc:`ValueError` is raised for any short interval.

    Example::

        >>> import numpy as np
        >>> from torch_brain.data import Interval
        >>> from torch_brain.samplers import RandomFixedWindowSampler

        >>> sampling_intervals = {
        ...     "session_1": Interval(0.0, 100.0),
        ...     "session_2": Interval(0.0, 200.0),
        ... }
        >>> sampler = RandomFixedWindowSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     window_length=1.0,
        ... )
        >>> len(sampler)
        300
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: torch.Generator | None = None,
        drop_short: bool = True,
    ):
        if window_length <= 0:
            raise ValueError("window_length must be greater than 0.")

        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self) -> int:
        num_samples = 0
        total_short_dropped = 0.0

        for intervals in self.sampling_intervals.values():
            for start, end in zip(intervals.start, intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        r"""Returns the estimated number of samples per epoch across all sessions."""
        return self._estimated_len

    def __iter__(self):
        r"""Yields shuffled :class:`~torch_brain.dataset.DatasetIndex` objects with random temporal jitter."""
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start + left_offset,
                        end,
                        self.window_length,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )

        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]
