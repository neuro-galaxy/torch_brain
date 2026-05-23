import logging
from typing import List, Dict
from functools import cached_property

import torch
from temporaldata import Interval

from torch_brain.dataset import DatasetIndex


class SequentialFixedWindowSampler(torch.utils.data.Sampler[DatasetIndex]):
    r"""Samples fixed-length windows sequentially in a deterministic, reproducible order.

    Given the :obj:`sampling_intervals` dictionary mapping session IDs to
    :class:`temporaldata.Interval` objects, this sampler produces
    :class:`~torch_brain.dataset.DatasetIndex` objects in a fixed order. Windows are
    stepped through each interval using a configurable :obj:`step` size, making this
    sampler well-suited for evaluation where full coverage and reproducibility are
    required.

    If an interval's length is not an exact multiple of :obj:`step`, a final overlapping
    window is appended to ensure the entire interval is covered.

    Args:
        sampling_intervals: Sampling intervals for each session.
            Typically obtained from
            :meth:`~torch_brain.dataset.Dataset.get_sampling_intervals`.
        window_length: Duration of each sampled window in seconds.
        step: Step size between the start of consecutive windows in
            seconds. If ``None`` (default), sets :obj:`step` to :obj:`window_length` (non-overlapping
            windows).
        drop_short: If ``False`` (default), a :exc:`ValueError` is raised for any short interval.
            If ``True``, intervals shorter than :obj:`window_length` are
            silently skipped with a warning logged.

    Example::

        >>> import numpy as np
        >>> from temporaldata import Interval
        >>> from torch_brain.samplers import SequentialFixedWindowSampler

        >>> sampling_intervals = {
        ...     "session_1": Interval(
        ...         start=np.array([0.0]),
        ...         end=np.array([100.0]),
        ...     ),
        ...     "session_2": Interval(
        ...         start=np.array([0.0]),
        ...         end=np.array([100.0]),
        ...     ),
        ... }
        >>> sampler = SequentialFixedWindowSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     window_length=10.0,
        ...     step=5.0,
        ... )
        >>> len(sampler)
        38
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        step: float | None = None,
        drop_short=False,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in sampling_intervals:
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

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(start, end, self.step, dtype=torch.float64)
                    if t + self.window_length <= end
                ]

                indices.extend(indices_)

                # we need to make sure that the entire interval is covered
                if indices_[-1].end < end:
                    indices.append(
                        DatasetIndex(session_name, end - self.window_length, end)
                    )

        if self.drop_short and total_short_dropped > 0:
            num_samples = len(indices)
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")

        return indices

    def __len__(self):
        r"""Returns the total number of windows across all sessions."""
        return len(self._indices)

    def __iter__(self):
        r"""Yields :class:`~torch_brain.dataset.DatasetIndex` objects in sequential order."""
        yield from self._indices
