from typing import List, Dict, Tuple

import torch
import torch.distributed as dist

from torch_brain.data import Interval
from torch_brain.dataset import DatasetIndex


class DistributedStitchingFixedWindowSampler(torch.utils.data.DistributedSampler):
    r"""Distributed sliding-window sampler that co-locates windows for prediction stitching.

    This sampler is designed for distributed evaluation with overlapping windows, where
    predictions from adjacent windows must later be stitched together. It assigns all
    windows from the same contiguous interval to the *same* rank, so stitching can be
    performed locally without any cross-rank communication.

    In addition to the window indices, the sampler exposes a :attr:`sequence_index`
    tensor that maps each window to its parent interval. A downstream stitcher can use
    this to detect when all windows of a sequence have been processed and immediately
    stitch and evaluate that sequence, keeping peak memory low.

    Intervals are assigned to ranks using a greedy load-balancing heuristic (largest
    interval first) so that the number of windows per rank stays as equal as possible.

    .. note::
        :obj:`step` must be ``<= window_length``. Use a smaller step to create
        overlapping windows for smoother stitched predictions.

    Args:
        sampling_intervals: Sampling intervals for each session.
            Typically obtained from
            :meth:`~torch_brain.dataset.Dataset.get_sampling_intervals`.
        batch_size: Number of windows per batch, used by the stitcher to track
            sequence boundaries within a batch.
        window_length: Duration of each sliding window in seconds.
        step: Stride between consecutive windows in seconds. If
            ``None`` (default), sets :obj:`step` to :obj:`window_length` (non-overlapping windows).
        num_replicas: Total number of processes. If ``None`` (default),
            resolved from :func:`torch.distributed.get_world_size`.
        rank: Rank of the current process. If ``None`` (default), resolved from
            :func:`torch.distributed.get_rank`.

    Attributes:
        sequence_index (torch.Tensor): 1-D integer tensor of length ``len(self)``
            mapping each window index to its contiguous-interval index on this rank.
            Consecutive windows sharing the same value belong to the same interval and
            will be stitched together.

    Example::

        >>> import numpy as np
        >>> from torch_brain.data import Interval
        >>> from torch_brain.samplers import DistributedStitchingFixedWindowSampler

        >>> sampling_intervals = {
        ...     "session_1": Interval(0.0, 100.0),
        ... }
        >>> sampler = DistributedStitchingFixedWindowSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     batch_size=8,
        ...     window_length=2.0,
        ...     step=1.0,
        ...     num_replicas=1,
        ...     rank=0,
        ... )
        >>> len(sampler)
        99
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        batch_size: int,
        window_length: float,
        step: float | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if window_length <= 0:
            raise ValueError("window_length must be greater than 0.")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.sampling_intervals = sampling_intervals
        self.batch_size = batch_size
        self.window_length = window_length
        self.step = window_length if step is None else step
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.step <= 0:
            raise ValueError("Step must be greater than 0.")
        if self.step > self.window_length:
            raise ValueError("Step must be less than or equal to window_length.")

        # Generate indices for this rank
        self.indices, self.sequence_index = self._generate_indices()
        self.num_samples = len(self.indices)

    def _generate_indices(self) -> Tuple[List[DatasetIndex], torch.Tensor]:
        """Build window indices for this rank using a greedy load-balancing assignment.

        Intervals are sorted by window count (largest first) and assigned to the rank
        with the fewest windows so far. All windows within an interval are kept together
        on the assigned rank to enable local stitching without cross-rank communication.

        Returns:
            Tuple of ``(indices, sequence_index)`` where ``indices`` is the list of
            :class:`~torch_brain.dataset.DatasetIndex` objects for this rank and
            ``sequence_index`` is a :class:`torch.Tensor` mapping each window to its
            parent interval index.
        """
        # first, we will compute the number of contiguous windows across all intervals
        all_intervals = []
        interval_sizes = []
        for session_name, intervals in self.sampling_intervals.items():
            for start, end in zip(intervals.start, intervals.end):
                if end - start >= self.window_length:
                    # calculate number of windows in this interval
                    num_windows = (
                        int((end - start - self.window_length + 1e-9) / self.step) + 1
                    )
                    if num_windows > 0:
                        interval_sizes.append(num_windows)
                        all_intervals.append((session_name, start, end))

        # sort intervals by size in descending order for better load balancing
        sorted_indices = torch.argsort(torch.tensor(interval_sizes), descending=True)
        all_intervals = [all_intervals[i] for i in sorted_indices]
        interval_sizes = [interval_sizes[i] for i in sorted_indices]

        # track total windows per rank for load balancing
        rank_sizes = [0] * self.num_replicas

        # assign intervals to ranks to minimize imbalance
        indices_list = []
        for session_name, start, end in all_intervals:
            # assign to rank with fewest windows
            target_rank = min(range(self.num_replicas), key=lambda r: rank_sizes[r])

            indices = []
            # generate all windows for this interval
            for t in torch.arange(
                start,
                end - self.window_length + 1e-9,
                self.step,
                dtype=torch.float64,
            ):
                t = t.item()
                indices.append(DatasetIndex(session_name, t, t + self.window_length))

            # add final window if needed
            last_start = indices[-1].start if indices else start
            if last_start + self.window_length < end:
                indices.append(
                    DatasetIndex(session_name, end - self.window_length, end)
                )

            if target_rank == self.rank:
                # only add indices to this rank
                indices_list.append(indices)

            rank_sizes[target_rank] += len(indices)

        # shuffle indices for this rank
        indices_list = [indices_list[i] for i in torch.randperm(len(indices_list))]
        indices = [item for sublist in indices_list for item in sublist]
        sequence_index = torch.tensor(
            [i for i, sublist in enumerate(indices_list) for _ in sublist]
        )

        return indices, sequence_index

    def __iter__(self):
        r"""Yields :class:`~torch_brain.dataset.DatasetIndex` objects assigned to this rank."""
        return iter(self.indices)

    def __len__(self) -> int:
        r"""Returns the number of windows assigned to this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Store the current epoch number for API compatibility.

        This sampler is deterministic and does not re-shuffle on each epoch, so
        calling this method has no effect on the produced indices. It is provided
        for compatibility with training loops that call ``sampler.set_epoch(epoch)``
        unconditionally.

        Args:
            epoch: The epoch number to record.
        """
        self.epoch = epoch
