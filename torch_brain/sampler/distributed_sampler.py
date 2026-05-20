import logging
from typing import List, Dict, Optional

import torch
import torch.distributed as dist

from temporaldata import Interval
from torch_brain.dataset import DatasetIndex


class DistributedEvaluationSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps a sampler to be used in a distributed evaluation setting. Unlike the standard
    distributed samplers from PyTorch and PyTorch Lightning which ensure equal samples per rank
    by potentially dropping samples, this sampler preserves all samples by distributing them
    across ranks without dropping any, which is important to guarantee that evaluation is done
    on the complete dataset.

    .. warning::
        This wrapper assumes that there is no communication between ranks except at the
        beginning or end of the evaluation, so it is only suitable for standard evaluation.
        This is because some ranks might end up performing more steps than others.

    Args:
        sampler (torch.utils.data.Sampler): The original sampler to wrap.
        num_replicas (int): Number of processes participating in the distributed
            evaluation.
        rank (int): Rank of the current process.

    Example ::

        >>> from torch_brain.sampler import SequentialFixedWindowSampler, DistributedEvaluationSamplerWrapper

        >>> sampling_intervals = {
        ...     "session_1": Interval(0, 100),
        ...     "session_2": Interval(0, 100),
        ... }

        >>> sampler = SequentialFixedWindowSampler(sampling_intervals=sampling_intervals, window_length=10)
        >>> dist_sampler = DistributedEvaluationSamplerWrapper(sampler)

    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def set_params(self, num_replicas, rank):
        logging.info(
            f"Setting distributed sampler params: "
            f"num_replicas={num_replicas}, rank={rank}"
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def _check_params(self):
        return (self.num_replicas is not None) and (self.rank is not None)

    def rank_len(self):
        r"""Returns the number of samples assigned to the current process."""
        total_len = len(self.sampler)
        evenly_split = total_len // self.num_replicas
        extra = int((total_len % self.num_replicas) < self.rank)
        return evenly_split + extra

    def __len__(self):
        r"""Returns the number of samples assigned to the current process if
        the rank and num_replicas are set. Otherwise, returns the total number
        of samples in the original sampler.
        """
        if not self._check_params():
            return len(self.sampler)
        else:
            return self.rank_len()

    def __iter__(self):
        assert (
            self._check_params()
        ), "Rank and num_replicas must be set before using the distributed sampler."
        indices = list(self.sampler)
        indices = indices[self.rank : len(indices) : self.num_replicas]
        return iter(indices)


class DistributedStitchingFixedWindowSampler(torch.utils.data.DistributedSampler):
    r"""A sampler designed specifically for evaluation that enables sliding window
    inference with prediction stitching across distributed processes.

    This sampler divides sequences into overlapping windows and distributes them across
    processes for parallel inference, it keeps windows that need to be stitched together
    on the same rank, to allow stitching on that same rank without communication.

    Additionally, it will keep track of the windows that need to be stitched together to
    allow for stitching as soon as all windows from the same contiguous sequence are
    available. This information can be passed to the stitcher which can stitch and compute
    a metric for the sequence as soon as all windows from that sequence are available,
    allowing it to free up memory quickly.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset. Each interval is defined by a start and end time.
        window_length (float): Length of the sliding window.
        step (Optional[float], optional): Step size between windows. If None, defaults
            to window_length. Smaller steps create more overlap between windows.
        batch_size (int): Number of windows to process in each batch.
        num_replicas (Optional[int], optional): Number of processes participating in
            distributed inference. If None, will be set using torch.distributed.
        rank (Optional[int], optional): Rank of the current process. If None, will be
            set using torch.distributed.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        step: Optional[float] = None,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
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
        self.window_length = window_length
        self.step = step or window_length
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.step <= 0:
            raise ValueError("Step must be greater than 0.")
        if self.step > self.window_length:
            raise ValueError("Step must be less than window length.")

        # Generate indices for this rank
        self.indices, self.sequence_index = self._generate_indices()
        self.num_samples = len(self.indices)

    def _generate_indices(self) -> List[DatasetIndex]:
        """Generate indices for this rank, balancing the workload across ranks based on
        the number of windows in each interval."""
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
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number. Not strictly necessary for sequential sampler
        but included for API compatibility."""
        self.epoch = epoch
