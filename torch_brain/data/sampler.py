import math
import logging
from typing import List, Dict, Tuple, Optional, TypeVar, Iterator
from functools import cached_property

import torch
import torch.distributed as dist

from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
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
        return self._estimated_len

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
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


class SequentialFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`sampling_intervals` parameter.
    :obj:`sampling_intervals` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (Optional[float], optional): Step size between windows. If None, it
            defaults to `window_length`. Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."
        assert self.step <= self.window_length, "Step must be less than window length."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
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
        return len(self._indices)

    def __iter__(self):
        yield from self._indices


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = False,
    ):
        self.sampling_intervals = sampling_intervals
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return sum(len(intervals) for intervals in self.sampling_intervals.values())

    def __iter__(self):
        # Flatten the intervals from all sessions into a single list
        all_intervals = [
            (session_id, start, end)
            for session_id, intervals in self.sampling_intervals.items()
            for start, end in zip(intervals.start, intervals.end)
        ]

        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, start, end in all_intervals
        ]

        if self.shuffle:
            # Yield a single DatasetIndex representing the selected interval
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices


class DistributedEvaluationSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps a sampler to be used in a distributed evaluation setting. Unlike the standard
    distributed samplers from PyTorch and PyTorch Lightning which ensure equal samples per rank
    by potentially dropping samples, this sampler preserves all samples by distributing them
    across ranks without dropping any, which is important to guarantee that evaluation is done
    on the complete dataset.

    .. warning::
        This wrapper assumes that there is no communication between ranks except at the
        begining or end of the evaluation, so it is only suitable for standard evaluation.
        This is because some ranks might end up performing more steps than others.

    Args:
        sampler (torch.utils.data.Sampler): The original sampler to wrap.
        num_replicas (int): Number of processes participating in the distributed
            evaluation.
        rank (int): Rank of the current process.

    Example ::

        >>> from torch_brain.data.sampler import SequentialFixedWindowSampler, DistributedEvaluationSamplerWrapper

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
