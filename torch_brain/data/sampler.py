import math
import logging
from typing import List, Dict, Tuple, Optional, TypeVar, Iterator
from functools import cached_property
from collections import defaultdict

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


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps a sampler to be used in a distributed setting. This sampler will
    only return indices that are assigned to the current process based on the
    rank and num_replicas.

    Args:
        sampler (torch.utils.data.Sampler): The original sampler to wrap.
        num_replicas (int): Number of processes participating in the distributed
            training.
        rank (int): Rank of the current process.

    Example ::

        >>> from torch_brain.data.sampler import SequentialFixedWindowSampler, DistributedSamplerWrapper

        >>> sampling_intervals = {
        ...     "session_1": Interval(0, 100),
        ...     "session_2": Interval(0, 100),
        ... }

        >>> sampler = SequentialFixedWindowSampler(sampling_intervals=sampling_intervals, window_length=10)
        >>> dist_sampler = DistributedSamplerWrapper(sampler)

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


class StitcherSamplerWrapper(torch.utils.data.Sampler):
    def __init__(self, sampler, num_replicas=1, rank=0):

        self.num_replicas = num_replicas
        self.rank = rank

        # TODO: see how lightning detects distributed samplers
        if hasattr(sampler, "set_epoch"):
            raise ValueError(
                f"StitcherSamplerWrapper cannot wrap a sampler that is already a "
                f"distributed sampler. Sampler {sampler} has set_epoch method which "
                f"implies that it is a distributed sampler."
            )

        # if in distributed mode, check that the sampler is not random
        if num_replicas > 1:
            # run sampler twice to check if it's random
            first_run = list(sampler)
            second_run = list(sampler)

            # verify each DatasetIndex object is identical between runs
            is_random = len(first_run) != len(second_run)
            if not is_random:
                for idx1, idx2 in zip(first_run, second_run):
                    if (
                        idx1.recording_id != idx2.recording_id
                        or idx1.start != idx2.start
                        or idx1.end != idx2.end
                    ):
                        is_random = True
                        break

            if is_random:
                raise ValueError(
                    f"StitcherSamplerWrapper cannot wrap a random sampler. "
                    f"Sampler {sampler} produced different samples on repeated calls, "
                    f"which implies that it is a random sampler."
                )

        self.indices, self.sequence_index = self._process_indices(list(sampler))

    def _process_indices(self, indices):
        # first, we group indices by recording id
        indices_grouped_by_recording = defaultdict(list)

        for index in indices:
            indices_grouped_by_recording[index.recording_id].append(index)

        # next, we want to group together the indices that correspond to windows that will
        # overlap with each other, and will need to be stitched together
        indices_groups = [[]]
        for indices in indices_grouped_by_recording.values():
            # sort indices by start time
            indices.sort(key=lambda x: x.start)

            # find overlapping indices linearly since they're sorted
            i = 0
            # the first index will always start a new group
            indices_groups.append([indices[i]])
            while i < len(indices) - 1:
                # we assume that intervals can't contain each other. in other words,
                # that an interval must end before the next interval ends.
                if indices[i].end < indices[i + 1].start:
                    # start a new group
                    indices_groups.append([indices[i + 1]])
                else:
                    # add to the last group
                    indices_groups[-1].append(indices[i + 1])
                i += 1

        # now we want to group the indices by rank
        group_sizes = [len(indices_groups[i]) for i in range(len(indices_groups))]

        # sort groups by size in descending order for better load balancing
        sorted_indices = torch.argsort(torch.tensor(group_sizes), descending=True)
        indices_groups = [indices_groups[i] for i in sorted_indices]
        group_sizes = [group_sizes[i] for i in sorted_indices]

        # track total windows per rank for load balancing
        rank_sizes = [0] * self.num_replicas

        # assign indices to ranks to minimize imbalance
        reordered_indices = []
        sequence_index = []
        sequence_index_ptr = 0
        for indices in indices_groups:
            # assign to rank with fewest windows
            target_rank = min(range(self.num_replicas), key=lambda r: rank_sizes[r])
            if target_rank == self.rank:
                reordered_indices.extend(indices)
                sequence_index.extend([sequence_index_ptr for _ in range(len(indices))])
                sequence_index_ptr += 1

            rank_sizes[target_rank] += len(indices)

        sequence_index = torch.tensor(sequence_index)
        return reordered_indices, sequence_index

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number. Not strictly necessary for sequential sampler
        but included for API compatibility."""
        self.epoch = epoch
