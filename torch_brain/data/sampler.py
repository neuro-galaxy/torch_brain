import logging
import math
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import torch

from torch_brain.data.dataset import DatasetIndex


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`interval_dict` parameter. :obj:`interval_dict` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied by default.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        generator: Optional[torch.Generator],
        drop_short: bool = True,
        enable_jitter: bool = True,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short
        self.enable_jitter = enable_jitter

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.interval_dict.items():
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
        for session_name, sampling_intervals in self.interval_dict.items():
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

                left_offset = 0
                if self.enable_jitter:
                    # sample a random offset
                    left_offset = (
                        torch.rand(1, generator=self.generator).item()
                        * self.window_length
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
    sampling intervals are defined in the :obj:`interval_dict` parameter.
    :obj:`interval_dict` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (Optional[float], optional): Step size between windows. If None, it
            defaults to `window_length`. Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.interval_dict = interval_dict
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

        for session_name, sampling_intervals in self.interval_dict.items():
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
        return len(self._indices)

    def __iter__(self):
        yield from self._indices


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
    ):
        self.interval_dict = interval_dict
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return sum(len(intervals) for intervals in self.interval_dict.values())

    def __iter__(self):
        # Flatten the intervals from all sessions into a single list
        all_intervals = [
            (session_id, start, end)
            for session_id, intervals in self.interval_dict.items()
            for start, end in intervals
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
    Example:
        >>> sampler = SequentialFixedWindowSampler(interval_dict, window_length=10)
        >>> dist_sampler = DistributedSamplerWrapper(sampler)
        >>> loader = torch.utils.data.DataLoader(dataset, sampler=dist_sampler)
        # Before starting the training loop, set the rank and num_replicas attributes:
        >>> dist_sampler.set_params(trainer.world_size, trainer.global_rank)
        # Now use it
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
