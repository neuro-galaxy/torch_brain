import logging

import torch


class DistributedEvaluationSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps any sampler for use in distributed evaluation without dropping samples.

    Unlike the standard distributed samplers from PyTorch and PyTorch Lightning, which
    ensure equal steps per rank by potentially dropping samples, this wrapper preserves
    *all* samples by interleaving them across ranks. This guarantees that evaluation
    metrics are computed over the complete dataset.

    .. warning::
        Because this wrapper does not pad to equal length, some ranks will perform more
        steps than others. There must be no inter-rank communication (e.g. allreduce)
        during the evaluation loop — only barrier-style synchronization at the start
        and end is safe.

    Rank and world-size are intentionally **not** resolved in ``__init__``; call
    :meth:`set_params` once the distributed environment is initialised before iterating.

    Args:
        sampler: The base sampler whose indices will be
            distributed across ranks.
        num_replicas: Total number of processes participating in
            evaluation. If ``None``, must be set later via :meth:`set_params`.
        rank: Rank of the current process. If ``None``, must be set
            later via :meth:`set_params`.

    Example::

        >>> from torch_brain.data import Interval
        >>> from torch_brain.samplers import SequentialFixedWindowSampler, DistributedEvaluationSamplerWrapper

        >>> sampling_intervals = {
        ...     "session_1": Interval(0.0, 100.0),
        ...     "session_2": Interval(0.0, 100.0),
        ... }
        >>> sampler = SequentialFixedWindowSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     window_length=10.0,
        ... )
        >>> dist_sampler = DistributedEvaluationSamplerWrapper(sampler)
        >>> dist_sampler.set_params(num_replicas=4, rank=0)
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def set_params(self, num_replicas: int, rank: int) -> None:
        """Configure distributed parameters after the process group has been initialised.

        Args:
            num_replicas: Total number of processes participating in evaluation.
            rank: Rank of the current process within the process group.
        """
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
        if self.num_replicas is None or self.rank is None:
            raise RuntimeError(
                "num_replicas and rank must be set before calling rank_len(). "
                "Call set_params() first."
            )
        total_len = len(self.sampler)
        evenly_split = total_len // self.num_replicas
        extra = int(self.rank < (total_len % self.num_replicas))
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
        r"""Yields the subset of indices assigned to :attr:`rank` via strided interleaving.

        Raises:
            AssertionError: If :meth:`set_params` has not been called yet.
        """
        assert self._check_params(), (
            "Rank and num_replicas must be set before using the distributed sampler."
        )
        indices = list(self.sampler)
        indices = indices[self.rank : len(indices) : self.num_replicas]
        return iter(indices)
