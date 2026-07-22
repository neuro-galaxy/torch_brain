from collections.abc import Iterator

import torch

from torch_brain.datasets import DatasetIndex


class SessionBatchSampler(torch.utils.data.Sampler[list[DatasetIndex]]):
    r"""Wraps any sampler yielding :class:`~torch_brain.datasets.DatasetIndex` and
    groups its output into per-session batches.

    All indices in a batch share the same session id. The inner sampler controls
    the order of indices within each session; :obj:`shuffle_batches` controls the
    order of batches across sessions.

    Args:
        sampler: Inner sampler whose :meth:`__iter__` yields
            :class:`~torch_brain.datasets.DatasetIndex` objects.
        batch_size: Number of samples per batch.
        shuffle_batches: If ``True`` (default), the final batch order is shuffled.
            If ``False``, batches are yielded in the order they were built.
        generator: Optional RNG used when :obj:`shuffle_batches` is ``True``.
            If ``None`` (default), uses the default global PyTorch generator.
        drop_last: If ``True`` (default), the last incomplete batch for each
            session is dropped.

    Example::

        >>> from torch_brain.data import Interval
        >>> from torch_brain.samplers import RandomFixedWindowSampler, SessionBatchSampler

        >>> sampling_intervals = {
        ...     "session_1": Interval(0.0, 100.0),
        ...     "session_2": Interval(0.0, 200.0),
        ... }
        >>> inner_sampler = RandomFixedWindowSampler(
        ...     sampling_intervals=sampling_intervals,
        ...     window_length=1.0,
        ... )
        >>> batch_sampler = SessionBatchSampler(inner_sampler, batch_size=16)
        >>> loader = torch.utils.data.DataLoader(
        ...     dataset=your_dataset,
        ...     batch_sampler=batch_sampler,
        ... )
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler[DatasetIndex],
        batch_size: int,
        *,
        shuffle_batches: bool = True,
        generator: torch.Generator | None = None,
        drop_last: bool = True,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.generator = generator
        self.drop_last = drop_last
        self._batches_cache: list[list[DatasetIndex]] | None = None

    def _build_batches(self) -> list[list[DatasetIndex]]:
        indices_by_session: dict[str, list[DatasetIndex]] = {}
        for idx in self.sampler:
            session_id = idx.recording_id
            if session_id not in indices_by_session:
                indices_by_session[session_id] = []
            indices_by_session[session_id].append(idx)

        batches: list[list[DatasetIndex]] = []
        for indices in indices_by_session.values():
            max_len = len(indices)
            if self.drop_last:
                max_len = (max_len // self.batch_size) * self.batch_size
            for i in range(0, max_len, self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        return batches

    def _prepare_cache(self) -> None:
        if self._batches_cache is None:
            self._batches_cache = self._build_batches()

    def __len__(self) -> int:
        r"""Returns the total number of batches across all sessions."""
        self._prepare_cache()
        return len(self._batches_cache)

    def __iter__(self) -> Iterator[list[DatasetIndex]]:
        r"""Returns an iterator over per-session batches of
        :class:`~torch_brain.datasets.DatasetIndex`."""
        self._prepare_cache()
        batches = self._batches_cache
        self._batches_cache = None  # reset so next epoch rebuilds
        return self._yield_batches(batches)

    def _yield_batches(
        self, batches: list[list[DatasetIndex]]
    ) -> Iterator[list[DatasetIndex]]:
        if self.shuffle_batches and batches:
            for idx in torch.randperm(len(batches), generator=self.generator).tolist():
                yield batches[idx]
        else:
            yield from batches
