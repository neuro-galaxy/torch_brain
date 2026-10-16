import numpy as np
import pytest
import torch

from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex
from torch_brain.samplers import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
    SessionWiseBatchSampler,
    TrialSampler,
)


# helper
class _FakeSampler(torch.utils.data.Sampler[DatasetIndex]):
    """Yields a fixed, pre-defined list of DatasetIndex objects, in order."""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# helper
class _CountingSampler(torch.utils.data.Sampler[DatasetIndex]):
    """Yields indices shifted by an offset that increments on every __iter__
    call, to simulate a stateful/random inner sampler for cache tests."""

    def __init__(self, indices):
        self.indices = indices
        self.call_count = 0

    def __iter__(self):
        offset = self.call_count
        self.call_count += 1
        return iter(
            DatasetIndex(idx.recording_id, idx.start + offset, idx.end + offset)
            for idx in self.indices
        )

    def __len__(self):
        return len(self.indices)


# helper
def compare_slice_indices(a, b):
    return (
        (a.recording_id == b.recording_id)
        and np.isclose(a.start, b.start)
        and np.isclose(a.end, b.end)
    )


# helper
def samples_in_sampling_intervals(samples, sampling_intervals):
    for s in samples:
        assert s.recording_id in sampling_intervals
        allowed_intervals = sampling_intervals[s.recording_id]
        if not (
            sum(
                [
                    (s.start >= start) and (s.end <= end)
                    for start, end in zip(
                        allowed_intervals.start, allowed_intervals.end, strict=True
                    )
                ]
            )
            == 1
        ):
            return False

    return True


def test_sequential_sampler():
    sampler = SequentialFixedWindowSampler(
        sampling_intervals={
            "session1": Interval(
                start=np.array([0.0, 3.0]),
                end=np.array([2.0, 4.5]),
            ),
            "session2": Interval(
                start=np.array([0.1, 2.5, 15.0]),
                end=np.array([1.25, 5.0, 18.7]),
            ),
            "session3": Interval(
                start=np.array([1000.0]),
                end=np.array([1002.0]),
            ),
        },
        window_length=1.1,
        step=0.75,
    )
    assert len(sampler) == 18

    s_iter = iter(sampler)
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0.0, 1.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0.75, 1.85))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0.9, 2.0))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 3.0, 4.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 3.4, 4.5))
    #
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 0.1, 1.2))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 0.15, 1.25))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 2.5, 3.6))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 3.25, 4.35))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 3.9, 5.0))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 15.0, 16.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 15.75, 16.85))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 16.5, 17.6))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 17.25, 18.35))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 17.6, 18.7))
    #
    assert compare_slice_indices(next(s_iter), DatasetIndex("session3", 1000.0, 1001.1))
    assert compare_slice_indices(
        next(s_iter), DatasetIndex("session3", 1000.75, 1001.85)
    )
    assert compare_slice_indices(next(s_iter), DatasetIndex("session3", 1000.9, 1002.0))


def test_random_sampler():

    sampling_intervals = {
        "session1": Interval(
            start=np.array([0.0, 3.0]),
            end=np.array([2.0, 4.5]),
        ),  # 3
        "session2": Interval(
            start=np.array([0.1, 2.5, 15.0]),
            end=np.array([1.25, 5.0, 18.7]),
        ),  # 7
        "session3": Interval(
            start=np.array([1000.0]),
            end=np.array([1002.0]),
        ),  # 2
    }

    sampler = RandomFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    assert len(sampler) == 9

    # sample and check that all indices are within the expected range
    samples = list(sampler)
    assert len(samples) == 9
    assert samples_in_sampling_intervals(samples, sampling_intervals)

    # sample again and check that the indices are different this time
    samples2 = list(sampler)
    assert len(samples) == 9
    for s1 in samples:
        for s2 in samples2:
            assert not compare_slice_indices(s1, s2)

    # Test "index in valid range" when step > window_length
    sampler = RandomFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    samples = list(sampler)
    assert samples_in_sampling_intervals(samples, sampling_intervals)

    # Having window_length bigger than any interval should raise an error
    with pytest.raises(ValueError):
        sampler = RandomFixedWindowSampler(
            sampling_intervals=sampling_intervals,
            window_length=5,
            generator=torch.Generator().manual_seed(42),
        )

        len(sampler)


def test_trial_sampler():
    sampling_intervals = {
        "session1": Interval(
            start=np.array([0.0, 3.0]),
            end=np.array([2.0, 4.5]),
        ),
        "session2": Interval(
            start=np.array([0.1, 2.5, 15.0]),
            end=np.array([1.25, 5.0, 18.7]),
        ),
        "session3": Interval(
            start=np.array([1000.0, 1002.0]),
            end=np.array([1002.0, 1003.0]),
        ),
    }

    sampler = TrialSampler(
        sampling_intervals=sampling_intervals,
        shuffle=True,
    )
    assert len(sampler) == 7

    # Check that the sampled interval is within the expected range
    samples = list(sampler)
    assert len(samples) == 7
    assert samples_in_sampling_intervals(samples, sampling_intervals)

    # With the same seed, the sampler should always give the same outputs.
    sampler1 = TrialSampler(
        sampling_intervals=sampling_intervals,
        generator=torch.Generator().manual_seed(42),
        shuffle=True,
    )
    sampler2 = TrialSampler(
        sampling_intervals=sampling_intervals,
        generator=torch.Generator().manual_seed(42),
        shuffle=True,
    )
    samples1 = list(sampler1)
    samples2 = list(sampler2)
    assert compare_slice_indices(samples1[0], samples2[0])

    # There should be that specific slice somewhere
    # (though unlikely to be in position 0).
    matches = []
    for sample in samples1:
        matches.append(
            compare_slice_indices(sample, DatasetIndex("session1", 0.0, 2.0))
        )

    assert len([x for x in matches if x]) == 1 and not matches[0]

    # Do this again, with the sequential sampler
    sampler1 = TrialSampler(sampling_intervals=sampling_intervals, shuffle=False)
    samples1 = list(sampler1)
    assert compare_slice_indices(samples1[0], DatasetIndex("session1", 0.0, 2.0))


def test_session_batch_sampler():
    # 4 indices for session1, 3 for session2, interleaved to check that the
    # batch sampler groups them correctly regardless of the inner sampler's order.
    indices = [
        DatasetIndex("session1", 0.0, 1.0),
        DatasetIndex("session2", 0.0, 1.0),
        DatasetIndex("session1", 1.0, 2.0),
        DatasetIndex("session2", 1.0, 2.0),
        DatasetIndex("session1", 2.0, 3.0),
        DatasetIndex("session2", 2.0, 3.0),
        DatasetIndex("session1", 3.0, 4.0),
    ]

    # batch_size must be positive
    with pytest.raises(ValueError):
        SessionWiseBatchSampler(_FakeSampler(indices), batch_size=0)

    # drop_last=True: session1 (4 items) -> 2 batches of 2; session2 (3 items)
    # -> 1 batch of 2, remainder of 1 dropped.
    sampler = SessionWiseBatchSampler(
        _FakeSampler(indices), batch_size=2, drop_last=True
    )
    assert len(sampler) == 3

    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 2
        # every index in a batch must belong to the same session
        assert len({idx.recording_id for idx in batch}) == 1

    session_batch_counts = {"session1": 0, "session2": 0}
    for batch in batches:
        session_batch_counts[batch[0].recording_id] += 1
    assert session_batch_counts == {"session1": 2, "session2": 1}

    # drop_last=False: session1 -> 2 batches of 2; session2 -> 1 batch of 2
    # and 1 batch of 1 (the remainder is kept).
    sampler = SessionWiseBatchSampler(
        _FakeSampler(indices), batch_size=2, drop_last=False
    )
    assert len(sampler) == 4

    batches = list(sampler)
    assert len(batches) == 4
    for batch in batches:
        assert len({idx.recording_id for idx in batch}) == 1

    batch_sizes_by_session = {"session1": [], "session2": []}
    for batch in batches:
        batch_sizes_by_session[batch[0].recording_id].append(len(batch))
    assert sorted(batch_sizes_by_session["session1"]) == [2, 2]
    assert sorted(batch_sizes_by_session["session2"]) == [1, 2]

    # len() should stay consistent with a second pass over the sampler
    # (cache_batches defaults to True, so both calls see the same batches).
    assert len(sampler) == len(list(sampler))


def test_session_batch_sampler_shuffle():
    torch.manual_seed(0)
    indices = [
        DatasetIndex(f"session{s}", float(i), float(i) + 1.0)
        for s in range(3)
        for i in range(4)
    ]

    def batch_signature(batches):
        return [
            tuple((idx.recording_id, idx.start, idx.end) for idx in batch)
            for batch in batches
        ]

    # shuffle=False (default): batches are always yielded in build order.
    batch_sampler = SessionWiseBatchSampler(
        _FakeSampler(indices), batch_size=2, shuffle=False
    )
    order1 = batch_signature(list(batch_sampler))
    order2 = batch_signature(list(batch_sampler))
    assert order1 == order2

    # shuffle=True: the batch order changes across successive epochs.
    batch_sampler = SessionWiseBatchSampler(
        _FakeSampler(indices), batch_size=2, shuffle=True
    )
    shuffled1 = batch_signature(list(batch_sampler))
    shuffled2 = batch_signature(list(batch_sampler))
    assert sorted(shuffled1) == sorted(shuffled2)  # same set of batches
    assert shuffled1 != shuffled2  # different order


def test_session_batch_sampler_cache_batches():
    indices = [
        DatasetIndex("session1", 0.0, 1.0),
        DatasetIndex("session2", 0.0, 1.0),
        DatasetIndex("session1", 1.0, 2.0),
        DatasetIndex("session2", 1.0, 2.0),
    ]

    # NOTE: we deliberately iterate via `list(iter(batch_sampler))` rather than
    # `list(batch_sampler)`. The latter triggers CPython's length_hint
    # optimization, which calls len(batch_sampler) (and therefore
    # _prepare_cache()) before __iter__ is consumed, causing a spurious extra
    # rebuild when cache_batches=False. Iterating an explicit iterator avoids
    # that.

    # cache_batches=True (default): batches are built once from the inner
    # sampler and reused verbatim across epochs, even if the inner sampler
    # is stateful and would otherwise yield different indices each time.
    inner = _CountingSampler(indices)
    batch_sampler = SessionWiseBatchSampler(inner, batch_size=2, cache_batches=True)
    batches1 = list(iter(batch_sampler))
    batches2 = list(iter(batch_sampler))
    assert batches1 == batches2
    assert inner.call_count == 1

    # cache_batches=False: batches are rebuilt on every __iter__ call, so a
    # stateful inner sampler produces different batches each epoch.
    inner = _CountingSampler(indices)
    batch_sampler = SessionWiseBatchSampler(inner, batch_size=2, cache_batches=False)
    batches1 = list(iter(batch_sampler))
    batches2 = list(iter(batch_sampler))
    assert batches1 != batches2
    assert inner.call_count == 2
