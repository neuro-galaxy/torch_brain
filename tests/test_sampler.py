import pytest

import numpy as np
import torch

from temporaldata import Interval

from torch_brain.data.sampler import (
    SequentialFixedWindowSampler,
    RandomFixedWindowSampler,
    TrialSampler,
)
from torch_brain.dataset import DatasetIndex


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
                        allowed_intervals.start, allowed_intervals.end
                    )
                ]
            )
            == 1
        ):
            return False

    return True


def assert_all_windows_valid(samples, window_length, sampling_intervals):
    """Check every sample has the correct window length and lies within bounds."""
    for s in samples:
        assert np.isclose(
            s.end - s.start, window_length
        ), f"Bad window length: {s.end - s.start}"
    assert samples_in_sampling_intervals(samples, sampling_intervals)


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


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestRandomFixedWindowSamplerEdgeCases:
    """Edge-case and property tests for RandomFixedWindowSampler."""

    SAMPLING_INTERVALS = {
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
    }

    def test_window_length_equals_interval(self):
        """An interval exactly equal to window_length should produce one sample."""
        intervals = {
            "s": Interval(start=np.array([5.0]), end=np.array([7.0])),
        }
        sampler = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=2.0,
            generator=torch.Generator().manual_seed(0),
        )
        samples = list(sampler)
        assert len(samples) == 1
        assert np.isclose(samples[0].start, 5.0)
        assert np.isclose(samples[0].end, 7.0)

    def test_all_windows_have_correct_length(self):
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.SAMPLING_INTERVALS,
            window_length=1.1,
            generator=torch.Generator().manual_seed(99),
        )
        samples = list(sampler)
        assert_all_windows_valid(samples, 1.1, self.SAMPLING_INTERVALS)

    def test_consistent_epoch_length_across_seeds(self):
        """Epoch length must be deterministic regardless of the random jitter."""
        lengths = set()
        for seed in range(20):
            sampler = RandomFixedWindowSampler(
                sampling_intervals=self.SAMPLING_INTERVALS,
                window_length=1.1,
                generator=torch.Generator().manual_seed(seed),
            )
            lengths.add(len(list(sampler)))
        assert len(lengths) == 1, f"Epoch lengths vary across seeds: {lengths}"

    def test_drop_short_true(self):
        intervals = {
            "s": Interval(
                start=np.array([0.0, 10.0]),
                end=np.array([0.5, 15.0]),
            ),
        }
        sampler = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=2.0,
            generator=torch.Generator().manual_seed(0),
            drop_short=True,
        )
        samples = list(sampler)
        for s in samples:
            assert s.start >= 10.0

    def test_drop_short_false_raises(self):
        intervals = {
            "s": Interval(start=np.array([0.0]), end=np.array([0.5])),
        }
        with pytest.raises(ValueError, match="too short"):
            sampler = RandomFixedWindowSampler(
                sampling_intervals=intervals,
                window_length=2.0,
                drop_short=False,
            )
            len(sampler)

    def test_single_session_single_interval(self):
        intervals = {
            "only": Interval(start=np.array([0.0]), end=np.array([10.0])),
        }
        sampler = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=3.0,
            generator=torch.Generator().manual_seed(42),
        )
        samples = list(sampler)
        assert len(samples) == 3
        assert_all_windows_valid(samples, 3.0, intervals)

    def test_many_small_intervals(self):
        n = 200
        starts = np.arange(0, n * 2, 2, dtype=np.float64)
        ends = starts + 1.5
        intervals = {"s": Interval(start=starts, end=ends)}
        sampler = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=1.0,
            generator=torch.Generator().manual_seed(7),
        )
        samples = list(sampler)
        assert len(samples) == n
        assert_all_windows_valid(samples, 1.0, intervals)

    def test_no_generator_runs_without_error(self):
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.SAMPLING_INTERVALS,
            window_length=1.1,
            generator=None,
        )
        samples = list(sampler)
        assert len(samples) == 9


class TestSequentialFixedWindowSamplerEdgeCases:
    """Edge-case and property tests for SequentialFixedWindowSampler."""

    def test_window_length_equals_interval(self):
        intervals = {
            "s": Interval(start=np.array([0.0]), end=np.array([5.0])),
        }
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=intervals, window_length=5.0
        )
        samples = list(sampler)
        assert len(samples) == 1
        assert np.isclose(samples[0].start, 0.0)
        assert np.isclose(samples[0].end, 5.0)

    def test_step_smaller_than_window_covers_full_interval(self):
        intervals = {
            "s": Interval(start=np.array([0.0]), end=np.array([10.0])),
        }
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=intervals, window_length=4.0, step=2.0
        )
        samples = list(sampler)
        assert samples[0].start == 0.0
        assert samples[-1].end == 10.0

    def test_drop_short_true(self):
        intervals = {
            "s": Interval(
                start=np.array([0.0, 10.0]),
                end=np.array([1.0, 20.0]),
            ),
        }
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=5.0,
            drop_short=True,
        )
        samples = list(sampler)
        for s in samples:
            assert s.start >= 10.0

    def test_drop_short_false_raises(self):
        intervals = {
            "s": Interval(start=np.array([0.0]), end=np.array([1.0])),
        }
        with pytest.raises(ValueError, match="too short"):
            sampler = SequentialFixedWindowSampler(
                sampling_intervals=intervals,
                window_length=5.0,
                drop_short=False,
            )
            list(sampler)

    def test_all_windows_have_correct_length(self):
        intervals = {
            "a": Interval(
                start=np.array([0.0, 100.0]),
                end=np.array([50.0, 200.0]),
            ),
            "b": Interval(start=np.array([0.0]), end=np.array([30.0])),
        }
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=intervals, window_length=7.0, step=3.5
        )
        for s in list(sampler):
            assert np.isclose(s.end - s.start, 7.0)

    def test_many_intervals(self):
        n = 200
        starts = np.arange(0, n * 10, 10, dtype=np.float64)
        ends = starts + 8.0
        intervals = {"s": Interval(start=starts, end=ends)}
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=intervals, window_length=3.0, step=2.0
        )
        samples = list(sampler)
        assert len(samples) > 0
        for s in samples:
            assert np.isclose(s.end - s.start, 3.0)
