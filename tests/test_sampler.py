import pytest

import numpy as np
import torch

from temporaldata import Interval

from torch_brain.data.sampler import (
    SequentialFixedWindowSampler,
    RandomFixedWindowSampler,
    TrialSampler,
)
from torch_brain.data.dataset import DatasetIndex


# helper
def compare_slice_indices(a, b):
    return (
        (a.session_id == b.session_id)
        and np.isclose(a.start, b.start)
        and np.isclose(a.end, b.end)
    )


# helper
def samples_in_interval_dict(samples, interval_dict):
    for s in samples:
        assert s.session_id in interval_dict
        allowed_intervals = interval_dict[s.session_id]
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


def test_sequential_sampler():
    sampler = SequentialFixedWindowSampler(
        interval_dict={
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

    interval_dict = {
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
        interval_dict=interval_dict,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    assert len(sampler) == 9

    # sample and check that all indices are within the expected range
    samples = list(sampler)
    assert len(samples) == 9
    assert samples_in_interval_dict(samples, interval_dict) == True

    # sample again and check that the indices are different this time
    samples2 = list(sampler)
    assert len(samples) == 9
    for s1 in samples:
        for s2 in samples2:
            assert not compare_slice_indices(s1, s2)

    # Test "index in valid range" when step > window_length
    sampler = RandomFixedWindowSampler(
        interval_dict=interval_dict,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    samples = list(sampler)
    assert samples_in_interval_dict(samples, interval_dict) == True

    # Having window_length bigger than any interval should raise an error
    with pytest.raises(ValueError):
        sampler = RandomFixedWindowSampler(
            interval_dict=interval_dict,
            window_length=5,
            generator=torch.Generator().manual_seed(42),
        )

        len(sampler)


def test_trial_sampler():
    interval_dict = {
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
        interval_dict=interval_dict,
    )
    assert len(sampler) == 7

    # Check that the sampled interval is within the expected range
    samples = list(sampler)
    assert len(samples) == 7
    assert samples_in_interval_dict(samples, interval_dict)

    # With the same seed, the sampler should always give the same outputs.
    sampler1 = TrialSampler(
        interval_dict=interval_dict,
        generator=torch.Generator().manual_seed(42),
    )
    sampler2 = TrialSampler(
        interval_dict=interval_dict,
        generator=torch.Generator().manual_seed(42),
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
    sampler1 = TrialSampler(interval_dict=interval_dict, shuffle=False)
    samples1 = list(sampler1)
    assert compare_slice_indices(samples1[0], DatasetIndex("session1", 0.0, 2.0))
