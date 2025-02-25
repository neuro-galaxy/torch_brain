import pytest
import numpy as np
import torch
from temporaldata import Interval

from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
    StitcherSamplerWrapper,
)


def test_stitcher_sampler_wrapper():
    # create test interval dict
    sampling_intervals = {
        "session1": Interval(start=np.array([0.0, 20.0]), end=np.array([10.0, 30.0])),
        "session2": Interval(start=np.array([0.0]), end=np.array([15.0])),
    }

    window_length = 5.0
    step = 2.5
    batch_size = 2

    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
        step=step,
    )

    # Test non-distributed sampler
    wrapped_sampler = StitcherSamplerWrapper(sampler)
    samples = list(wrapped_sampler)

    # Get all batches
    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    # Basic checks
    assert len(batches) > 0

    # Check window properties
    for batch in batches:
        for window in batch:
            assert window.end - window.start == window_length

    # Check sequence index
    assert hasattr(wrapped_sampler, "sequence_index")
    assert len(wrapped_sampler.sequence_index) == len(samples)
    assert torch.allclose(
        wrapped_sampler.sequence_index, torch.tensor([0] * 5 + [1] * 3 + [2] * 3)
    )


def test_distributed_stitcher_sampler_wrapper():
    # create test interval dict
    sampling_intervals = {
        "session1": Interval(start=np.array([0.0, 20.0]), end=np.array([10.0, 30.0])),
        "session2": Interval(start=np.array([0.0]), end=np.array([15.0])),
    }

    window_length = 5.0
    step = 2.5
    batch_size = 2
    num_replicas = 2

    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
        step=step,
    )

    # Test rank 0
    sampler0 = StitcherSamplerWrapper(sampler, num_replicas=num_replicas, rank=0)
    samples0 = list(sampler0)

    # Test rank 1
    sampler1 = StitcherSamplerWrapper(sampler, num_replicas=num_replicas, rank=1)
    samples1 = list(sampler1)

    # Get all batches from both samplers
    batches0 = [
        samples0[i : i + batch_size] for i in range(0, len(samples0), batch_size)
    ]
    batches1 = [
        samples1[i : i + batch_size] for i in range(0, len(samples1), batch_size)
    ]

    # Basic checks
    assert len(batches0) > 0
    assert len(batches1) > 0

    # Check window properties
    for batch in batches0:
        for window in batch:
            assert window.end - window.start == window_length

    for batch in batches1:
        for window in batch:
            assert window.end - window.start == window_length

    # Check that windows from same interval stay on same rank
    def get_interval_ids(batches):
        return {window.recording_id for batch in batches for window in batch}

    rank0_intervals = get_interval_ids(batches0)
    rank1_intervals = get_interval_ids(batches1)

    # No overlap between ranks for same interval
    assert len(rank0_intervals.intersection(rank1_intervals)) == 0

    # Check sequence indices are available and make sense
    assert hasattr(sampler0, "sequence_index")
    assert len(sampler0.sequence_index) == len(sampler0.indices)
    assert all(isinstance(idx.item(), int) for idx in sampler0.sequence_index)


def test_stitcher_sampler_wrapper_errors():
    # Test error when wrapping a distributed sampler
    class MockDistributedSampler:
        def __iter__(self):
            return iter([])

        def set_epoch(self, epoch):
            pass

    with pytest.raises(
        ValueError, match="cannot wrap a sampler that is already a distributed sampler"
    ):
        StitcherSamplerWrapper(MockDistributedSampler())

    # Test error when wrapping a random sampler in distributed mode
    sampling_intervals = {
        "session1": Interval(start=np.array([0.0, 20.0]), end=np.array([10.0, 30.0])),
        "session2": Interval(start=np.array([0.0]), end=np.array([15.0])),
    }

    random_sampler = RandomFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=5.0,
    )
    with pytest.raises(ValueError, match="cannot wrap a random sampler"):
        StitcherSamplerWrapper(random_sampler, num_replicas=2, rank=0)

    # Test that random sampler is allowed in non-distributed mode
    try:
        StitcherSamplerWrapper(random_sampler, num_replicas=1, rank=0)
    except ValueError:
        pytest.fail(
            "StitcherSamplerWrapper should allow random sampler in non-distributed mode"
        )
