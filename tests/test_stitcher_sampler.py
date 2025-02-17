import pytest
import numpy as np
from temporaldata import Interval

from torch_brain.data.sampler import DistributedStitchingFixedWindowSampler


def test_distributed_stitching_sampler():
    # create test interval dict
    sampling_intervals = {
        "session1": Interval(start=np.array([0.0, 20.0]), end=np.array([10.0, 30.0])),
        "session2": Interval(start=np.array([0.0]), end=np.array([15.0])),
    }

    window_length = 5.0
    step = 2.5
    batch_size = 2
    num_replicas = 2

    # Test rank 0
    sampler0 = DistributedStitchingFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
        step=step,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=0,
    )
    samples0 = list(sampler0)

    # Test rank 1
    sampler1 = DistributedStitchingFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
        step=step,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=1,
    )
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
