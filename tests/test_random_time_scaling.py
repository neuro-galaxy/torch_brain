import numpy as np

from torch_brain.data import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)
from torch_brain.transforms.random_time_scaling import RandomTimeScaling


def test_irregular_scaling():
    timestamps = np.linspace(0, 100, 100)
    unit_index = [0] * 10 + [1] * 20 + [2] * 70
    unit_index = np.array(unit_index)

    # shuffle units
    np.random.shuffle(unit_index)
    types = np.zeros(100)

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["a", "b", "c"]),
        ),
        domain=Interval(start=np.array([0]), end=np.array([100])),
    )

    rts = RandomTimeScaling(min_scale=2.0, max_scale=2.0)
    data_prime = rts(data)

    assert data_prime.start == 0
    assert data_prime.end == 200
    assert data_prime.spikes.timestamps[0] == 0
    assert data_prime.spikes.timestamps[-1] == 200

    rts = RandomTimeScaling(
        min_scale=1.0, max_scale=1.0, min_offset=-1.0, max_offset=-1.0
    )
    data_prime = rts(data)

    assert data_prime.start == -1
    assert data_prime.end == 99
    assert data_prime.spikes.timestamps[0] == -1
    assert data_prime.spikes.timestamps[-1] == 99
    assert data_prime.domain.start == -1
    assert data_prime.domain.end == 99
    assert data_prime.spikes.domain.start == -1
    assert data_prime.spikes.domain.end == 99


def test_regular_scaling():
    lfps = RegularTimeSeries(
        sampling_rate=10.0,
        data=np.random.randn(100),
    )
    data = Data(lfps=lfps, domain=lfps.domain)

    rts = RandomTimeScaling(min_scale=2.0, max_scale=2.0)
    data_prime = rts(data)

    assert data_prime.lfps.domain.start == 0
    assert data_prime.lfps.domain.end == 20.0


def test_scaling_with_large_positive_offset():

    timestamps = np.linspace(0, 10, 100)
    unit_index = np.arange(100)
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain="auto",
    )

    lfps = RegularTimeSeries(
        sampling_rate=10.0,
        data=np.random.randn(100),
    )

    trials = Interval(
        start=np.arange(10.0),
        end=np.arange(10.0) + 1,
    )

    data = Data(
        lfps=lfps,
        spikes=spikes,
        trials=trials,
        domain=Interval(0.0, 10.0),
    )

    rts = RandomTimeScaling(min_scale=2.0, max_scale=2.0, min_offset=15, max_offset=15)
    data_prime = rts(data)

    assert data_prime.domain.start == 15.0  # larger than original end
    assert data_prime.domain.end == 35.0

    assert data_prime.spikes.domain.start == 15.0
    assert data_prime.spikes.domain.end == 35.0

    assert data_prime.lfps.domain.start == 15.0
    assert data_prime.lfps.domain.end == 35.0

    assert data_prime.trials.start.min() == 15.0
    assert data_prime.trials.end.max() == 35.0
