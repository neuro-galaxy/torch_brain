import numpy as np
from temporaldata import (
    ArrayDict,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    Data,
)


def test_data():
    data = Data(
        session_id="session_0",
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 5]),
            end=np.array([1, 2, 6]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
        domain="auto",
    )

    assert np.allclose(data.domain.start, np.array([0, 5]))
    assert np.allclose(data.domain.end, np.array([3.996, 6]))
