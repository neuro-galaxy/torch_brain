import pytest
import numpy as np
from temporaldata import IrregularTimeSeries, concat


def test_irregular_timeseries_concat():
    data1 = IrregularTimeSeries(
        unit_index=np.array([0, 0, 1, 0, 1, 2]),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        waveforms=np.zeros((6, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    data2 = IrregularTimeSeries(
        unit_index=np.array([0, 1, 2]),
        timestamps=np.array([0.7, 0.8, 0.9]),
        waveforms=np.zeros((3, 48)),
        timekeys=["timestamps"],
        domain="auto",
    )

    data = concat([data1, data2])

    assert len(data) == len(data1) + len(data2)
    assert np.all(data.unit_index == np.array([0, 0, 1, 0, 1, 2, 0, 1, 2]))
    assert np.all(
        data.timestamps == np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    )
