import numpy as np

from temporaldata import Data, ArrayDict, IrregularTimeSeries
from torch_brain.transforms.unit_dropout import TriangleDistribution, UnitDropout


def test_distro():
    for i in range(100):
        num_units = TriangleDistribution(
            min_units=100, mode_units=150, max_units=200
        ).sample(196)
        assert num_units >= 100 and num_units <= 200


def test_spikes():
    timestamps = np.zeros(100)
    unit_index = [0] * 10 + [1] * 20 + [2] * 70
    unit_index = np.array(unit_index)
    # shuffle units
    np.random.shuffle(unit_index)
    types = np.zeros(100)

    for i in range(100):
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
            domain="auto",
        )
        transform = UnitDropout(min_units=1, mode_units=2, max_units=2)
        data_t = transform(data)
        assert data_t.spikes.timestamps.shape[0] in (10, 20, 30, 70, 80, 90, 100)
        assert 1 <= len(data_t.units.id) <= 2
        assert len(data_t.spikes.timestamps) == len(data_t.spikes.unit_index)
        assert np.unique(data_t.spikes.unit_index).shape[0] == len(data_t.units.id)

        original_unit_ids = data.units.id[data.spikes.unit_index]
        original_timestamps = data.spikes.timestamps

        transformed_unit_ids = data_t.units.id[data_t.spikes.unit_index]
        transformed_timestamps = data_t.spikes.timestamps

        for unit_id in data.units.id:
            if unit_id in transformed_unit_ids:
                assert np.allclose(
                    original_timestamps[original_unit_ids == unit_id],
                    transformed_timestamps[transformed_unit_ids == unit_id],
                )
