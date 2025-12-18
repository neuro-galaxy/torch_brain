import os
import h5py
import pytest
from datetime import datetime
import numpy as np

from temporaldata import Data, Interval, IrregularTimeSeries, ArrayDict
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Task

from torch_brain.dataset import Dataset


@pytest.fixture
def dummy_spiking_brainset(tmp_path):
    BRAINSET_ID = "mock_brainset"
    store_path = tmp_path / BRAINSET_ID

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id=BRAINSET_ID,
            origin_version="",
            derived_version="1.0.0",
            source="",
            description="",
        ),
        subject=SubjectDescription(
            id="alice",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="session1",
            recording_date=datetime.now(),
            task=Task.REACHING,
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["unit1", "unit2", "unit3"])),
        domain=Interval(0, 1),
    )

    filename = store_path / f"{dummy_data.session.id}.h5"
    os.makedirs(filename.parent, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id=BRAINSET_ID,
            origin_version="dandiset/000005/draft",
            derived_version="1.0.0",
            source="",
            description="",
        ),
        subject=SubjectDescription(
            id="bob",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="session2",
            recording_date=datetime.now(),
            task=Task.REACHING,
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["unit1", "unit2"])),
        domain=Interval(0, 1),
    )

    filename = store_path / f"{dummy_data.session.id}.h5"
    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    # create dummy session files on another dataset
    dummy_data = Data(
        brainset=BrainsetDescription(
            id=BRAINSET_ID,
            origin_version="dandiset/000005/draft",
            derived_version="1.0.0",
            source="",
            description="",
        ),
        subject=SubjectDescription(
            id="bob",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="session3",
            recording_date=datetime.now(),
            task=Task.REACHING,
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["unit1", "unit2", "unit3", "unit4"])),
        domain=Interval(0, 1),
    )

    filename = store_path / f"{dummy_data.session.id}.h5"
    os.makedirs(filename.parent, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    return store_path


def test_recording_discovery(dummy_spiking_brainset):
    ds = Dataset(str(dummy_spiking_brainset))
    expected_rec_ids = ["session1", "session2", "session3"]
    for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
        assert actual_id == expected_id

    ds = Dataset(dummy_spiking_brainset, recording_ids=["session1", "session3"])
    expected_rec_ids = ["session1", "session3"]
    for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
        assert actual_id == expected_id
