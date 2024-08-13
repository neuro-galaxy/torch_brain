from pathlib import Path
import os
import pytest
from dateutil import parser
import numpy as np
import h5py

from temporaldata import Data, IrregularTimeSeries, Interval, RegularTimeSeries
from torch_brain.data import Dataset
from brainsets.taxonomy import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Task, Species, RecordingTech
from brainsets import serialize_fn_map

GABOR_POS_2D_MEAN = 10.0
GABOR_POS_2D_STD = 1.0
RUNNING_SPEED_MEAN = 20.0
RUNNING_SPEED_STD = 2.0


@pytest.fixture
def dummy_data(tmp_path):

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id="allen_neuropixels_mock",
            origin_version="dandiset/000005/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/#/dandiset/000005",
            description="",
        ),
        subject=SubjectDescription(
            id="alice",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="20100102_1",
            recording_date=parser.parse("2010-01-01T00:00:00"),
            task=Task.REACHING,
        ),
        device=DeviceDescription(
            id="20100102",
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            domain="auto",
        ),
        domain=Interval(0, 1),
        running_speed=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            running_speed=np.random.normal(RUNNING_SPEED_MEAN, RUNNING_SPEED_STD, 1000),
            domain="auto",
        ),
        gabors=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            pos_2d=np.random.normal(GABOR_POS_2D_MEAN, GABOR_POS_2D_STD, (1000, 2)),
            domain="auto",
        ),
    )

    filename = tmp_path / dummy_data.brainset.id / f"{dummy_data.session.id}.h5"
    os.makedirs(filename.parent, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id="allen_neuropixels_mock",
            origin_version="dandiset/000005/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/#/dandiset/000005",
            description="",
        ),
        subject=SubjectDescription(
            id="bob",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="20100130_1",
            recording_date=parser.parse("2010-01-01T00:00:00"),
            task=Task.REACHING,
        ),
        device=DeviceDescription(
            id="20100130",
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            domain="auto",
        ),
        domain=Interval(0, 1),
        running_speed=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            running_speed=np.random.normal(RUNNING_SPEED_MEAN, RUNNING_SPEED_STD, 1000),
            domain="auto",
        ),
        gabors=IrregularTimeSeries(
            timestamps=np.arange(0, 1, 0.001),
            pos_2d=np.random.normal(GABOR_POS_2D_MEAN, GABOR_POS_2D_STD, (1000, 2)),
            domain="auto",
        ),
    )

    filename = tmp_path / dummy_data.brainset.id / f"{dummy_data.session.id}.h5"
    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    return tmp_path


def test_dataset_selection(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        include=[{"selection": [{"brainset": "allen_neuropixels_mock"}]}],
    )
    assert len(ds.session_dict) == 2
    assert ds.session_dict["allen_neuropixels_mock/20100102_1"]["filename"] == (
        dummy_data / "allen_neuropixels_mock" / "20100102_1.h5"
    )

    ds = Dataset(
        dummy_data,
        split=None,
        include=[
            {"selection": [{"brainset": "allen_neuropixels_mock", "subject": "alice"}]}
        ],
    )
    assert len(ds.session_dict) == 1

    ds = Dataset(
        dummy_data,
        split=None,
        include=[
            {
                "selection": [
                    {"brainset": "allen_neuropixels_mock", "session": "20100102_1"}
                ]
            }
        ],
    )
    assert len(ds.session_dict) == 1


def test_get_session_data(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        include=[{"selection": [{"brainset": "allen_neuropixels_mock"}]}],
    )

    data = ds.get_session_data("allen_neuropixels_mock/20100102_1")

    assert len(data.spikes) == 1000
    assert len(data.gabors) == 1000
