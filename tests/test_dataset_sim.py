import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from dateutil import parser
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

from torch_brain.data import Dataset

try:
    from brainsets import serialize_fn_map
    from brainsets.descriptions import (
        BrainsetDescription,
        DeviceDescription,
        SessionDescription,
        SubjectDescription,
    )
    from brainsets.taxonomy import RecordingTech, Species, Task

    BRAINSETS_AVAILABLE = True
except ImportError:
    BRAINSETS_AVAILABLE = False

GABOR_POS_2D_MEAN = 10.0
GABOR_POS_2D_STD = 1.0
RUNNING_SPEED_MEAN = 20.0
RUNNING_SPEED_STD = 2.0


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
@pytest.fixture
def dummy_data(tmp_path):

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id="allen_neuropixels_mock_1",
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
        units=ArrayDict(id=np.array(["unit1", "unit2", "unit3"])),
    )

    filename = tmp_path / dummy_data.brainset.id / f"{dummy_data.session.id}.h5"
    os.makedirs(filename.parent, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    # create dummy session files
    dummy_data = Data(
        brainset=BrainsetDescription(
            id="allen_neuropixels_mock_1",
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

        # create dummy session files on another dataset
    dummy_data = Data(
        brainset=BrainsetDescription(
            id="allen_neuropixels_mock_2",
            origin_version="dandiset/000005/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/#/dandiset/000005",
            description="",
        ),
        subject=SubjectDescription(
            id="charlie",
            species=Species.MACACA_MULATTA,
        ),
        session=SessionDescription(
            id="20110130_1",
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
    os.makedirs(filename.parent, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dummy_data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    return tmp_path


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_dataset_selection(dummy_data):
    include_config_1 = [{"selection": [{"brainset": "allen_neuropixels_mock_1"}]}]
    include_config_2 = [
        {"selection": [{"brainset": "allen_neuropixels_mock_1", "subject": "alice"}]}
    ]
    include_config_3 = [
        {
            "selection": [
                {"brainset": "allen_neuropixels_mock_1", "session": "20100102_1"}
            ]
        }
    ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        yaml.dump(
            include_config_1, temp_config_file, encoding="utf-8", allow_unicode=True
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        assert len(ds.recording_dict) == 2
        assert ds.recording_dict["allen_neuropixels_mock_1/20100102_1"]["filename"] == (
            dummy_data / "allen_neuropixels_mock_1" / "20100102_1.h5"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        yaml.dump(
            include_config_2, temp_config_file, encoding="utf-8", allow_unicode=True
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        assert len(ds.recording_dict) == 1

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        yaml.dump(
            include_config_3, temp_config_file, encoding="utf-8", allow_unicode=True
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        assert len(ds.recording_dict) == 1


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_recording_data(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )

    data = ds.get_recording_data("allen_neuropixels_mock_1/20100102_1")

    assert len(data.spikes) == 1000
    assert len(data.gabors) == 1000


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_subject_ids(dummy_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        yaml.dump(
            [{"selection": [{"brainset": "allen_neuropixels_mock_1"}]}],
            temp_config_file,
            encoding="utf-8",
            allow_unicode=True,
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        subject_ids = ds.get_subject_ids()
        assert subject_ids == [
            "allen_neuropixels_mock_1/alice",
            "allen_neuropixels_mock_1/bob",
        ]


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_sampling_intervals(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )

    intervals = ds.get_sampling_intervals()
    assert len(intervals) == 1
    assert "allen_neuropixels_mock_1/20100102_1" in intervals


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_unit_ids(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )
    unit_ids = ds.get_unit_ids()
    assert len(unit_ids) == 3
    assert "allen_neuropixels_mock_1/20100102_1/unit1" in unit_ids
    assert "allen_neuropixels_mock_1/20100102_1/unit2" in unit_ids


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_recording_config_dict(dummy_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        config = [
            {
                "selection": [{"brainset": "allen_neuropixels_mock_1"}],
                "config": {"test_param": 123},
            }
        ]
        yaml.dump(
            config,
            temp_config_file,
            encoding="utf-8",
            allow_unicode=True,
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        config_dict = ds.get_recording_config_dict()
        assert "allen_neuropixels_mock_1/20100102_1" in config_dict
        assert config_dict["allen_neuropixels_mock_1/20100102_1"]["test_param"] == 123


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_session_ids(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )
    session_ids = ds.get_session_ids()
    assert len(session_ids) == 1
    assert "allen_neuropixels_mock_1/20100102_1" in session_ids


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_recording_data(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )
    data = ds.get_recording_data("allen_neuropixels_mock_1/20100102_1")

    # Check basic properties
    assert data.brainset.id == "allen_neuropixels_mock_1"
    assert data.session.id == "allen_neuropixels_mock_1/20100102_1"
    assert data.subject.id == "allen_neuropixels_mock_1/alice"

    # Check data fields
    assert hasattr(data, "running_speed")
    assert hasattr(data, "gabors")
    assert hasattr(data, "spikes")


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_get_slice(dummy_data):
    ds = Dataset(
        dummy_data,
        split=None,
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )

    # Get a slice from 0.2 to 0.4 seconds
    data = ds.get("allen_neuropixels_mock_1/20100102_1", 0.2, 0.4)

    # Check basic properties
    assert data.brainset.id == "allen_neuropixels_mock_1"
    assert data.session.id == "allen_neuropixels_mock_1/20100102_1"
    assert data.subject.id == "allen_neuropixels_mock_1/alice"


@pytest.mark.skipif(not BRAINSETS_AVAILABLE, reason="brainsets not installed")
def test_disable_data_leakage_check(dummy_data):
    ds = Dataset(
        dummy_data,
        split="train",
        recording_id="allen_neuropixels_mock_1/20100102_1",
    )

    assert ds._check_for_data_leakage_flag == True
    ds.disable_data_leakage_check()
    assert ds._check_for_data_leakage_flag == False


def test_get_brainset_ids(dummy_data):
    include_config = [
        {
            "selection": [
                {"brainset": "allen_neuropixels_mock_1"},
                {"brainset": "allen_neuropixels_mock_2"},
            ]
        }
    ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config_file:
        yaml.dump(
            include_config, temp_config_file, encoding="utf-8", allow_unicode=True
        )
        temp_config_file.flush()
        ds = Dataset(
            str(dummy_data),
            split=None,
            config=temp_config_file.name,
        )
        brainset_ids = ds.get_brainset_ids()
        assert len(brainset_ids) == 2
        assert "allen_neuropixels_mock_1" in brainset_ids
        assert "allen_neuropixels_mock_2" in brainset_ids
