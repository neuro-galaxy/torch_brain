import os
import h5py
import pytest
from datetime import datetime
import numpy as np

from temporaldata import (
    Data,
    Interval,
    IrregularTimeSeries,
    ArrayDict,
    RegularTimeSeries,
)
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Task

from torch_brain.dataset import Dataset, DatasetIndex, NestedDataset
from torch_brain.dataset.mixins import SpikingDatasetMixin


def create_spiking_data(brainset_id, subject_id, session_id, length):
    num_spikes = np.random.randint(5, int(length * 20))
    num_units = np.random.randint(1, num_spikes)
    return Data(
        brainset=BrainsetDescription(
            id=brainset_id,
            origin_version="",
            derived_version="",
            source="",
            description="",
        ),
        subject=SubjectDescription(subject_id, Species.UNKNOWN),
        session=SessionDescription(session_id, datetime.now()),
        spikes=IrregularTimeSeries(
            timestamps=np.sort(np.random.uniform(0, length, num_spikes)),
            unit_index=np.random.randint(0, num_units, num_spikes),
            domain="auto",
        ),
        units=ArrayDict(id=np.array([f"unit_{i}" for i in range(num_units)])),
        lfp=RegularTimeSeries(
            sampling_rate=10.0,
            value=np.random.normal(0.0, 1.0, (int(length * 10.0), 30)),
            domain="auto",
        ),
        domain=Interval(0, length),
    )


@pytest.fixture
def dummy_spiking_brainset(tmp_path):
    BRAINSET_ID = "mock_brainset"
    store_path = tmp_path / BRAINSET_ID
    os.makedirs(store_path, exist_ok=True)

    data = create_spiking_data(BRAINSET_ID, "alice", "session1", 1.0)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    data = create_spiking_data(BRAINSET_ID, "bob", "session2", 1.5)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    data = create_spiking_data(BRAINSET_ID, "bob", "session3", 0.9)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    data = create_spiking_data(BRAINSET_ID, "charlie", "session4", 2.0)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    return store_path


class TestDataset:
    def test_recording_discovery(self, dummy_spiking_brainset):
        # When recording ids are not specified, root is a string
        ds = Dataset(str(dummy_spiking_brainset))
        expected_rec_ids = ["session1", "session2", "session3", "session4"]
        for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
            assert actual_id == expected_id

        # When recording ids are not specified
        ds = Dataset(dummy_spiking_brainset)
        expected_rec_ids = ["session1", "session2", "session3", "session4"]
        for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
            assert actual_id == expected_id

        # When recording ids are specified
        ds = Dataset(dummy_spiking_brainset, recording_ids=["session1", "session3"])
        expected_rec_ids = ["session1", "session3"]
        for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
            assert actual_id == expected_id

        # Test that recording ids are sorted
        ds = Dataset(dummy_spiking_brainset, recording_ids=["session3", "session1"])
        expected_rec_ids = ["session1", "session3"]
        for actual_id, expected_id in zip(ds.recording_ids, expected_rec_ids):
            assert actual_id == expected_id

    def test_incorrect_paths(self, dummy_spiking_brainset):
        with pytest.raises(ValueError, match="No recordings found at"):
            Dataset("idonotexist")

        with pytest.raises(FileNotFoundError):
            Dataset(dummy_spiking_brainset, recording_ids=["session1", "nonexistent"])

    def test_get_recording(self, dummy_spiking_brainset):
        ds = Dataset(dummy_spiking_brainset)
        rec = ds.get_recording("session1")
        assert isinstance(rec, Data)
        assert rec.session.id == "session1"
        assert rec.subject.id == "alice"

        ds = Dataset(dummy_spiking_brainset, keep_files_open=False)
        rec = ds.get_recording("session2")
        assert isinstance(rec, Data)
        assert rec.session.id == "session2"
        assert rec.subject.id == "bob"

    def test_getitem(self, dummy_spiking_brainset):
        ds = Dataset(dummy_spiking_brainset)
        sample = ds[DatasetIndex("session1", 0.2, 0.4)]
        assert (sample.domain.end[-1] - sample.domain.start[0]) == 0.2

    def test_getitem_with_transform(self, dummy_spiking_brainset):
        def tf(data):
            data.subject.id = data.subject.id + "_transformed"
            return data

        # With transform as argument during init
        ds = Dataset(dummy_spiking_brainset, transform=tf)
        sample = ds[DatasetIndex("session1", 0.2, 0.4)]
        assert sample.subject.id == "alice_transformed"

        # With transform set later
        ds = Dataset(dummy_spiking_brainset)
        ds.transform = tf
        sample = ds[DatasetIndex("session1", 0.2, 0.4)]
        assert sample.subject.id == "alice_transformed"

    def test_get_sampling_intervals(self, dummy_spiking_brainset):
        ds = Dataset(dummy_spiking_brainset)
        samp_intervals = ds.get_sampling_intervals()
        expected = {
            "session1": Interval(0, 1.0),
            "session2": Interval(0, 1.5),
            "session3": Interval(0, 0.9),
            "session4": Interval(0, 2.0),
        }
        assert len(samp_intervals) == len(expected)
        for rid, expect in expected.items():
            actual = samp_intervals[rid]
            assert len(actual) == len(expect)
            assert (actual.start == expect.start).all()
            assert (actual.end == expect.end).all()

    def test_default_apply_namespace(self, dummy_spiking_brainset):
        # Test default
        ds = Dataset(dummy_spiking_brainset)
        sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]
        assert sample.session.id == "test_space/session1"
        assert sample.subject.id == "test_space/alice"

        # Test selectivity of namespace attributes
        ds = Dataset(dummy_spiking_brainset, namespace_attributes=["session.id"])
        sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]
        assert sample.session.id == "test_space/session1"
        assert sample.subject.id == "alice"

        # Test namespacing of unit ids
        ds = Dataset(
            dummy_spiking_brainset,
            namespace_attributes=["session.id", "subject.id", "units.id"],
        )
        sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]
        assert sample.session.id == "test_space/session1"
        assert sample.subject.id == "test_space/alice"
        for unit_id in sample.units.id:
            assert str(unit_id).startswith("test_space/")

        # Test error at incorrect attribute
        ds = Dataset(
            dummy_spiking_brainset,
            namespace_attributes=["session.id", "subject.id", "lfp.value"],
        )
        with pytest.raises(TypeError):
            sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]

        # Test error at non-existent attribute
        ds = Dataset(
            dummy_spiking_brainset,
            namespace_attributes=["session.id", "subject.id", "does.not.exist"],
        )
        with pytest.raises(AttributeError):
            sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]

    def test_repr(self, dummy_spiking_brainset):
        ds = Dataset(dummy_spiking_brainset)
        assert str(ds) == "Dataset(n_recordings=4)"

        class ChildDataset(Dataset): ...

        ds = ChildDataset(dummy_spiking_brainset)
        assert str(ds) == "ChildDataset(n_recordings=4)"

        class NoTransform:
            def __call__(self, data):
                return data

            def __repr__(self):
                return "<NoTransform>"

        ds = Dataset(dummy_spiking_brainset, transform=NoTransform())
        assert str(ds) == "Dataset(n_recordings=4, transform=<NoTransform>)"

    def test_illegal_recording_id(self, dummy_spiking_brainset):
        ds = Dataset(dummy_spiking_brainset)
        with pytest.raises(KeyError):
            ds.get_recording("illegal_recording_id")


class TestNestedDataset:
    def test_init(self, dummy_spiking_brainset):
        # List based datasets that conflict in names
        ds1 = Dataset(dummy_spiking_brainset, recording_ids=["session1", "session2"])
        ds2 = Dataset(dummy_spiking_brainset, recording_ids=["session4", "session3"])
        with pytest.raises(ValueError, match="^Duplicate dataset class names found"):
            nested = NestedDataset([ds1, ds2])

        # List based datasets that don't conflict in names
        class ChildDataset(Dataset): ...

        ds2r = ChildDataset(
            dummy_spiking_brainset, recording_ids=["session4", "session3"]
        )
        nested = NestedDataset([ds1, ds2r])
        expected_rids = [
            "ChildDataset/session3",
            "ChildDataset/session4",
            "Dataset/session1",
            "Dataset/session2",
        ]
        assert nested.recording_ids == expected_rids

        # dict based datasets
        nested = NestedDataset({"ds1": ds1, "ds2": ds2})
        expected_rids = ["ds1/session1", "ds1/session2", "ds2/session3", "ds2/session4"]
        assert nested.recording_ids == expected_rids

        # test if input type other than expected:
        with pytest.raises(TypeError):
            NestedDataset(1)


class TestSpikingDatasetMixin:
    def test_get_unit_ids(self, dummy_spiking_brainset):
        class SpikingDataset(SpikingDatasetMixin, Dataset): ...

        ds = SpikingDataset(dummy_spiking_brainset)
        expected_unit_ids = sorted(
            sum(
                [ds.get_recording(rid).units.id.tolist() for rid in ds.recording_ids],
                [],
            )
        )
        actual_unit_ids = ds.get_unit_ids()
        assert actual_unit_ids == expected_unit_ids


def test_ensure_index_has_namespace():
    from torch_brain.dataset.dataset import _ensure_index_has_namespace

    # Ensure it doesn't corrupt existing namespace
    idx = DatasetIndex("test", 0.0, 1.1, "real_namespace")
    idx = _ensure_index_has_namespace(idx)
    assert idx._namespace == "real_namespace"

    # Ensure it adds a "" if no namespace present
    delattr(idx, "_namespace")
    idx = _ensure_index_has_namespace(idx)
    assert idx._namespace == ""
