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
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import Species

from torch_brain.dataset import Dataset, DatasetIndex, NestedDataset
from torch_brain.dataset.mixins import SpikingDatasetMixin, SEEGDatasetMixin


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


def create_seeg_data(brainset_id, subject_id, session_id, length, sampling_rate=256.0):
    n_channels = 3
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
        seeg_data=RegularTimeSeries(
            sampling_rate=sampling_rate,
            value=np.random.normal(0.0, 1.0, (int(length * sampling_rate), n_channels)),
            domain="auto",
        ),
        channels=ArrayDict(
            id=np.array([f"ch{i}" for i in range(n_channels)]),
            name=np.array([f"C{i}" for i in range(n_channels)]),
            included=np.array([True, False, True]),
            localization_L=np.array([1.0, 2.0, 3.0]),
            localization_I=np.array([4.0, 5.0, 6.0]),
            localization_P=np.array([7.0, 8.0, 9.0]),
        ),
        domain=Interval(0, length),
    )


def build_seeg_channel_view_cache(ds):
    channel_views = {}
    for rid in ds.recording_ids:
        rec = ds.get_recording(rid)
        channels = rec.channels
        ids = np.asarray(channels.id)
        names = np.asarray(getattr(channels, "name", ids))
        included_mask = np.asarray(
            getattr(channels, "included", np.ones(len(ids), dtype=bool)),
            dtype=bool,
        )
        lip = None
        if all(
            hasattr(channels, attr)
            for attr in ("localization_L", "localization_I", "localization_P")
        ):
            lip = np.stack(
                (
                    np.asarray(channels.localization_L, dtype=float),
                    np.asarray(channels.localization_I, dtype=float),
                    np.asarray(channels.localization_P, dtype=float),
                ),
                axis=1,
            )

        channel_views[rid] = SEEGDatasetMixin.ChannelView(
            ids=ids,
            names=names,
            included_mask=included_mask,
            lip=lip,
        )
    return channel_views


def build_seeg_recording_info_cache(ds):
    infos = {}
    for rid in ds.recording_ids:
        rec = ds.get_recording(rid)
        channel_view = ds.seeg_dataset_mixin_channel_views[rid]
        infos[rid] = SEEGDatasetMixin.RecordingInfo(
            recording_id=rid,
            subject_id=rec.subject.id,
            session_id=rec.session.id,
            sampling_rate_hz=ds.get_sampling_rate(rid),
            domain=ds.seeg_dataset_mixin_domain_intervals[rid],
            n_channels=int(len(channel_view.ids)),
            n_included_channels=int(np.sum(channel_view.included_mask)),
        )
    return infos


class _SEEGDatasetWithConstant(SEEGDatasetMixin, Dataset):
    seeg_dataset_mixin_sampling_rate_hz = 256.0


class _SEEGDatasetWithConstantUniquify(_SEEGDatasetWithConstant):
    seeg_dataset_mixin_uniquify_channel_ids = True


def configure_seeg_dataset_caches(
    ds,
    *,
    domain: bool = False,
    channel_views: bool = False,
    recording_infos: bool = False,
):
    if domain:
        ds.seeg_dataset_mixin_domain_intervals = {
            rid: ds.get_recording(rid).seeg_data.domain for rid in ds.recording_ids
        }
    if channel_views:
        ds.seeg_dataset_mixin_channel_views = build_seeg_channel_view_cache(ds)
    if recording_infos:
        if not domain:
            ds.seeg_dataset_mixin_domain_intervals = {
                rid: ds.get_recording(rid).seeg_data.domain for rid in ds.recording_ids
            }
        if not channel_views:
            ds.seeg_dataset_mixin_channel_views = build_seeg_channel_view_cache(ds)
        ds.seeg_dataset_mixin_recording_infos = build_seeg_recording_info_cache(ds)


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


@pytest.fixture
def dummy_seeg_brainset(tmp_path):
    BRAINSET_ID = "mock_seeg_brainset"
    store_path = tmp_path / BRAINSET_ID
    os.makedirs(store_path, exist_ok=True)

    data = create_seeg_data(BRAINSET_ID, "alice", "session1", 1.0)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    data = create_seeg_data(BRAINSET_ID, "bob", "session2", 1.2)
    with h5py.File(store_path / f"{data.session.id}.h5", "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

    return store_path


class TestDataset:
    def test_recording_discovery(self, dummy_spiking_brainset):
        # When recording ids are not specified, root is a string
        ds = Dataset(str(dummy_spiking_brainset))
        expected_rec_ids = ["session1", "session2", "session3", "session4"]
        assert ds.recording_ids == expected_rec_ids

        # When recording ids are not specified
        ds = Dataset(dummy_spiking_brainset)
        expected_rec_ids = ["session1", "session2", "session3", "session4"]
        assert ds.recording_ids == expected_rec_ids

        # When recording ids are specified
        ds = Dataset(dummy_spiking_brainset, recording_ids=["session1", "session3"])
        expected_rec_ids = ["session1", "session3"]
        assert ds.recording_ids == expected_rec_ids

        # Test that recording ids are sorted
        ds = Dataset(dummy_spiking_brainset, recording_ids=["session3", "session1"])
        expected_rec_ids = ["session1", "session3"]
        assert ds.recording_ids == expected_rec_ids

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
        # Test default namespacing
        ds = Dataset(dummy_spiking_brainset)
        sample = ds[DatasetIndex("session1", 0.2, 0.4, _namespace="test_space")]
        assert sample.session.id == "session1"
        assert sample.subject.id == "alice"

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


class TestSEEGDatasetMixin:
    def test_get_sampling_rate(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        assert ds.get_sampling_rate() == 256.0
        assert ds.get_sampling_rate("session2") == 256.0

    def test_get_sampling_rate_requires_dataset_constant(self, dummy_seeg_brainset):
        class SEEGDataset(SEEGDatasetMixin, Dataset): ...

        ds = SEEGDataset(dummy_seeg_brainset)
        with pytest.raises(
            NotImplementedError, match="seeg_dataset_mixin_sampling_rate_hz"
        ):
            ds.get_sampling_rate()

    def test_get_domain_intervals_uses_dataset_cache(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        configure_seeg_dataset_caches(ds, domain=True)

        all_intervals = ds.get_domain_intervals()
        assert set(all_intervals) == set(ds.recording_ids)

        subset = ds.get_domain_intervals(recording_ids=["session2"])
        assert list(subset) == ["session2"]

    def test_get_domain_intervals_requires_dataset_cache(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        with pytest.raises(
            NotImplementedError, match="seeg_dataset_mixin_domain_intervals"
        ):
            ds.get_domain_intervals()

    def test_get_domain_intervals_raises_on_missing_ids(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        ds.seeg_dataset_mixin_domain_intervals = {
            "session1": ds.get_recording("session1").seeg_data.domain
        }
        with pytest.raises(KeyError, match="Missing domain intervals"):
            ds.get_domain_intervals(recording_ids=["session2"])

    def test_get_channel_view(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        configure_seeg_dataset_caches(ds, domain=True, channel_views=True)
        full_view = ds.get_channel_view("session1", included_only=False)
        included_view = ds.get_channel_view("session1", included_only=True)

        assert full_view.ids.tolist() == ["ch0", "ch1", "ch2"]
        assert full_view.names.tolist() == ["C0", "C1", "C2"]
        assert full_view.included_mask.tolist() == [True, False, True]
        assert full_view.lip is not None
        assert full_view.lip.shape == (3, 3)

        assert included_view.ids.tolist() == ["ch0", "ch2"]
        assert included_view.names.tolist() == ["C0", "C2"]
        assert included_view.included_mask.tolist() == [True, True]
        assert included_view.lip is not None
        assert included_view.lip.shape == (2, 3)

    def test_get_channel_view_requires_dataset_cache(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        with pytest.raises(
            NotImplementedError, match="seeg_dataset_mixin_channel_views"
        ):
            ds.get_channel_view("session1")

    def test_get_channel_view_raises_on_missing_recording(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        ds.seeg_dataset_mixin_channel_views = {
            "session1": build_seeg_channel_view_cache(ds)["session1"]
        }
        with pytest.raises(KeyError, match="Missing channel view"):
            ds.get_channel_view("session2")

    def test_get_channel_ids_are_recording_disambiguated(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        configure_seeg_dataset_caches(ds, channel_views=True)

        all_ids = ds.get_channel_ids()
        assert all_ids == [
            "ch0/session1",
            "ch0/session2",
            "ch1/session1",
            "ch1/session2",
            "ch2/session1",
            "ch2/session2",
        ]

        included_ids = ds.get_channel_ids(included_only=True)
        assert included_ids == [
            "ch0/session1",
            "ch0/session2",
            "ch2/session1",
            "ch2/session2",
        ]

    def test_get_channel_ids_match_hook_uniquified_recording_ids(
        self, dummy_seeg_brainset
    ):
        ds = _SEEGDatasetWithConstantUniquify(dummy_seeg_brainset)
        configure_seeg_dataset_caches(ds, channel_views=True)

        recording_ids = ds.get_recording("session1").channels.id.tolist()
        assert recording_ids == [
            "alice/session1/ch0",
            "alice/session1/ch1",
            "alice/session1/ch2",
        ]

        all_ids = ds.get_channel_ids()
        assert all_ids == [
            "alice/session1/ch0",
            "alice/session1/ch1",
            "alice/session1/ch2",
            "bob/session2/ch0",
            "bob/session2/ch1",
            "bob/session2/ch2",
        ]

        included_ids = ds.get_channel_ids(included_only=True)
        assert included_ids == [
            "alice/session1/ch0",
            "alice/session1/ch2",
            "bob/session2/ch0",
            "bob/session2/ch2",
        ]

    def test_get_recording_info(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        configure_seeg_dataset_caches(ds, recording_infos=True)
        info = ds.get_recording_info("session1")

        assert info.recording_id == "session1"
        assert info.subject_id == "alice"
        assert info.session_id == "session1"
        assert info.sampling_rate_hz == 256.0
        assert info.n_channels == 3
        assert info.n_included_channels == 2

    def test_get_recording_info_requires_dataset_cache(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        with pytest.raises(
            NotImplementedError, match="seeg_dataset_mixin_recording_infos"
        ):
            ds.get_recording_info("session1")

    def test_get_recording_info_raises_on_missing_recording(self, dummy_seeg_brainset):
        ds = _SEEGDatasetWithConstant(dummy_seeg_brainset)
        ds.seeg_dataset_mixin_recording_infos = {
            "session1": SEEGDatasetMixin.RecordingInfo(
                recording_id="session1",
                subject_id="alice",
                session_id="session1",
                sampling_rate_hz=256.0,
                domain=Interval(0, 1.0),
                n_channels=3,
                n_included_channels=2,
            )
        }
        with pytest.raises(KeyError, match="Missing recording info"):
            ds.get_recording_info("session2")


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
