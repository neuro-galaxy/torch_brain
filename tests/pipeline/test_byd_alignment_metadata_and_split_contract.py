import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path

from torch_brain.data import LazyInterval

add_pipelines_to_path()
from keles_byd_2024 import pipeline as m  # noqa: E402


class _FakeNWBIO:
    def __init__(self, *args, **kwargs):
        self._nwb = SimpleNamespace(
            trials=SimpleNamespace(to_dataframe=lambda: pd.DataFrame({"x": [1]}))
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._nwb


class _FakeRegularTimeSeries:
    def __init__(self, *, data, sampling_rate, domain_start, domain):
        self.data = np.asarray(data, dtype=np.float32)
        self.sampling_rate = float(sampling_rate)
        self.domain_start = float(domain_start)
        self.timestamps = (
            self.domain_start + np.arange(self.data.shape[0]) / self.sampling_rate
        )
        self.domain = SimpleNamespace(
            start=np.array([domain_start]),
            end=np.array([domain_start + len(data) / sampling_rate]),
        )


class _FakeData:
    last_instance = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "brainset" in kwargs:
            _FakeData.last_instance = self

    def to_hdf5(self, file, serialize_fn_map=None):
        file["written"] = True
        file["seeg_data"] = SimpleNamespace(attrs={})


class _FakeStore(dict):
    def __init__(self):
        super().__init__()
        self.attrs = {}


class _FakeH5File:
    last_store = None

    def __init__(self, *args, **kwargs):
        self._store = _FakeStore()
        Path(args[0]).touch()

    def __enter__(self):
        _FakeH5File.last_store = self._store
        return self._store

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeInterval:
    def __init__(self, n):
        self.start = np.arange(n, dtype=np.float64)
        self.end = self.start + 1.0
        self.label = np.ones(n, dtype=np.int64)


def _install_process_patches(monkeypatch, tmp_path):
    for name in ("speech", "global_flow", "word_gap"):
        (tmp_path / f"{name}_binary_labels.csv").write_text(
            "start_time_s,label\n0.5,0\n1.5,1\n"
        )

    def build_channels(electrode_df, *, subject_number):
        assert subject_number == 41
        return SimpleNamespace(included=np.array([True, True]))

    monkeypatch.setattr(m, "NWBHDF5IO", _FakeNWBIO)
    monkeypatch.setattr(m, "RegularTimeSeries", _FakeRegularTimeSeries)
    monkeypatch.setattr(m, "Data", _FakeData)
    monkeypatch.setattr(m.h5py, "File", _FakeH5File)

    monkeypatch.setattr(
        m,
        "_extract_lfp",
        lambda nwb: (
            np.zeros((100, 2), dtype=np.float32),
            np.linspace(0.0, 99.0, 100, dtype=np.float64),
            pd.DataFrame({"origchannel_name": ["LAMY1", "RAMY1"]}),
            1.0,
        ),
    )
    monkeypatch.setattr(m, "_extract_encoding_interval", lambda trials_df: (0.0, 10.0))
    monkeypatch.setattr(
        m,
        "_apply_task_window",
        lambda lfp_data, lfp_time, trials_df, edge: (
            np.arange(60, dtype=np.float32).reshape(30, 2),
            np.linspace(-10.0, 20.0, 30, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(m, "_build_channels", build_channels)
    monkeypatch.setattr(
        m,
        "_compute_included_mask",
        lambda electrode_df: np.array([True, True]),
    )
    monkeypatch.setattr(
        m, "_infer_subject_metadata", lambda nwb: SimpleNamespace(id="sub-CS41")
    )
    monkeypatch.setattr(
        m,
        "_infer_session_metadata",
        lambda nwb, session_id: SimpleNamespace(id=session_id),
    )
    monkeypatch.setattr(
        m, "_infer_device_metadata", lambda nwb: SimpleNamespace(id="device")
    )


def _install_split_patches(monkeypatch):
    monkeypatch.setattr(
        m,
        "_load_label_csv",
        lambda csv_path: (
            np.array([0.2, 1.0, 2.0], dtype=np.float64),
            np.array([1, 0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        m,
        "_build_windows",
        lambda times, labels, **kwargs: (
            np.array([-0.8, 0.0], dtype=np.float64),
            np.array([0.2, 1.0], dtype=np.float64),
            np.array([1, 0], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        m,
        "generate_temporal_folds",
        lambda start, end, labels, **kwargs: {
            0: {
                "train": _FakeInterval(2),
                "val": _FakeInterval(1),
                "test": _FakeInterval(1),
            }
        },
    )


def test_process_file_persists_alignment_metadata_contract(monkeypatch, tmp_path):
    _install_process_patches(monkeypatch, tmp_path)

    out = m.process_file(
        input_file=str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
        output_dir=str(tmp_path),
        labels_dir=str(tmp_path),
        label_files=[],
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=True,
        balance_splits=False,
        balance_seed=0,
    )

    assert out.endswith("sub-CS41_ses-P41CSR1.h5")
    data = _FakeData.last_instance
    assert data is not None

    assert data.brainset.derived_version == "1.2.0"
    assert data.session.id == "sub-CS41_ses-P41CSR1"
    assert data.alignment_version == "2.0.0"
    assert data.alignment_reference == m.ALIGNMENT_REFERENCE
    assert data.alignment_method == m.ALIGNMENT_METHOD
    assert data.alignment_applied_at_prepare is True
    assert data.seeg_sampling_rate_hz == 1.0
    np.testing.assert_allclose(
        data.seeg_data.data,
        np.arange(60, dtype=np.float32).reshape(30, 2),
    )

    params = json.loads(data.alignment_parameters_json)
    assert params["lfp_movie_pad_s"] == float(m.LFP_MOVIE_PAD_S)
    assert params["edge_offset_s"] == float(m.EDGE_OFFSET_S)
    assert params["encoding_start_s"] == 0.0
    assert params["encoding_stop_s"] == 10.0
    assert _FakeH5File.last_store["seeg_data"].attrs == {
        "unit": "V",
        "scale_to_uV": 1e6,
    }


def test_build_channels_copies_byd_coordinate_fields():
    channels = m._build_channels(
        pd.DataFrame(
            {
                "origchannel_name": ["LAMY1", "RAMY1"],
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "z": [5.0, 6.0],
            }
        ),
        subject_number=None,
    )

    np.testing.assert_allclose(channels.coord_byd_mni152_r, [1.0, 2.0])
    np.testing.assert_allclose(channels.coord_byd_mni152_a, [3.0, 4.0])
    np.testing.assert_allclose(channels.coord_byd_mni152_s, [5.0, 6.0])


def test_process_file_writes_neuroprobe_style_split_and_channel_keys(
    monkeypatch, tmp_path
):
    _install_process_patches(monkeypatch, tmp_path)
    _install_split_patches(monkeypatch)

    m.process_file(
        input_file=str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
        output_dir=str(tmp_path),
        labels_dir=str(tmp_path),
        label_files=["speech_binary_labels.csv"],
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=False,
        balance_splits=False,
        balance_seed=0,
    )

    data = _FakeData.last_instance
    assert data is not None
    split_key = "full$binary$within_session$speech$fold0$train"
    assert hasattr(data.splits, split_key)
    assert hasattr(data.channel_splits, split_key)


def test_split_twofold_partition_invariants():
    n = 12
    start = np.arange(n, dtype=np.float64)
    end = start + 1.0
    labels = np.array([0, 1] * 6, dtype=np.int64)

    folds = m.generate_temporal_folds(start, end, labels, domain=[0, 12], balance=False)
    assert set(folds.keys()) == {0, 1}

    for fold_idx, split in folds.items():
        train = split["train"]
        val = split["val"]
        test = split["test"]

        train_set = set(train.start.tolist())
        val_set = set(val.start.tolist())
        test_set = set(test.start.tolist())

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert len(train_set | val_set | test_set) == n

        if fold_idx == 0:
            assert len(train.start) == 6
            assert len(val.start) == 3
            assert len(test.start) == 3
        else:
            assert len(train.start) == 6
            assert len(val.start) == 3
            assert len(test.start) == 3


def _interval_signature(interval):
    return list(
        zip(
            interval.start.tolist(),
            interval.end.tolist(),
            interval.label.tolist(),
            strict=True,
        )
    )


def _word_gap_split_signatures(data):
    signatures = {}
    for fold_idx in (0, 1):
        for split_name in ("train", "val", "test"):
            split_key = m.split_selector_key(
                label_mode="binary",
                task_name="word_gap",
                fold_idx=fold_idx,
                split_name=split_name,
            )
            signatures[(fold_idx, split_name)] = _interval_signature(
                getattr(data.splits, split_key)
            )
    return signatures


def _process_byd_with_other_labels(monkeypatch, tmp_path, other_labels):
    _install_process_patches(monkeypatch, tmp_path)

    times = np.arange(24, dtype=np.float64) + 0.5
    target_labels = np.array([0, 0, 0, 1] * 6, dtype=np.int64)
    other_labels = np.asarray(other_labels, dtype=np.int64)

    def _load_label_csv(csv_path):
        if str(csv_path).endswith("global_flow_binary_labels.csv"):
            return times, other_labels
        return times, target_labels

    monkeypatch.setattr(m, "_load_label_csv", _load_label_csv)
    monkeypatch.setattr(
        m,
        "_build_windows",
        lambda times, labels, **kwargs: (
            times.astype(np.float64),
            times.astype(np.float64) + 1.0,
            labels,
        ),
    )

    m.process_file(
        input_file=str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
        output_dir=str(tmp_path),
        labels_dir=str(tmp_path),
        label_files=[
            "global_flow_binary_labels.csv",
            "word_gap_binary_labels.csv",
        ],
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=False,
        balance_splits=True,
        balance_seed=7,
    )

    data = _FakeData.last_instance
    assert data is not None
    return _word_gap_split_signatures(data)


def test_balanced_byd_target_task_splits_ignore_other_label_changes(
    monkeypatch, tmp_path
):
    first = _process_byd_with_other_labels(
        monkeypatch,
        tmp_path,
        other_labels=np.array([0, 1] * 12, dtype=np.int64),
    )
    second = _process_byd_with_other_labels(
        monkeypatch,
        tmp_path,
        other_labels=np.array([0, 0, 0, 0, 0, 1] * 4, dtype=np.int64),
    )

    assert first == second


def _extract_audit_events(caplog):
    prefix = "BYD_PREP_AUDIT "
    events = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith(prefix):
            events.append(json.loads(msg[len(prefix) :]))
    return events


def test_process_file_emits_success_audit_log(monkeypatch, caplog, tmp_path):
    _install_process_patches(monkeypatch, tmp_path)
    _install_split_patches(monkeypatch)

    caplog.set_level("INFO")
    out = m.process_file(
        input_file=str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
        output_dir=str(tmp_path),
        labels_dir=str(tmp_path),
        label_files=["speech_binary_labels.csv"],
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=False,
        balance_splits=False,
        balance_seed=0,
    )

    assert out.endswith(".h5")
    events = _extract_audit_events(caplog)
    assert len(events) >= 1
    evt = events[-1]
    assert evt["status"] == "success"
    assert evt["alignment_version"] == "2.0.0"
    assert evt["alignment_method"] == m.ALIGNMENT_METHOD
    assert evt["lfp_movie_pad_s"] == pytest.approx(float(m.LFP_MOVIE_PAD_S))
    assert evt["edge_offset_s"] == pytest.approx(float(m.EDGE_OFFSET_S))
    assert evt["encoding_start_s"] == pytest.approx(0.0)
    assert evt["encoding_stop_s"] == pytest.approx(10.0)
    assert evt["recording_id"] == "sub-CS41_ses-P41CSR1"
    assert evt["label_window_counts"]["speech_binary_labels.csv"]["raw_rows"] == 3
    assert evt["label_window_counts"]["speech_binary_labels.csv"]["kept_windows"] == 2
    assert (
        evt["label_window_counts"]["speech_binary_labels.csv"]["dropped_windows"] == 1
    )


def test_process_file_emits_failure_audit_log(monkeypatch, caplog, tmp_path):
    _install_process_patches(monkeypatch, tmp_path)
    monkeypatch.setattr(
        m,
        "_extract_encoding_interval",
        lambda trials_df: (_ for _ in ()).throw(ValueError("no encoding phase")),
    )

    caplog.set_level("INFO")
    with pytest.raises(ValueError, match="no encoding phase"):
        m.process_file(
            input_file=str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
            output_dir=str(tmp_path),
            labels_dir=str(tmp_path),
            label_files=["speech_binary_labels.csv"],
            pre_offset_s=0.0,
            post_offset_s=1.0,
            no_splits=False,
            balance_splits=False,
            balance_seed=0,
        )

    events = _extract_audit_events(caplog)
    assert len(events) >= 1
    evt = events[-1]
    assert evt["status"] == "failure"
    assert evt["recording_id"] == "sub-CS41_ses-P41CSR1"
    assert evt["failure_type"] == "ValueError"
    assert "no encoding phase" in evt["failure_reason"]
    assert evt["alignment_version"] == "2.0.0"


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_hdf5_write_preserves_previous_output(
    tmp_path, monkeypatch, existing
):
    real_h5_file = h5py.File
    output = tmp_path / "sub-CS41_ses-P41CSR1.h5"
    if existing:
        with real_h5_file(output, "w") as handle:
            handle["previous"] = np.arange(5, dtype=np.int16)
        previous_bytes = output.read_bytes()
    _install_process_patches(monkeypatch, tmp_path)
    monkeypatch.setattr(m.h5py, "File", real_h5_file)

    def interrupted_save(self, handle):
        handle["partial"] = np.arange(3)
        handle.flush()
        if existing:
            assert output.read_bytes() == previous_bytes
        else:
            assert not output.exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(_FakeData, "to_hdf5", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        m.process_file(
            str(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
            str(tmp_path),
            labels_dir=str(tmp_path),
            label_files=[],
            pre_offset_s=0.0,
            post_offset_s=1.0,
            no_splits=True,
        )
    assert output.exists() is existing
    if existing:
        assert output.read_bytes() == previous_bytes
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("corruption", ["empty", "truncated", "incomplete_hdf5"])
def test_pipeline_reprocesses_malformed_hdf5(tmp_path, monkeypatch, corruption):
    output = tmp_path / "sub-CS41_ses-P41CSR1.h5"
    if corruption == "incomplete_hdf5":
        with h5py.File(output, "w") as handle:
            handle.create_group("seeg_data")
    else:
        output.write_bytes(b"" if corruption == "empty" else b"partial")
    calls = []
    real_h5_file = h5py.File
    _install_process_patches(monkeypatch, tmp_path)
    monkeypatch.setattr(m.h5py, "File", real_h5_file)

    def serialize_fixture(self, handle):
        calls.append(True)
        signal = handle.create_group("seeg_data")
        signal["data"] = self.seeg_data.data
        handle.create_group("brainset").attrs["derived_version"] = (
            self.brainset.derived_version
        )
        handle.attrs["preparation_settings_json"] = self.preparation_settings_json
        for key in ("channels", "subject", "session", "domain"):
            handle.create_group(key)
        handle["channels/id"] = np.array(["a", "b"], dtype="S1")

    monkeypatch.setattr(_FakeData, "to_hdf5", serialize_fixture)
    pipeline = m.Pipeline(tmp_path, tmp_path, args=m.parser.parse_args(["--no-splits"]))
    raw = tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"
    pipeline.process(raw)
    assert len(calls) == 1
    assert m._is_valid_processed_file(output)
    pipeline.process(raw)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "labels",
    [[0, 1, 0, 1, 0, 0, 1, 0], [0, 1, 2, 0, 1, 2, 0, 0], [0] * 8, []],
    ids=["binary", "multiclass", "single-class", "empty"],
)
def test_balanced_annotations_are_written_in_time_order(labels, tmp_path):
    labels = np.asarray(labels, dtype=np.int64)
    starts = np.arange(0, 20, 0.25) if len(labels) else np.array([])
    labels = np.resize(labels, len(starts))
    folds = m.generate_temporal_folds(
        starts,
        starts + 1,
        labels,
        domain=[0, 20],
        balance=True,
        rng=np.random.default_rng(42),
    )
    for fold in folds.values():
        for interval in fold.values():
            assert np.all(np.diff(interval.start) >= 0)
            np.testing.assert_array_equal(interval.end, interval.start + 1)
            np.testing.assert_array_equal(
                interval.label, labels[(interval.start * 4).astype(int)]
            )
    result = folds[0]["train"]

    # Exercise the serialized annotations through the real lazy slicing path.
    path = tmp_path / "annotations.h5"
    with h5py.File(path, "w") as handle:
        result.to_hdf5(handle)
    with h5py.File(path, "r") as handle:
        sliced = LazyInterval.from_hdf5(handle).slice(0.75, 1.25, reset_origin=False)
        overlap = (result.end > 0.75) & (result.start < 1.25)
        np.testing.assert_array_equal(sliced.start, result.start[overlap])
        np.testing.assert_array_equal(sliced.label, result.label[overlap])
