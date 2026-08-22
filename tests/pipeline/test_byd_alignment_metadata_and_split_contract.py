import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from _utils import add_pipelines_to_path

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
        self.domain = domain


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


class _FakeH5File:
    last_store = None

    def __init__(self, *args, **kwargs):
        self._store = {}

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


def _install_process_patches(monkeypatch):
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


def test_process_file_persists_alignment_metadata_contract(monkeypatch, tmp_path):
    _install_process_patches(monkeypatch)

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

    assert data.brainset.derived_version == "1.1.1"
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
    _install_process_patches(monkeypatch)
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
        "_split_twofold",
        lambda start, end, labels, **kwargs: {
            0: {
                "train": _FakeInterval(2),
                "val": _FakeInterval(1),
                "test": _FakeInterval(1),
            }
        },
    )

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
    n = 10
    start = np.arange(n, dtype=np.float64)
    end = start + 1.0
    labels = np.array([0, 1] * 5, dtype=np.int64)

    folds = m._split_twofold(start, end, labels, balance=False)
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
            assert len(train.start) == 5
            assert len(val.start) == 2
            assert len(test.start) == 3
        else:
            assert len(train.start) == 5
            assert len(val.start) == 2
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
    _install_process_patches(monkeypatch)

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
