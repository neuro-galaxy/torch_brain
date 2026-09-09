import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
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
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_hdf5(self, file, serialize_fn_map=None):
        file["written"] = True
        file["seeg_data"] = SimpleNamespace(attrs={})


class _FakeH5File:
    def __init__(self, *args, **kwargs):
        self._store = {}
        Path(args[0]).touch()

    def __enter__(self):
        return self._store

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeInterval:
    def __init__(self, n):
        self.start = np.arange(n, dtype=np.float64)
        self.end = self.start + 1.0
        self.label = np.ones(n, dtype=np.int64)


def _install_common_patches(monkeypatch):
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
    monkeypatch.setattr(
        m,
        "_apply_task_window",
        lambda lfp_data, lfp_time, trials_df, edge: (
            np.zeros((30, 2), dtype=np.float32),
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


def _extract_audit_events(caplog):
    prefix = "BYD_PREP_AUDIT "
    events = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith(prefix):
            events.append(json.loads(msg[len(prefix) :]))
    return events


def test_process_file_emits_success_audit_log(monkeypatch, caplog, tmp_path):
    _install_common_patches(monkeypatch)
    monkeypatch.setattr(m, "_extract_encoding_interval", lambda trials_df: (0.0, 10.0))

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
    _install_common_patches(monkeypatch)
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
    _install_common_patches(monkeypatch)
    monkeypatch.setattr(m.h5py, "File", real_h5_file)
    monkeypatch.setattr(m, "_extract_encoding_interval", lambda trials_df: (0.0, 10.0))

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

    def process_file(*args, **kwargs):
        calls.append(args)
        with h5py.File(output, "w") as handle:
            signal = handle.create_group("seeg_data")
            signal["data"] = np.zeros((10, 2), dtype=np.float32)
            signal.attrs["unit"] = "V"
            signal.attrs["scale_to_uV"] = 1e6
            for key in ("channels", "subject", "session", "domain"):
                handle.create_group(key)

    monkeypatch.setattr(m, "process_file", process_file)
    pipeline = m.Pipeline(tmp_path, tmp_path, args=None)
    raw = tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"
    pipeline.process(raw)
    assert len(calls) == 1
    assert m._is_valid_processed_file(output)
    pipeline.process(raw)
    assert len(calls) == 1
