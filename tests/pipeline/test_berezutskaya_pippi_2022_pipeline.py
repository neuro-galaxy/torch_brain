import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path

from torch_brain.data import Data, Interval, LazyInterval, RegularTimeSeries
from torch_brain.datasets import BerezutskayaPippi2022

add_pipelines_to_path()
from berezutskaya_pippi_2022 import pipeline as pippi_pipeline  # noqa: E402


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_local_bids_fixture(root: Path) -> Path:
    subject_id = "sub-01"
    recording_id = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    ieeg_dir = root / subject_id / "ses-iemu" / "ieeg"
    ieeg_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        root / "participants.tsv",
        "participant_id\tsex\tage\nsub-01\tF\t33\n",
    )
    _write_text(
        ieeg_dir / f"{recording_id}_channels.tsv",
        (
            "name\ttype\tstatus\tstatus_description\tsampling_frequency\tunits\n"
            "A1\tSEEG\tgood\tn/a\t512\tµV\n"
            "A2\tSEEG\tbad\tnoisy\t512\tµV\n"
            "TRIG\tTRIG\tgood\tn/a\t512\tn/a\n"
        ),
    )
    _write_text(
        ieeg_dir / f"{subject_id}_ses-iemu_acq-clinical_electrodes.tsv",
        "name\tx\ty\tz\nA1\t1\t2\t3\nA2\t4\t5\t6\nTRIG\t0\t0\t0\n",
    )
    _write_text(
        ieeg_dir / f"{subject_id}_ses-iemu_acq-clinical_coordsystem.json",
        json.dumps({"iEEGCoordinateSystem": "ACPC"}),
    )
    _write_text(
        ieeg_dir / f"{recording_id}_ieeg.json",
        json.dumps({"Manufacturer": "Micromed", "SamplingFrequency": 512}),
    )
    _write_text(ieeg_dir / f"{recording_id}_ieeg.vhdr", "Dummy header")
    _write_text(ieeg_dir / f"{recording_id}_ieeg.vmrk", "Dummy marker")
    _write_text(ieeg_dir / f"{recording_id}_ieeg.eeg", "dummy")
    _write_text(
        ieeg_dir / f"{subject_id}_ses-iemu_task-film_run-1_events.tsv",
        (
            "onset\tduration\ttrial_type\tvalue\n"
            "10.0\t0.0\tstart task\t9\n"
            "40.0\t0.0\tend task\t4\n"
        ),
    )
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        labels_dir / "speech_binary_labels.csv",
        "start_time_s,label\n0.5,1\n1.5,0\n2.5,1\n3.5,0\n",
    )
    _write_text(
        labels_dir / "speech_multiclass_labels.csv",
        "start_time_s,label\n0.5,2\n1.5,1\n2.5,2\n3.5,0\n",
    )
    return labels_dir


def test_build_manifest_row_preserves_acquisition():
    row = pippi_pipeline._build_manifest_row(
        "sub-45_ses-iemu_task-film_acq-clinical_run-1",
        available_relpaths=set(
            pippi_pipeline._recording_relative_paths(
                "sub-45_ses-iemu_task-film_acq-clinical_run-1"
            ).values()
        ),
    )
    assert row["test_subject"] == 45
    assert row["test_run"] == 1
    assert row["acquisition"] == "clinical"


def test_discover_local_manifest_rows_filters_to_seeg(tmp_path):
    labels_dir = _write_local_bids_fixture(tmp_path)
    rows = pippi_pipeline._discover_local_manifest_rows(tmp_path)
    assert [row["recording_id"] for row in rows] == [
        "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    ]
    assert labels_dir.exists()


def _write_brain_area_labels(root: Path, text: str) -> None:
    brain_areas = root / "brain_areas"
    brain_areas.mkdir(parents=True, exist_ok=True)
    _write_text(brain_areas / "brain_area_labels.csv", text)


def test_build_channels_adds_brain_area_labels_preserving_channel_order(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
            "01,A2,ctx-lh-middletemporal,ctx_lh_G_temporal_middle\n"
        ),
    )
    channel_table = pd.DataFrame(
        {
            "name": ["A2", "A1", "TRIG"],
            "type": ["SEEG", "SEEG", "TRIG"],
            "status": ["good", "good", "good"],
        }
    )
    electrodes_table = pd.DataFrame(
        {"name": ["A1", "A2"], "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]}
    )

    channels = pippi_pipeline._build_channels(
        channel_table,
        electrodes_table,
        seeg_names=["A2", "A1"],
        subject_number=1,
    )

    assert channels.name.tolist() == ["A2", "A1"]
    assert channels.label_dkt.tolist() == [
        "ctx-lh-middletemporal",
        "ctx-rh-superiorfrontal",
    ]
    assert channels.label_destrieux.tolist() == [
        "ctx_lh_G_temporal_middle",
        "ctx_rh_G_front_sup",
    ]


def test_build_channels_rejects_missing_brain_area_labels_for_covered_subject(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
        ),
    )
    channel_table = pd.DataFrame(
        {"name": ["A1", "A2"], "type": ["SEEG", "SEEG"], "status": ["good", "good"]}
    )
    electrodes_table = pd.DataFrame(
        {"name": ["A1", "A2"], "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]}
    )

    with pytest.raises(ValueError, match="Missing brain area labels.*A2"):
        pippi_pipeline._build_channels(
            channel_table,
            electrodes_table,
            seeg_names=["A1", "A2"],
            subject_number=1,
        )


def test_build_channels_rejects_duplicate_brain_area_label_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
            "01,A1,ctx-rh-middlefrontal,ctx_rh_G_front_middle\n"
        ),
    )
    channel_table = pd.DataFrame({"name": ["A1"], "type": ["SEEG"], "status": ["good"]})
    electrodes_table = pd.DataFrame(
        {"name": ["A1"], "x": [1.0], "y": [3.0], "z": [5.0]}
    )

    with pytest.raises(ValueError, match="Duplicate brain area labels"):
        pippi_pipeline._build_channels(
            channel_table,
            electrodes_table,
            seeg_names=["A1"],
            subject_number=1,
        )


def test_build_channels_adds_empty_brain_area_labels_for_uncovered_subject(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
        ),
    )
    channel_table = pd.DataFrame({"name": ["D1"], "type": ["SEEG"], "status": ["good"]})
    electrodes_table = pd.DataFrame(
        {"name": ["D1"], "x": [1.0], "y": [3.0], "z": [5.0]}
    )

    channels = pippi_pipeline._build_channels(
        channel_table,
        electrodes_table,
        seeg_names=["D1"],
        subject_number=2,
    )

    assert channels.label_dkt.tolist() == [""]
    assert channels.label_destrieux.tolist() == [""]


def test_build_channels_requires_nonempty_brain_area_label_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    channel_table = pd.DataFrame({"name": ["A1"], "type": ["SEEG"], "status": ["good"]})
    electrodes_table = pd.DataFrame(
        {"name": ["A1"], "x": [1.0], "y": [3.0], "z": [5.0]}
    )

    with pytest.raises(FileNotFoundError, match="brain_area_labels.csv"):
        pippi_pipeline._build_channels(
            channel_table,
            electrodes_table,
            seeg_names=["A1"],
            subject_number=1,
        )

    _write_brain_area_labels(
        tmp_path,
        "subject,electrode,label_dkt,label_destrieux\n",
    )
    with pytest.raises(ValueError, match="contains no rows"):
        pippi_pipeline._build_channels(
            channel_table,
            electrodes_table,
            seeg_names=["A1"],
            subject_number=1,
        )


def test_get_manifest_merges_local_and_remote_rows_with_local_precedence(
    tmp_path, monkeypatch
):
    local_rows = [{"recording_id": "rid-local", "acquisition": "local"}]
    remote_rows = [
        {"recording_id": "rid-local", "acquisition": "remote"},
        {"recording_id": "rid-remote", "acquisition": "remote"},
    ]
    monkeypatch.setattr(
        pippi_pipeline, "_discover_local_manifest_rows", lambda _raw_dir: local_rows
    )
    monkeypatch.setattr(
        pippi_pipeline, "_discover_remote_manifest_rows", lambda _raw_dir: remote_rows
    )

    manifest = pippi_pipeline.Pipeline.get_manifest(tmp_path, args=None)

    assert list(manifest.index) == ["rid-local", "rid-remote"]
    assert manifest.loc["rid-local", "acquisition"] == "local"
    assert manifest.loc["rid-remote", "acquisition"] == "remote"


def test_get_manifest_uses_local_rows_when_remote_discovery_fails(
    tmp_path, monkeypatch
):
    local_rows = [{"recording_id": "rid-local", "acquisition": "local"}]
    monkeypatch.setattr(
        pippi_pipeline, "_discover_local_manifest_rows", lambda _raw_dir: local_rows
    )

    def _raise_remote(_raw_dir):
        raise RuntimeError("network down")

    monkeypatch.setattr(pippi_pipeline, "_discover_remote_manifest_rows", _raise_remote)

    manifest = pippi_pipeline.Pipeline.get_manifest(tmp_path, args=None)

    assert list(manifest.index) == ["rid-local"]
    assert manifest.loc["rid-local", "acquisition"] == "local"


def test_build_manifest_row_allows_missing_participants_file():
    recording_id = "sub-45_ses-iemu_task-film_acq-clinical_run-1"
    relpaths = pippi_pipeline._recording_relative_paths(recording_id)
    available_relpaths = set(relpaths.values()) - {"participants.tsv"}

    row = pippi_pipeline._build_manifest_row(
        recording_id,
        available_relpaths=available_relpaths,
    )

    assert row["participants_relpath"] is None
    assert row["channels_relpath"] == relpaths["channels"]


@pytest.fixture
def pippi_preparation(tmp_path, monkeypatch):
    labels_dir = _write_local_bids_fixture(tmp_path)
    output_dir = tmp_path / "processed" / "berezutskaya_pippi_2022"
    # Spread events across the film so both temporal halves have labeled windows.
    _write_text(
        labels_dir / "speech_binary_labels.csv",
        "start_time_s,label\n0.5,1\n3.5,0\n8.0,1\n11.0,0\n"
        "16.0,1\n19.0,0\n23.0,1\n26.0,0\n",
    )
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
            "01,A2,ctx-lh-middletemporal,ctx_lh_G_temporal_middle\n"
        ),
    )

    class _FakeRaw:
        def __init__(self):
            self.ch_names = ["A1", "A2", "TRIG"]
            self.info = {
                "sfreq": 512.0,
                "meas_date": None,
            }
            self.times = np.arange(0.0, 50.0, 1 / 512.0)

        def get_data(self, picks):
            assert picks == ["A1", "A2"]
            a1 = np.sin(self.times)
            a2 = np.cos(self.times)
            return np.stack([a1, a2], axis=0)

    monkeypatch.setattr(
        pippi_pipeline,
        "mne",
        SimpleNamespace(
            io=SimpleNamespace(read_raw_brainvision=lambda *args, **kwargs: _FakeRaw())
        ),
    )

    return {
        "vhdr_path": tmp_path
        / "sub-01"
        / "ses-iemu"
        / "ieeg"
        / "sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr",
        "output_dir": output_dir,
        "labels_dir": labels_dir,
        "label_files": (),
        "pre_offset_s": 0.0,
        "post_offset_s": 1.0,
        "balance_splits": False,
    }


@pytest.mark.parametrize("task_start", [10.0, 10.0 + 1 / 1024])
def test_process_file_writes_expected_h5(pippi_preparation, task_start):
    raw_root = pippi_preparation["vhdr_path"].parents[3]
    _write_text(
        raw_root / "sub-01/ses-iemu/ieeg/sub-01_ses-iemu_task-film_run-1_events.tsv",
        "onset\tduration\ttrial_type\tvalue\n"
        f"{task_start}\t0.0\tstart task\t9\n"
        f"{task_start + 30.0}\t0.0\tend task\t4\n",
    )
    output_path = pippi_pipeline.process_file(**pippi_preparation)

    assert output_path.exists()
    with h5py.File(output_path, "r") as handle:
        assert handle["brainset"].attrs["derived_version"] == "1.1.2"
        assert "seeg_data" in handle
        assert "splits" in handle
        assert "channel_splits" in handle
        assert handle["seeg_data"].attrs["unit"] == "V"
        assert handle["seeg_data"].attrs["scale_to_uV"] == 1e6
        times = np.arange(0.0, 50.0, 1 / 512.0)
        keep = np.logical_and(times >= task_start, times <= task_start + 30.0)
        expected_signal = np.stack(
            [np.sin(times[keep]), np.cos(times[keep])],
            axis=1,
        ).astype(np.float32)
        np.testing.assert_allclose(handle["seeg_data"]["data"][:], expected_signal)
        channels_included = handle["channels"]["included"][:]
        assert channels_included.tolist() == [True, False]
        units = handle["channels"]["units"][:]
        assert [value.decode("ascii") for value in units.tolist()] == ["uV", "uV"]
        label_dkt = handle["channels"]["label_dkt"][:]
        assert [value.decode("ascii") for value in label_dkt.tolist()] == [
            "ctx-rh-superiorfrontal",
            "ctx-lh-middletemporal",
        ]
        label_destrieux = handle["channels"]["label_destrieux"][:]
        assert [value.decode("ascii") for value in label_destrieux.tolist()] == [
            "ctx_rh_G_front_sup",
            "ctx_lh_G_temporal_middle",
        ]
        np.testing.assert_allclose(handle["channels"]["coord_acpc_x"][:], [1.0, 4.0])
        np.testing.assert_allclose(handle["channels"]["coord_acpc_y"][:], [2.0, 5.0])
        np.testing.assert_allclose(handle["channels"]["coord_acpc_z"][:], [3.0, 6.0])
        split_key = pippi_pipeline.split_selector_key(
            subset_tier="full",
            label_mode="binary",
            task_name="speech",
            fold_idx=0,
            split_name="train",
        )
        high_cov_split_key = pippi_pipeline.split_selector_key(
            subset_tier="high-cov",
            label_mode="binary",
            task_name="speech",
            fold_idx=0,
            split_name="train",
        )
        low_cov_split_key = pippi_pipeline.split_selector_key(
            subset_tier="low-cov",
            label_mode="binary",
            task_name="speech",
            fold_idx=0,
            split_name="train",
        )
        assert split_key in handle["splits"]
        assert split_key in handle["channel_splits"]
        assert high_cov_split_key in handle["splits"]
        assert high_cov_split_key in handle["channel_splits"]
        assert low_cov_split_key not in handle["splits"]
        assert low_cov_split_key not in handle["channel_splits"]
        assert "label_maps_json" in handle.attrs

        label_times = np.array([0.5, 3.5, 8.0, 11.0, 16.0, 19.0, 23.0, 26.0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        for fold, indices_by_split in enumerate(
            (
                {"train": [4, 5, 6, 7], "val": [0, 1], "test": [2, 3]},
                {"train": [0, 1, 2, 3], "val": [4, 5], "test": [6, 7]},
            )
        ):
            for split, indices in indices_by_split.items():
                key = pippi_pipeline.split_selector_key(
                    subset_tier="full",
                    label_mode="binary",
                    task_name="speech",
                    fold_idx=fold,
                    split_name=split,
                )
                selected = handle["splits"][key]
                np.testing.assert_array_equal(
                    selected["start"][:], label_times[indices]
                )
                np.testing.assert_array_equal(
                    selected["end"][:], label_times[indices] + 1.0
                )
                np.testing.assert_array_equal(selected["label"][:], labels[indices])

    movie_times = times[keep] - task_start
    with Data.load(output_path) as data:
        np.testing.assert_allclose(data.seeg_data.timestamps, movie_times)
        np.testing.assert_allclose(data.domain.start, [movie_times[0]])
        np.testing.assert_allclose(data.domain.end, [movie_times[-1] + 1 / 512.0])

    recording_id = output_path.stem
    dataset = BerezutskayaPippi2022(
        root=str(pippi_preparation["output_dir"].parent),
        test_subject=1,
        test_session=1,
        split="train",
        fold=0,
        keep_files_open=False,
        uniquify_channel_ids_with_session=True,
    )
    with dataset.get_recording(recording_id) as recording:
        np.testing.assert_array_equal(recording.splits.start, [16.0, 19.0, 23.0, 26.0])
        np.testing.assert_array_equal(recording.splits.end, [17.0, 20.0, 24.0, 27.0])
        np.testing.assert_array_equal(recording.splits.label, [1, 0, 1, 0])
        np.testing.assert_array_equal(recording.channels.included, [True, False])
        np.testing.assert_array_equal(
            recording.channels.id,
            [f"sub-01/{recording_id}/0", f"sub-01/{recording_id}/1"],
        )


@pytest.mark.parametrize("existing", [False, True])
def test_failed_serialization_preserves_previous_output(
    pippi_preparation, monkeypatch, existing
):
    output_dir = pippi_preparation["output_dir"]
    output_dir.mkdir(parents=True)
    output = output_dir / "sub-01_ses-iemu_task-film_acq-clinical_run-1.h5"
    if existing:
        with h5py.File(output, "w") as handle:
            handle["previous"] = np.arange(5, dtype=np.int16)
        previous_bytes = output.read_bytes()
    serialize = Data.to_hdf5

    def interrupted_serialize(self, handle, *args, **kwargs):
        serialize(self, handle, *args, **kwargs)
        if hasattr(self, "seeg_data"):
            handle.flush()
            if existing:
                assert output.read_bytes() == previous_bytes
            else:
                assert not output.exists()
            raise RuntimeError("interrupted serialization")

    monkeypatch.setattr(Data, "to_hdf5", interrupted_serialize)
    with pytest.raises(RuntimeError, match="interrupted serialization"):
        pippi_pipeline.process_file(**pippi_preparation, no_splits=True)

    assert output.exists() is existing
    if existing:
        assert output.read_bytes() == previous_bytes
    assert not list(output_dir.glob("*.tmp"))


def _write_binary_label_csv(path: Path, labels: np.ndarray) -> None:
    lines = ["start_time_s,label"]
    lines.extend(f"{float(idx) + 0.5},{int(label)}" for idx, label in enumerate(labels))
    _write_text(path, "\n".join(lines) + "\n")


def _split_signature_from_h5(path: Path) -> dict[tuple[int, str], list[tuple]]:
    signatures = {}
    with h5py.File(path, "r") as handle:
        for fold_idx in (0, 1):
            for split_name in ("train", "val", "test"):
                split_key = pippi_pipeline.split_selector_key(
                    subset_tier="full",
                    label_mode="binary",
                    task_name="word_gap",
                    fold_idx=fold_idx,
                    split_name=split_name,
                )
                split = handle["splits"][split_key]
                signatures[(fold_idx, split_name)] = list(
                    zip(
                        split["start"][:].tolist(),
                        split["end"][:].tolist(),
                        split["label"][:].tolist(),
                        strict=True,
                    )
                )
    return signatures


def test_balanced_pippi_target_task_splits_ignore_other_label_changes(
    tmp_path, monkeypatch
):
    _write_local_bids_fixture(tmp_path)
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "subject,electrode,label_dkt,label_destrieux\n"
            "01,A1,ctx-rh-superiorfrontal,ctx_rh_G_front_sup\n"
            "01,A2,ctx-lh-middletemporal,ctx_lh_G_temporal_middle\n"
        ),
    )

    class _FakeRaw:
        def __init__(self):
            self.ch_names = ["A1", "A2", "TRIG"]
            self.info = {
                "sfreq": 512.0,
                "meas_date": None,
            }
            self.times = np.arange(0.0, 50.0, 1 / 512.0)

        def get_data(self, picks):
            assert picks == ["A1", "A2"]
            return np.stack([np.sin(self.times), np.cos(self.times)], axis=0)

    monkeypatch.setattr(
        pippi_pipeline, "_read_raw_brainvision", lambda _path: _FakeRaw()
    )

    target_labels = np.array([0, 0, 0, 1] * 6, dtype=np.int64)
    labels_a = tmp_path / "labels_a"
    labels_b = tmp_path / "labels_b"
    _write_binary_label_csv(
        labels_a / "global_flow_binary_labels.csv",
        np.array([0, 1] * 12, dtype=np.int64),
    )
    _write_binary_label_csv(labels_a / "word_gap_binary_labels.csv", target_labels)
    _write_binary_label_csv(
        labels_b / "global_flow_binary_labels.csv",
        np.array([0, 0, 0, 0, 0, 1] * 4, dtype=np.int64),
    )
    _write_binary_label_csv(labels_b / "word_gap_binary_labels.csv", target_labels)

    vhdr_path = (
        tmp_path
        / "sub-01"
        / "ses-iemu"
        / "ieeg"
        / "sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr"
    )
    label_files = [
        "global_flow_binary_labels.csv",
        "word_gap_binary_labels.csv",
    ]
    first_path = pippi_pipeline.process_file(
        vhdr_path=vhdr_path,
        output_dir=tmp_path / "processed_a",
        labels_dir=labels_a,
        label_files=label_files,
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=False,
        balance_splits=True,
        balance_seed=7,
    )
    second_path = pippi_pipeline.process_file(
        vhdr_path=vhdr_path,
        output_dir=tmp_path / "processed_b",
        labels_dir=labels_b,
        label_files=label_files,
        pre_offset_s=0.0,
        post_offset_s=1.0,
        no_splits=False,
        balance_splits=True,
        balance_seed=7,
    )

    assert _split_signature_from_h5(first_path) == _split_signature_from_h5(second_path)


def test_split_selector_key_rejects_invalid_subset_tier():
    with pytest.raises(ValueError, match="Invalid Pippi subset_tier"):
        pippi_pipeline.split_selector_key(
            subset_tier="unknown",
            label_mode="binary",
            task_name="speech",
            fold_idx=0,
            split_name="train",
        )


def test_subject_subset_tiers_match_expected_groups():
    assert pippi_pipeline.pippi_subset_tiers_for_subject(1) == ("full", "high-cov")
    assert pippi_pipeline.pippi_subset_tiers_for_subject(2) == ("full", "low-cov")
    assert pippi_pipeline.pippi_subset_tiers_for_subject(45) == ("full",)


def test_load_seeg_signal_uses_brainvision_header_sampling_rate(tmp_path, monkeypatch):
    _write_local_bids_fixture(tmp_path)
    channel_table = pd.read_csv(
        tmp_path
        / "sub-01"
        / "ses-iemu"
        / "ieeg"
        / "sub-01_ses-iemu_task-film_acq-clinical_run-1_channels.tsv",
        sep="\t",
    )
    vhdr_path = (
        tmp_path
        / "sub-01"
        / "ses-iemu"
        / "ieeg"
        / "sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr"
    )

    class _FakeRaw:
        def __init__(self):
            self.ch_names = ["A1", "A2", "TRIG"]
            # Deliberately differs from sidecar fixture values (512) to ensure
            # we source the rate from the BrainVision header.
            self.info = {"sfreq": 777.0}
            self.times = np.arange(0.0, 50.0, 1 / 777.0)

        def get_data(self, picks):
            assert picks == ["A1", "A2"]
            return np.stack([np.sin(self.times), np.cos(self.times)], axis=0)

    monkeypatch.setattr(pippi_pipeline, "_read_raw_brainvision", lambda _p: _FakeRaw())
    seeg_data, seeg_names = pippi_pipeline._load_seeg_signal(
        vhdr_path=vhdr_path,
        channel_table=channel_table,
        task_start_s=10.0,
        task_end_s=40.0,
    )

    assert isinstance(seeg_data, RegularTimeSeries)
    assert seeg_data.sampling_rate == 777.0
    assert seeg_names == ["A1", "A2"]


def test_pipeline_download_returns_manifest_derived_asset(tmp_path, monkeypatch):
    pipeline_instance = pippi_pipeline.Pipeline.__new__(pippi_pipeline.Pipeline)
    pipeline_instance.raw_dir = tmp_path
    pipeline_instance.args = SimpleNamespace(redownload=False)
    pipeline_instance.update_status = lambda _status: None

    vhdr_relpath = "sub-01/ses-iemu/ieeg/arbitrary_name_ieeg.vhdr"
    vhdr_path = tmp_path / vhdr_relpath
    vhdr_path.parent.mkdir(parents=True, exist_ok=True)
    vhdr_path.write_text("dummy", encoding="utf-8")
    recording_id = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    row = pippi_pipeline._build_manifest_row(
        recording_id,
        available_relpaths=set(
            pippi_pipeline._recording_relative_paths(recording_id).values()
        ),
    )
    row["vhdr_relpath"] = vhdr_relpath
    manifest_item = SimpleNamespace(Index=recording_id, **row)
    downloaded = []

    def download_file(bucket, key, target):
        assert bucket == pippi_pipeline.OPENNEURO_BUCKET
        downloaded.append(key.removeprefix(pippi_pipeline.OPENNEURO_PREFIX))
        Path(target).write_text("downloaded")

    monkeypatch.setattr(
        pippi_pipeline,
        "get_cached_s3_client",
        lambda: SimpleNamespace(download_file=download_file),
    )

    download_output = pippi_pipeline.Pipeline.download(pipeline_instance, manifest_item)

    assert isinstance(download_output, pippi_pipeline.DownloadedAsset)
    assert download_output.vhdr_path == vhdr_path
    assert download_output.recording_id == manifest_item.Index
    assert download_output.acquisition == "clinical"
    assert download_output.subject_number == 1
    assert download_output.run == 1
    assert set(downloaded) == {
        value
        for key, value in row.items()
        if key.endswith("_relpath") and key != "vhdr_relpath"
    }
    assert vhdr_path.read_text() == "dummy"
    downloaded.clear()
    pippi_pipeline.Pipeline.download(pipeline_instance, manifest_item)
    assert downloaded == []


def test_manifest_recovers_recording_with_missing_companion(tmp_path, monkeypatch):
    _write_local_bids_fixture(tmp_path)
    remote_rows = pippi_pipeline._discover_local_manifest_rows(tmp_path)
    (tmp_path / remote_rows[0]["eeg_relpath"]).unlink()
    monkeypatch.setattr(
        pippi_pipeline, "_discover_remote_manifest_rows", lambda raw_dir: remote_rows
    )
    manifest = pippi_pipeline.Pipeline.get_manifest(tmp_path, args=None)
    assert list(manifest.index) == [remote_rows[0]["recording_id"]]


def test_pipeline_process_uses_downloaded_asset_recording_id_for_skip(
    tmp_path, monkeypatch
):
    pipeline_instance = pippi_pipeline.Pipeline.__new__(pippi_pipeline.Pipeline)
    pipeline_instance.processed_dir = tmp_path / "processed"
    pipeline_instance.processed_dir.mkdir(parents=True, exist_ok=True)
    pipeline_instance.args = SimpleNamespace(reprocess=False)
    pipeline_instance.update_status = lambda _status: None

    recording_id = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    # If process() uses vhdr_path filename instead of recording_id, it will not skip.
    (pipeline_instance.processed_dir / f"{recording_id}.h5").write_text(
        "existing", encoding="utf-8"
    )
    called = []
    monkeypatch.setattr(
        pippi_pipeline,
        "process_file",
        lambda **kwargs: called.append(kwargs),
    )

    pippi_pipeline.Pipeline.process(
        pipeline_instance,
        pippi_pipeline.DownloadedAsset(
            vhdr_path=tmp_path / "sub-01" / "ses-iemu" / "ieeg" / "not_id_ieeg.vhdr",
            recording_id=recording_id,
            acquisition="clinical",
            subject_number=1,
            run=1,
        ),
    )

    assert called == []


def test_pipeline_process_passes_downloaded_asset_metadata_to_process_file(
    tmp_path, monkeypatch
):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_text(
        labels_dir / "speech_binary_labels.csv",
        "start_time_s,label\n0.0,1\n",
    )
    pipeline_instance = pippi_pipeline.Pipeline.__new__(pippi_pipeline.Pipeline)
    pipeline_instance.processed_dir = tmp_path / "processed"
    pipeline_instance.args = SimpleNamespace(
        reprocess=True,
        labels=None,
        labels_dir=str(labels_dir),
        pre_offset_s=0.25,
        post_offset_s=0.75,
        no_splits=True,
        balance_splits=False,
        balance_seed=9,
    )
    pipeline_instance.update_status = lambda _status: None

    captured = {}

    def _fake_process_file(**kwargs):
        captured.update(kwargs)
        return tmp_path / "processed" / "out.h5"

    monkeypatch.setattr(pippi_pipeline, "process_file", _fake_process_file)
    download_output = pippi_pipeline.DownloadedAsset(
        vhdr_path=tmp_path
        / "sub-01"
        / "ses-iemu"
        / "ieeg"
        / "arbitrary_name_ieeg.vhdr",
        recording_id="sub-01_ses-iemu_task-film_acq-clinical_run-1",
        acquisition="clinical",
        subject_number=1,
        run=1,
    )

    pippi_pipeline.Pipeline.process(pipeline_instance, download_output)

    assert captured["recording_id"] == download_output.recording_id
    assert captured["acquisition"] == download_output.acquisition
    assert captured["subject_number"] == download_output.subject_number
    assert captured["run"] == download_output.run


def test_pipeline_download_skips_optional_participants_relpath(tmp_path, monkeypatch):
    pipeline_instance = pippi_pipeline.Pipeline.__new__(pippi_pipeline.Pipeline)
    pipeline_instance.raw_dir = tmp_path
    pipeline_instance.args = SimpleNamespace(redownload=True)
    pipeline_instance.update_status = lambda _status: None

    downloaded_relpaths = []
    monkeypatch.setattr(pippi_pipeline, "get_cached_s3_client", lambda: object())
    monkeypatch.setattr(
        pippi_pipeline,
        "_download_relpath",
        lambda _client, *, relpath, target_root, overwrite: downloaded_relpaths.append(
            (relpath, target_root, overwrite)
        ),
    )

    recording_id = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    relpaths = pippi_pipeline._recording_relative_paths(recording_id)
    manifest_item = SimpleNamespace(
        Index=recording_id,
        acquisition="clinical",
        test_subject=1,
        test_run=1,
        participants_relpath=None,
        channels_relpath=relpaths["channels"],
        vhdr_relpath=relpaths["vhdr"],
        vmrk_relpath=relpaths["vmrk"],
        eeg_relpath=relpaths["eeg"],
        ieeg_json_relpath=relpaths["ieeg_json"],
        events_relpath=relpaths["events"],
        electrodes_relpath=relpaths["electrodes"],
        coordsystem_relpath=relpaths["coordsystem"],
    )

    pippi_pipeline.Pipeline.download(pipeline_instance, manifest_item)

    assert [relpath for relpath, _target_root, _overwrite in downloaded_relpaths] == [
        relpaths["channels"],
        relpaths["vhdr"],
        relpaths["vmrk"],
        relpaths["eeg"],
        relpaths["ieeg_json"],
        relpaths["events"],
        relpaths["electrodes"],
        relpaths["coordsystem"],
    ]


@pytest.mark.parametrize(
    "labels",
    [[0, 1, 0, 1, 0, 0, 1, 0], [0, 1, 2, 0, 1, 2, 0, 0], [0] * 8, []],
    ids=["binary", "multiclass", "single-class", "empty"],
)
def test_balanced_annotations_are_written_in_time_order(labels, tmp_path):
    labels = np.asarray(labels, dtype=np.int64)
    starts = np.array([3, 0, 1, 1, 2, 0.5, 4, 2.5], dtype=float)[: len(labels)]
    source = Interval(start=starts, end=starts + 1, label=labels)
    expected_rng, actual_rng = np.random.default_rng(42), np.random.default_rng(42)
    sampler = (
        pippi_pipeline._balanced_binary_indices
        if np.all(np.isin(np.unique(labels), [0, 1]))
        else pippi_pipeline._balanced_multiclass_indices
    )
    selected = sampler(labels, expected_rng)
    result = pippi_pipeline._balance_interval(source, actual_rng)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    assert np.all(np.diff(result.start) >= 0)
    expected = sorted(zip(starts[selected], labels[selected], strict=True))
    assert sorted(zip(result.start, result.label, strict=True)) == expected
    np.testing.assert_array_equal(result.end, result.start + 1)
    np.testing.assert_array_equal(source.start, starts)

    # Exercise the serialized annotations through the real lazy slicing path.
    path = tmp_path / "annotations.h5"
    with h5py.File(path, "w") as handle:
        result.to_hdf5(handle)
    with h5py.File(path, "r") as handle:
        sliced = LazyInterval.from_hdf5(handle).slice(0.75, 1.25, reset_origin=False)
        overlap = (result.end > 0.75) & (result.start < 1.25)
        np.testing.assert_array_equal(sliced.start, result.start[overlap])
        np.testing.assert_array_equal(sliced.label, result.label[overlap])
