from pathlib import Path

import h5py
import numpy as np
import pytest

from torch_brain.datasets.BerezutskayaPippi2022 import (
    BerezutskayaPippi2022,
    _from_recording_id,
    _to_recording_id,
)


def _mock_dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "berezutskaya_pippi_2022"


def _split_key(
    *,
    subset_tier: str,
    label_mode: str,
    h5_regime: str,
    task: str,
    fold: int,
    split: str,
) -> str:
    return f"{subset_tier}${label_mode}${h5_regime}${task}$fold{fold}${split}"


def _write_mock_h5(
    path: Path,
    *,
    subset_tiers: tuple[str, ...] = ("full",),
    label_mode: str = "binary",
    h5_regime: str = "within_session",
    task: str = "speech",
    fold: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        channels = h5.create_group("channels")
        channels.create_dataset("id", data=np.array(["0"], dtype="S8"))
        channels.create_dataset("name", data=np.array(["A1"], dtype="S8"))
        channels.create_dataset("included", data=np.array([True], dtype=bool))
        channels.create_dataset("x", data=np.array([1.0], dtype=float))
        channels.create_dataset("y", data=np.array([2.0], dtype=float))
        channels.create_dataset("z", data=np.array([3.0], dtype=float))
        seeg_data = h5.create_group("seeg_data")
        seeg_data.attrs["unit"] = "V"
        seeg_data.attrs["scale_to_uV"] = 1e6

        channel_splits = h5.create_group("channel_splits")
        splits_group = h5.create_group("splits")
        for subset_tier in subset_tiers:
            for split in splits:
                key = _split_key(
                    subset_tier=subset_tier,
                    label_mode=label_mode,
                    h5_regime=h5_regime,
                    task=task,
                    fold=fold,
                    split=split,
                )
                channel_splits.create_dataset(key, data=np.array([True], dtype=bool))
                interval = splits_group.create_group(key)
                interval.create_dataset("start", data=np.array([0.0], dtype=float))
                interval.create_dataset("end", data=np.array([1.0], dtype=float))
                interval.create_dataset("label", data=np.array([1], dtype=int))

        subject = h5.create_group("subject")
        subject.create_dataset("id", data=np.bytes_("sub-01"))
        session = h5.create_group("session")
        session.create_dataset("id", data=np.bytes_(path.stem))
        domain = h5.create_group("domain")
        domain.create_dataset("start", data=np.array([0.0], dtype=float))
        domain.create_dataset("end", data=np.array([1.0], dtype=float))


def _write_mock_recordings(
    tmp_path: Path,
    recording_ids: tuple[str, ...],
    *,
    subset_tiers: tuple[str, ...] = ("full",),
    fold: int = 0,
) -> None:
    dataset_dir = _mock_dataset_dir(tmp_path)
    for recording_id in recording_ids:
        _write_mock_h5(
            dataset_dir / f"{recording_id}.h5",
            subset_tiers=subset_tiers,
            fold=fold,
        )


def _make_dataset(tmp_path: Path, **overrides) -> BerezutskayaPippi2022:
    kwargs = {
        "root": str(tmp_path),
        "keep_files_open": False,
    }
    use_explicit_ids = (
        "recording_ids" in overrides and overrides["recording_ids"] is not None
    )
    if not use_explicit_ids:
        kwargs.update(
            {
                "subset_tier": "full",
                "label_mode": "binary",
                "task": "speech",
                "regime": "within-session",
                "test_subject": 1,
                "test_session": 1,
                "split": "train",
            }
        )
    kwargs.update(overrides)
    return BerezutskayaPippi2022(**kwargs)


def test_recording_id_roundtrip():
    recording_id = _to_recording_id(45, "clinical", 1)
    assert recording_id == "sub-45_ses-iemu_task-film_acq-clinical_run-1"
    assert _from_recording_id(recording_id) == (45, "clinical", 1)


@pytest.mark.parametrize(
    ("subject", "acquisition", "run"),
    [
        (True, "clinical", 1),
        (-1, "clinical", 1),
        (1, "", 1),
        (1, "clinical", 0),
    ],
)
def test_recording_id_rejects_invalid_inputs(subject, acquisition, run):
    with pytest.raises(ValueError, match="Invalid Pippi recording-id components"):
        _to_recording_id(subject=subject, acquisition=acquisition, run=run)


def test_split_selection_accepts_fold1_for_all_regimes(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
        fold=1,
    )
    for regime in (
        "within-session",
        "hold-in-session",
        "hold-out-session",
        "hold-out-subject",
    ):
        ds = _make_dataset(tmp_path, regime=regime, fold=1)
        assert ds.fold == 1


def test_split_selection_rejects_invalid_fold(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _make_dataset(tmp_path, fold=2)


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("within-session", 2),
        ("hold-in-session", 2),
        ("hold-out-session", 2),
        ("hold-out-subject", 2),
    ],
)
def test_num_folds_for_regime_reports_expected_count(regime: str, expected: int):
    assert BerezutskayaPippi2022.num_folds_for_regime(regime) == expected


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_within_session_uses_target_recording(tmp_path: Path, split: str):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
    )
    ds = _make_dataset(tmp_path, regime="within-session", split=split)
    assert ds.recording_ids == ["sub-01_ses-iemu_task-film_acq-clinical_run-1"]


def test_hold_in_session_train_uses_all_eligible_recordings(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
    )
    ds = _make_dataset(tmp_path, regime="hold-in-session", split="train")
    assert ds.recording_ids == [
        "sub-01_ses-iemu_task-film_acq-clinical_run-1",
        "sub-01_ses-iemu_task-film_acq-clinical_run-2",
        "sub-02_ses-iemu_task-film_acq-clinical_run-1",
    ]


def test_hold_out_session_train_excludes_target(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
    )
    ds = _make_dataset(tmp_path, regime="hold-out-session", split="train")
    assert ds.recording_ids == [
        "sub-01_ses-iemu_task-film_acq-clinical_run-2",
        "sub-02_ses-iemu_task-film_acq-clinical_run-1",
    ]


def test_hold_out_subject_train_excludes_target_subject(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
    )
    ds = _make_dataset(tmp_path, regime="hold-out-subject", split="train")
    assert ds.recording_ids == ["sub-02_ses-iemu_task-film_acq-clinical_run-1"]


def test_high_cov_subset_tier_filters_to_high_cov_subjects(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
            "sub-23_ses-iemu_task-film_acq-clinical_run-1",
        ),
        subset_tiers=("full", "high-cov", "low-cov"),
    )
    ds = _make_dataset(
        tmp_path,
        subset_tier="high-cov",
        regime="hold-in-session",
        split="train",
        test_subject=1,
        test_session=1,
    )
    assert ds.recording_ids == [
        "sub-01_ses-iemu_task-film_acq-clinical_run-1",
        "sub-23_ses-iemu_task-film_acq-clinical_run-1",
    ]


def test_low_cov_subset_tier_filters_to_low_cov_subjects(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
            "sub-50_ses-iemu_task-film_acq-clinical_run-1",
        ),
        subset_tiers=("full", "high-cov", "low-cov"),
    )
    ds = _make_dataset(
        tmp_path,
        subset_tier="low-cov",
        regime="hold-in-session",
        split="train",
        test_subject=2,
        test_session=1,
    )
    assert ds.recording_ids == [
        "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        "sub-50_ses-iemu_task-film_acq-clinical_run-1",
    ]


def test_subset_tier_rejects_test_recording_outside_tier(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-02_ses-iemu_task-film_acq-clinical_run-1",
        ),
        subset_tiers=("full", "high-cov", "low-cov"),
    )
    with pytest.raises(ValueError, match="No eligible Pippi recording found"):
        _make_dataset(
            tmp_path,
            subset_tier="high-cov",
            test_subject=2,
            test_session=1,
        )


def test_split_selection_rejects_ambiguous_subject_run_pair(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-HDgrid_run-1",
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous Pippi selection"):
        _make_dataset(tmp_path)


def test_explicit_recording_mode_rejects_split_args(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    with pytest.raises(ValueError, match="split-selection args must be omitted"):
        _make_dataset(
            tmp_path,
            recording_ids=["sub-01_ses-iemu_task-film_acq-clinical_run-1"],
            split="train",
        )


def test_get_sampling_intervals_requires_benchmark_mode(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub-01_ses-iemu_task-film_acq-clinical_run-1"],
    )
    with pytest.raises(RuntimeError, match="benchmark mode"):
        ds.get_sampling_intervals()


def test_sampling_rate_property_raises_with_per_recording_guidance(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    with pytest.raises(RuntimeError, match="Use get_sampling_rate\\(recording_id\\)"):
        _ = ds.sampling_rate


def test_get_sampling_rate_returns_per_recording_values(tmp_path: Path, monkeypatch):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
        ),
    )
    ds = _make_dataset(
        tmp_path,
        recording_ids=[
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            "sub-01_ses-iemu_task-film_acq-clinical_run-2",
        ],
    )

    class _FakeSignal:
        def __init__(self, sampling_rate: float):
            self.sampling_rate = sampling_rate

    class _FakeRecording:
        def __init__(self, sampling_rate: float):
            self.seeg_data = _FakeSignal(sampling_rate)

    monkeypatch.setattr(
        ds,
        "get_recording",
        lambda rid: (
            _FakeRecording(512.0) if rid.endswith("run-1") else _FakeRecording(2048.0)
        ),
    )

    assert ds.get_sampling_rate("sub-01_ses-iemu_task-film_acq-clinical_run-1") == 512.0
    assert (
        ds.get_sampling_rate("sub-01_ses-iemu_task-film_acq-clinical_run-2") == 2048.0
    )


def test_get_sampling_rate_caches_per_recording_id(tmp_path: Path, monkeypatch):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub-01_ses-iemu_task-film_acq-clinical_run-1"],
    )
    calls = []

    class _FakeSignal:
        sampling_rate = 512.0

    class _FakeRecording:
        seeg_data = _FakeSignal()

    def _fake_get_recording(rid: str):
        calls.append(rid)
        return _FakeRecording()

    monkeypatch.setattr(ds, "get_recording", _fake_get_recording)

    rid = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    assert ds.get_sampling_rate(rid) == 512.0
    assert ds.get_sampling_rate(rid) == 512.0
    assert calls == [rid]


def test_get_channel_metadata_requires_name_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        included = np.array([True, False], dtype=bool)
        coord_acpc_x = np.array([1.0, 2.0], dtype=float)
        coord_acpc_y = np.array([3.0, 4.0], dtype=float)
        coord_acpc_z = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")


def test_get_channel_metadata_requires_included_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        coord_acpc_x = np.array([1.0, 2.0], dtype=float)
        coord_acpc_y = np.array([3.0, 4.0], dtype=float)
        coord_acpc_z = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")


def test_get_channel_metadata_rejects_coordinate_length_mismatch(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        included = np.array([True, False], dtype=bool)
        coord_acpc_x = np.array([1.0, 2.0, 3.0], dtype=float)
        coord_acpc_y = np.array([4.0, 5.0, 6.0], dtype=float)
        coord_acpc_z = np.array([7.0, 8.0, 9.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(
        ValueError,
        match="coord_acpc_x.*expected length 2.*actual length 3",
    ):
        ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")


def test_get_channel_metadata_reports_acpc_coordinate_frame(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["A1", "A2"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_acpc_x = np.array([1.0, 2.0], dtype=float)
        coord_acpc_y = np.array([3.0, 4.0], dtype=float)
        coord_acpc_z = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    arrays = ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")

    assert "coords" not in arrays
    assert "coords_type" not in arrays
    assert set(arrays["coordinate_frames"]) == {"acpc"}
    np.testing.assert_allclose(
        arrays["coordinate_frames"]["acpc"],
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=float),
    )


def test_get_channel_metadata_includes_optional_group_and_hemisphere(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["A1", "A2"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_acpc_x = np.array([1.0, 2.0], dtype=float)
        coord_acpc_y = np.array([3.0, 4.0], dtype=float)
        coord_acpc_z = np.array([5.0, 6.0], dtype=float)
        group = np.array(["OFC", "AMY"], dtype=object)
        hemisphere = np.array(["L", "R"], dtype=object)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    arrays = ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")

    assert arrays["group"].tolist() == ["OFC", "AMY"]
    assert arrays["hemisphere"].tolist() == ["L", "R"]


def test_get_channel_metadata_includes_optional_brain_area_labels(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["A2", "A1"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_acpc_x = np.array([1.0, 2.0], dtype=float)
        coord_acpc_y = np.array([3.0, 4.0], dtype=float)
        coord_acpc_z = np.array([5.0, 6.0], dtype=float)
        label_dkt = np.array(
            ["ctx-lh-middletemporal", "ctx-rh-superiorfrontal"], dtype=object
        )
        label_destrieux = np.array(
            ["ctx_lh_G_temporal_middle", "ctx_rh_G_front_sup"],
            dtype=object,
        )

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    arrays = ds.get_channel_metadata("sub-01_ses-iemu_task-film_acq-clinical_run-1")

    assert arrays["label_dkt"].tolist() == [
        "ctx-lh-middletemporal",
        "ctx-rh-superiorfrontal",
    ]
    assert arrays["label_destrieux"].tolist() == [
        "ctx_lh_G_temporal_middle",
        "ctx_rh_G_front_sup",
    ]


def test_get_neural_signal_metadata_reads_saved_seeg_attrs(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-01_ses-iemu_task-film_acq-clinical_run-1",),
    )
    ds = _make_dataset(tmp_path)

    assert ds.get_neural_signal_metadata(
        "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    ) == {"unit": "V", "scale_to_uV": 1e6}


def test_get_neural_signal_metadata_requires_101_attrs(tmp_path: Path):
    recording_id = "sub-01_ses-iemu_task-film_acq-clinical_run-1"
    _write_mock_recordings(tmp_path, (recording_id,))
    with h5py.File(_mock_dataset_dir(tmp_path) / f"{recording_id}.h5", "a") as h5:
        del h5["seeg_data"].attrs["scale_to_uV"]
    ds = _make_dataset(tmp_path)

    with pytest.raises(ValueError, match="brainsets 1.1.0 neural signal metadata"):
        ds.get_neural_signal_metadata(recording_id)
