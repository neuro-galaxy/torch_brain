from pathlib import Path

import h5py
import numpy as np
import pytest

from torch_brain.data import Interval
from torch_brain.datasets.KelesBYD2024 import (
    KelesBYD2024,
    _from_recording_id,
    _to_recording_id,
)


def _mock_dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "keles_byd_2024"


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
    subset_tier: str,
    label_mode: str = "binary",
    h5_regime: str = "within_session",
    task: str = "speech",
    fold: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    include_channel_masks: bool = True,
    legacy_split_schema: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        channels = h5.create_group("channels")
        channels.create_dataset("id", data=np.array(["ch0"], dtype="S8"))
        seeg_data = h5.create_group("seeg_data")
        seeg_data.attrs["unit"] = "V"
        seeg_data.attrs["scale_to_uV"] = 1e6
        channel_splits = None
        splits_group = None
        if not legacy_split_schema:
            channel_splits = h5.create_group("channel_splits")
            splits_group = h5.create_group("splits")

        if include_channel_masks:
            for split in splits:
                split_key = _split_key(
                    subset_tier=subset_tier,
                    label_mode=label_mode,
                    h5_regime=h5_regime,
                    task=task,
                    fold=fold,
                    split=split,
                )
                if legacy_split_schema:
                    channels.create_dataset(
                        f"included_{label_mode}_{task}_fold{fold}_{split}",
                        data=np.array([True], dtype=bool),
                    )
                else:
                    channel_splits.create_dataset(
                        split_key,
                        data=np.array([True], dtype=bool),
                    )

        for split in splits:
            interval_key = _split_key(
                subset_tier=subset_tier,
                label_mode=label_mode,
                h5_regime=h5_regime,
                task=task,
                fold=fold,
                split=split,
            )
            interval_group = (
                h5.create_group(f"{label_mode}_{task}_fold{fold}_{split}")
                if legacy_split_schema
                else splits_group.create_group(interval_key)
            )
            interval_group.create_dataset("start", data=np.array([0.0], dtype=float))
            interval_group.create_dataset("end", data=np.array([1.0], dtype=float))
            interval_group.create_dataset("label", data=np.array([1], dtype=int))


def _write_mock_recordings(
    tmp_path: Path,
    recording_ids: tuple[str, ...],
    *,
    subset_tier: str,
    fold: int = 0,
    include_channel_masks: bool = True,
    filename_suffix: str = "",
    legacy_split_schema: bool = False,
) -> None:
    dataset_dir = _mock_dataset_dir(tmp_path)
    for recording_id in recording_ids:
        _write_mock_h5(
            dataset_dir / f"{recording_id}{filename_suffix}.h5",
            subset_tier=subset_tier,
            fold=fold,
            include_channel_masks=include_channel_masks,
            legacy_split_schema=legacy_split_schema,
        )


def _make_dataset(tmp_path: Path, **overrides) -> KelesBYD2024:
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
                "test_subject": 44,
                "test_session": 1,
                "split": "train",
            }
        )
    kwargs.update(overrides)
    return KelesBYD2024(**kwargs)


def test_recording_id_roundtrip():
    recording_id = _to_recording_id(44, 1)
    assert recording_id == "sub-CS44_ses-P44CSR1"
    assert _from_recording_id(recording_id) == (44, 1)


@pytest.mark.parametrize(
    ("subject", "session"),
    [
        (True, 1),
        (-1, 1),
        ("44", 1),
        (44, False),
        (44, "1"),
    ],
)
def test_recording_id_rejects_invalid_inputs(subject, session):
    with pytest.raises(ValueError, match="_to_recording_id received invalid"):
        _to_recording_id(subject=subject, session=session)


def test_recording_id_rejects_mismatched_cs_and_p_subject():
    with pytest.raises(ValueError, match="does not match"):
        _from_recording_id("sub-CS44_ses-P45CSR1")


@pytest.mark.parametrize(
    "regime",
    ("within-session", "hold-in-session", "hold-out-session", "hold-out-subject"),
)
def test_split_selection_accepts_fold1_for_all_regimes(tmp_path: Path, regime: str):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
        fold=1,
    )
    ds = _make_dataset(tmp_path, regime=regime, fold=1)
    assert ds.fold == 1


def test_split_selection_rejects_invalid_fold(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
        fold=0,
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _make_dataset(tmp_path, regime="within-session", fold=2)


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
    assert KelesBYD2024.num_folds_for_regime(regime) == expected


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_within_session_always_uses_target_recording(tmp_path: Path, split: str):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="within-session",
        split=split,
    )
    assert ds.recording_ids == ["sub-CS44_ses-P44CSR1"]


def test_within_session_accepts_behavior_ecephys_suffix_files(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
        filename_suffix="_behavior+ecephys",
    )
    ds = _make_dataset(
        tmp_path,
        regime="within-session",
        split="test",
    )
    assert ds.recording_ids == ["sub-CS44_ses-P44CSR1"]
    assert (
        ds._resolve_storage_recording_id("sub-CS44_ses-P44CSR1")
        == "sub-CS44_ses-P44CSR1_behavior+ecephys"
    )


def test_hold_in_session_train_uses_all_eligible_recordings(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="train",
    )
    assert ds.recording_ids == [
        "sub-CS44_ses-P44CSR1",
        "sub-CS44_ses-P44CSR2",
        "sub-CS62_ses-P62CSR2",
    ]


def test_hold_out_session_train_excludes_target(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-out-session",
        split="train",
    )
    assert ds.recording_ids == ["sub-CS44_ses-P44CSR2", "sub-CS62_ses-P62CSR2"]


def test_hold_out_subject_train_excludes_target_subject(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        (
            "sub-CS44_ses-P44CSR1",
            "sub-CS44_ses-P44CSR2",
            "sub-CS62_ses-P62CSR2",
        ),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-out-subject",
        split="train",
    )
    assert ds.recording_ids == ["sub-CS62_ses-P62CSR2"]


def test_subset_tier_rejects_non_full_value(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    with pytest.raises(ValueError, match="Invalid subset_tier"):
        _make_dataset(
            tmp_path,
            subset_tier="lite",
            regime="within-session",
            split="test",
            test_subject=44,
            test_session=1,
        )


def test_explicit_recording_ids_reject_split_selection_args(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    with pytest.raises(ValueError, match="Unexpected args: split"):
        _make_dataset(
            tmp_path,
            recording_ids=["sub-CS44_ses-P44CSR1"],
            split="train",
        )


def test_explicit_recording_ids_accept_behavior_ecephys_suffix_files(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-CS44_ses-P44CSR1",),
        subset_tier="full",
        filename_suffix="_behavior+ecephys",
    )
    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub-CS44_ses-P44CSR1"],
    )
    assert ds.recording_ids == ["sub-CS44_ses-P44CSR1"]
    assert (
        ds._resolve_storage_recording_id("sub-CS44_ses-P44CSR1")
        == "sub-CS44_ses-P44CSR1_behavior+ecephys"
    )


def test_within_session_accepts_legacy_selector_schema(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub-CS44_ses-P44CSR1",),
        subset_tier="full",
        filename_suffix="_behavior+ecephys",
        legacy_split_schema=True,
    )
    ds = _make_dataset(
        tmp_path,
        regime="within-session",
        split="test",
    )
    legacy_split_key = ds._legacy_split_key()

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        included = np.array([False], dtype=bool)

    class _FakeRecording:
        def __init__(self):
            self.channels = _FakeChannels()
            self.splits = object()
            self.paths = []
            self.subject = type("_Subject", (), {"id": "44"})()
            self.session = type("_Session", (), {"id": "sub-CS44_ses-P44CSR1"})()

        def get_nested_attribute(self, path: str):
            self.paths.append(path)
            if path == f"channels.included_{legacy_split_key}":
                return np.array([True], dtype=bool)
            if path == legacy_split_key:
                return Interval(
                    start=np.array([0.0]),
                    end=np.array([1.0]),
                    label=np.array([1]),
                )
            raise KeyError(path)

    rec = _FakeRecording()
    ds.get_recording_hook(rec)

    assert rec.splits.start.tolist() == [0.0]
    assert rec.splits.end.tolist() == [1.0]
    assert rec.splits.label.tolist() == [1]
    assert rec.channels.included.tolist() == [True]


def test_uniquify_channel_ids_option_sets_multichannel_mixin_components(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject=True)
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject is True
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_uniquify_channel_ids_option_accepts_non_boolean_passthrough(
    tmp_path: Path,
):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject="yes")
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject == "yes"
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_describe_selection_excludes_interval_path(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    selection = ds.describe_selection()
    assert "interval_path" not in selection


def test_get_sampling_intervals_reads_current_splits_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    class _FakeRecording:
        splits = Interval(
            start=np.array([0.0]),
            end=np.array([1.0]),
            label=np.array([1]),
        )

    fake_recording = _FakeRecording()
    monkeypatch.setattr(ds, "get_recording", lambda _rid: fake_recording)

    intervals = ds.get_sampling_intervals()
    assert list(intervals.keys()) == ["sub-CS44_ses-P44CSR1"]
    assert intervals["sub-CS44_ses-P44CSR1"] is fake_recording.splits


def test_get_sampling_intervals_requires_benchmark_mode(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path, recording_ids=["sub-CS44_ses-P44CSR1"])

    with pytest.raises(RuntimeError, match="benchmark mode"):
        ds.get_sampling_intervals()


def test_get_recording_hook_sets_active_split_interval_on_data(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    split_key = ds._split_key()

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        included = np.array([False], dtype=bool)

    class _FakeRecording:
        def __init__(self):
            self.channels = _FakeChannels()
            self.splits = object()
            self.paths = []
            self.subject = type("_Subject", (), {"id": "44"})()
            self.session = type("_Session", (), {"id": "sub-CS44_ses-P44CSR1"})()

        def get_nested_attribute(self, path: str):
            self.paths.append(path)
            if path == f"channel_splits.{split_key}":
                return np.array([True], dtype=bool)
            if path == f"splits.{split_key}":
                return Interval(
                    start=np.array([0.0]),
                    end=np.array([1.0]),
                    label=np.array([1]),
                )
            raise KeyError(path)

    rec = _FakeRecording()
    ds.get_recording_hook(rec)

    assert isinstance(rec.splits, Interval)
    assert rec.splits.start.tolist() == [0.0]
    assert rec.splits.end.tolist() == [1.0]
    assert rec.splits.label.tolist() == [1]
    assert rec.channels.included.tolist() == [True]
    assert rec.paths == [f"channel_splits.{split_key}", f"splits.{split_key}"]


def test_get_recording_hook_does_not_read_legacy_channel_split_path(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    split_key = ds._split_key()

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        included = np.array([False], dtype=bool)

    class _FakeRecording:
        def __init__(self):
            self.channels = _FakeChannels()
            self.splits = object()
            self.paths = []
            self.session = type("_Session", (), {"id": "sub-CS44_ses-P44CSR1"})()

        def get_nested_attribute(self, path: str):
            self.paths.append(path)
            if path == f"channel_splits.{split_key}":
                raise KeyError(path)
            if path == f"splits.{split_key}":
                return Interval(
                    start=np.array([0.0]),
                    end=np.array([1.0]),
                    label=np.array([1]),
                )
            raise KeyError(path)

    with pytest.raises(KeyError, match="channel_splits"):
        ds.get_recording_hook(_FakeRecording())


def test_get_recording_hook_requires_channel_split_and_interval_paths(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    split_key = ds._split_key()

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        included = np.array([False], dtype=bool)

    class _FakeRecording:
        def __init__(self):
            self.channels = _FakeChannels()
            self.splits = object()
            self.session = type("_Session", (), {"id": "sub-CS44_ses-P44CSR1"})()

        def get_nested_attribute(self, path: str):
            raise KeyError(path)

    with pytest.raises(KeyError, match="channel_splits") as excinfo:
        ds.get_recording_hook(_FakeRecording())

    message = str(excinfo.value)
    assert f"channel_splits.{split_key}" in message
    assert f"splits.{split_key}" in message


def test_get_channel_metadata_reads_byd_coordinate_frame(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["c0", "c1"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_byd_mni152_r = np.array([0.18, 21.12], dtype=float)
        coord_byd_mni152_a = np.array([26.56, -5.97], dtype=float)
        coord_byd_mni152_s = np.array([17.92, -14.22], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    arrays = ds.get_channel_metadata("sub-CS44_ses-P44CSR1")

    assert "coords" not in arrays
    assert "coords_type" not in arrays
    assert set(arrays["coordinate_frames"]) == {"byd_mni152_ras"}
    np.testing.assert_allclose(
        arrays["coordinate_frames"]["byd_mni152_ras"],
        np.array([[0.18, 26.56, 17.92], [21.12, -5.97, -14.22]], dtype=float),
    )


def test_sampling_rate_prefers_seeg_data_sampling_rate_field(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeSeeg:
        sampling_rate = 512.0
        timestamps = np.array([0.0, 1.0], dtype=np.float64)

    class _FakeRecording:
        seeg_data = _FakeSeeg()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    assert ds.sampling_rate == 512.0


def test_sampling_rate_requires_seeg_sampling_rate_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeSeeg:
        timestamps = np.array([0.0, 0.002, 0.004, 0.006], dtype=np.float64)

    class _FakeRecording:
        seeg_data = _FakeSeeg()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(AttributeError, match="must expose sampling_rate"):
        _ = ds.sampling_rate


def test_get_channel_metadata_requires_xyz_fields(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        name = np.array(["c0"], dtype=str)
        included = np.array([True], dtype=bool)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(AttributeError, match="Missing required channel coordinate"):
        ds.get_channel_metadata("sub-CS44_ses-P44CSR1")


def test_get_channel_metadata_requires_byd_coordinate_field_lengths(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["c0", "c1"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_byd_mni152_r = np.array([1.0], dtype=float)
        coord_byd_mni152_a = np.array([3.0, 4.0], dtype=float)
        coord_byd_mni152_s = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(
        ValueError,
        match="coord_byd_mni152_r.*expected length 2.*actual length 1",
    ):
        ds.get_channel_metadata("sub-CS44_ses-P44CSR1")


def test_get_channel_metadata_ignores_legacy_localization_fields_for_byd(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        name = np.array(["c0"], dtype=str)
        included = np.array([True], dtype=bool)
        localization_L = np.array([1.0], dtype=float)
        localization_I = np.array([2.0], dtype=float)
        localization_P = np.array([3.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(AttributeError, match="Missing required channel coordinate"):
        ds.get_channel_metadata("sub-CS44_ses-P44CSR1")


def test_get_channel_metadata_requires_name_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        included = np.array([True, False], dtype=bool)
        coord_byd_mni152_r = np.array([1.0, 2.0], dtype=float)
        coord_byd_mni152_a = np.array([3.0, 4.0], dtype=float)
        coord_byd_mni152_s = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub-CS44_ses-P44CSR1")


def test_get_channel_metadata_requires_included_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["ch0", "ch1"], dtype=str)
        name = np.array(["c0", "c1"], dtype=str)
        coord_byd_mni152_r = np.array([1.0, 2.0], dtype=float)
        coord_byd_mni152_a = np.array([3.0, 4.0], dtype=float)
        coord_byd_mni152_s = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub-CS44_ses-P44CSR1")


def test_get_neural_signal_metadata_reads_seeg_data_attrs(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    assert ds.get_neural_signal_metadata("sub-CS44_ses-P44CSR1") == {
        "unit": "V",
        "scale_to_uV": 1e6,
    }


def test_get_neural_signal_metadata_requires_101_attrs(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub-CS44_ses-P44CSR1",), subset_tier="full")
    with h5py.File(_mock_dataset_dir(tmp_path) / "sub-CS44_ses-P44CSR1.h5", "a") as h5:
        del h5["seeg_data"].attrs["unit"]
    ds = _make_dataset(tmp_path)

    with pytest.raises(ValueError, match="brainsets 1.1.0 neural signal metadata"):
        ds.get_neural_signal_metadata("sub-CS44_ses-P44CSR1")
