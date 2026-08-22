from pathlib import Path

import h5py
import numpy as np
import pytest

from torch_brain.data import Interval
from torch_brain.datasets.NeuroprobeV2 import (
    NeuroprobeV2,
    _from_recording_id,
    _to_recording_id,
)


def _mock_dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "neuroprobe_2025"


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        channels = h5.create_group("channels")
        channels.create_dataset("id", data=np.array(["ch0"], dtype="S8"))
        seeg_data = h5.create_group("seeg_data")
        seeg_data.attrs["unit"] = "uV"
        seeg_data.attrs["scale_to_uV"] = 1.0
        channel_splits = h5.create_group("channel_splits")

        if include_channel_masks:
            for split in splits:
                channel_splits.create_dataset(
                    _split_key(
                        subset_tier=subset_tier,
                        label_mode=label_mode,
                        h5_regime=h5_regime,
                        task=task,
                        fold=fold,
                        split=split,
                    ),
                    data=np.array([True], dtype=bool),
                )

        splits_group = h5.create_group("splits")
        for split in splits:
            interval_key = _split_key(
                subset_tier=subset_tier,
                label_mode=label_mode,
                h5_regime=h5_regime,
                task=task,
                fold=fold,
                split=split,
            )
            interval_group = splits_group.create_group(interval_key)
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
) -> None:
    dataset_dir = _mock_dataset_dir(tmp_path)
    for recording_id in recording_ids:
        _write_mock_h5(
            dataset_dir / f"{recording_id}.h5",
            subset_tier=subset_tier,
            fold=fold,
            include_channel_masks=include_channel_masks,
        )


def _make_dataset(tmp_path: Path, **overrides) -> NeuroprobeV2:
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
    return NeuroprobeV2(**kwargs)


def test_recording_id_roundtrip():
    recording_id = _to_recording_id(2, 4)
    assert recording_id == "sub_2_trial004"
    assert _from_recording_id(recording_id) == (2, 4)


def test_recording_id_accepts_numpy_integer_inputs():
    assert _to_recording_id(np.int64(2), np.int32(4)) == "sub_2_trial004"


@pytest.mark.parametrize(
    ("subject", "session"),
    [
        (True, 4),
        (-1, 4),
        ("2", 4),
        (2, False),
        (2, 1000),
        (2, "4"),
    ],
)
def test_recording_id_rejects_invalid_inputs(subject, session):
    with pytest.raises(ValueError, match="_to_recording_id received invalid"):
        _to_recording_id(subject=subject, session=session)


@pytest.mark.parametrize(
    "regime",
    ("within-session", "hold-in-session", "hold-out-session", "hold-out-subject"),
)
def test_split_selection_accepts_fold1_for_all_regimes(tmp_path: Path, regime: str):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
        fold=1,
    )
    ds = _make_dataset(tmp_path, regime=regime, fold=1)
    assert ds.fold == 1


def test_split_selection_rejects_invalid_fold(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
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
    assert NeuroprobeV2.num_folds_for_regime(regime) == expected


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_within_session_always_uses_target_recording(tmp_path: Path, split: str):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="within-session",
        split=split,
    )
    assert ds.recording_ids == ["sub_1_trial001"]


def test_hold_in_session_train_uses_all_eligible_recordings(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="train",
    )
    assert ds.recording_ids == ["sub_1_trial001", "sub_1_trial002", "sub_2_trial004"]


def test_hold_in_session_eval_uses_target_only(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="test",
    )
    assert ds.recording_ids == ["sub_1_trial001"]


def test_hold_out_session_train_excludes_target(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-out-session",
        split="train",
    )
    assert ds.recording_ids == ["sub_1_trial002", "sub_2_trial004"]


def test_hold_out_subject_train_excludes_target_subject(tmp_path: Path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002", "sub_2_trial004"),
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-out-subject",
        split="train",
    )
    assert ds.recording_ids == ["sub_2_trial004"]


def test_full_subset_tier_discovers_ids_from_disk_and_ignores_non_recording_h5(
    tmp_path: Path,
):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_2_trial004"),
        subset_tier="full",
    )
    _write_mock_h5(
        _mock_dataset_dir(tmp_path) / "metadata_blob.h5",
        subset_tier="full",
    )
    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="train",
    )
    assert ds.recording_ids == ["sub_1_trial001", "sub_2_trial004"]


def test_subset_tier_target_eligibility_checked(tmp_path: Path):
    with pytest.raises(ValueError, match="not eligible for subset_tier 'lite'"):
        _make_dataset(
            tmp_path,
            subset_tier="lite",
            regime="within-session",
            split="test",
            test_subject=9,
            test_session=9,
        )


def test_missing_split_key_no_longer_fails_fast_during_init(tmp_path: Path):
    dataset_dir = _mock_dataset_dir(tmp_path)
    _write_mock_h5(dataset_dir / "sub_1_trial001.h5", subset_tier="full", fold=1)
    _write_mock_h5(dataset_dir / "sub_1_trial002.h5", subset_tier="full", fold=0)

    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="train",
        fold=1,
    )
    assert ds.recording_ids == ["sub_1_trial001", "sub_1_trial002"]


def test_missing_channel_key_no_longer_fails_fast_during_init(tmp_path: Path):
    dataset_dir = _mock_dataset_dir(tmp_path)
    _write_mock_h5(dataset_dir / "sub_1_trial001.h5", subset_tier="full", fold=0)
    _write_mock_h5(
        dataset_dir / "sub_2_trial004.h5",
        subset_tier="full",
        fold=0,
        include_channel_masks=False,
    )

    ds = _make_dataset(
        tmp_path,
        regime="hold-in-session",
        split="train",
    )
    assert ds.recording_ids == ["sub_1_trial001", "sub_2_trial004"]


def test_explicit_recording_ids_reject_split_selection_args(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")
    with pytest.raises(ValueError, match="Unexpected args: split"):
        _make_dataset(
            tmp_path,
            recording_ids=["sub_1_trial001"],
            split="train",
        )


def test_uniquify_channel_ids_option_sets_multichannel_mixin_components(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject=True)
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject is True
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_uniquify_channel_ids_option_accepts_non_boolean_passthrough(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject="yes")
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject == "yes"
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_describe_selection_excludes_interval_path(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    selection = ds.describe_selection()
    assert "interval_path" not in selection


def test_get_sampling_intervals_reads_current_splits_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

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
    assert list(intervals.keys()) == ["sub_1_trial001"]
    assert intervals["sub_1_trial001"] is fake_recording.splits


def test_get_recording_hook_sets_active_split_interval_on_data(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

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
            self.subject = type("_Subject", (), {"id": "1"})()
            self.session = type("_Session", (), {"id": "sub_1_trial001"})()

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
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

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
            self.subject = type("_Subject", (), {"id": "1"})()
            self.session = type("_Session", (), {"id": "sub_1_trial001"})()

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
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)
    split_key = ds._split_key()

    class _FakeChannels:
        id = np.array(["ch0"], dtype=str)
        included = np.array([False], dtype=bool)

    class _FakeRecording:
        def __init__(self):
            self.channels = _FakeChannels()
            self.splits = object()
            self.subject = type("_Subject", (), {"id": "1"})()
            self.session = type("_Session", (), {"id": "sub_1_trial001"})()

        def get_nested_attribute(self, path: str):
            raise KeyError(path)

    with pytest.raises(KeyError, match="channel_splits") as excinfo:
        ds.get_recording_hook(_FakeRecording())

    message = str(excinfo.value)
    assert f"channel_splits.{split_key}" in message
    assert f"splits.{split_key}" in message


def test_get_channel_metadata_returns_full_arrays_with_included_mask(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        included = np.array([True, False])
        coord_btb_lip_l = np.array([1.0, 2.0], dtype=float)
        coord_btb_lip_i = np.array([3.0, 4.0], dtype=float)
        coord_btb_lip_p = np.array([5.0, 6.0], dtype=float)
        coord_btb_x = np.array([7.0, 8.0], dtype=float)
        coord_btb_y = np.array([9.0, 10.0], dtype=float)
        coord_btb_z = np.array([11.0, 12.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    channel_arrays = ds.get_channel_metadata("sub_1_trial001")
    assert channel_arrays["ids"].tolist() == ["c0", "c1"]
    assert channel_arrays["names"].tolist() == ["A", "B"]
    assert channel_arrays["included_mask"].tolist() == [True, False]
    assert channel_arrays["indices"].tolist() == [0, 1]
    assert "coords" not in channel_arrays
    assert "coords_type" not in channel_arrays
    assert set(channel_arrays["coordinate_frames"]) == {"btb_lip", "btb_xyz"}
    np.testing.assert_allclose(
        channel_arrays["coordinate_frames"]["btb_lip"],
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=float),
    )
    np.testing.assert_allclose(
        channel_arrays["coordinate_frames"]["btb_xyz"],
        np.array([[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]], dtype=float),
    )


def test_sampling_rate_reports_expected_value(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    assert ds.sampling_rate == 2048.0


def test_get_channel_metadata_requires_coordinate_fields(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        included = np.array([True, False])

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(AttributeError, match="Missing required channel coordinate"):
        ds.get_channel_metadata("sub_1_trial001")


def test_get_channel_metadata_requires_btb_xyz_field_lengths(
    tmp_path: Path, monkeypatch
):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        included = np.array([True, False])
        coord_btb_lip_l = np.array([1.0, 2.0], dtype=float)
        coord_btb_lip_i = np.array([3.0, 4.0], dtype=float)
        coord_btb_lip_p = np.array([5.0, 6.0], dtype=float)
        coord_btb_x = np.array([7.0], dtype=float)
        coord_btb_y = np.array([9.0, 10.0], dtype=float)
        coord_btb_z = np.array([11.0, 12.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(
        ValueError,
        match="coord_btb_x.*expected length 2.*actual length 1",
    ):
        ds.get_channel_metadata("sub_1_trial001")


def test_get_channel_metadata_requires_name_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        included = np.array([True, False])
        coord_btb_lip_l = np.array([1.0, 2.0], dtype=float)
        coord_btb_lip_i = np.array([3.0, 4.0], dtype=float)
        coord_btb_lip_p = np.array([5.0, 6.0], dtype=float)
        coord_btb_x = np.array([7.0, 8.0], dtype=float)
        coord_btb_y = np.array([9.0, 10.0], dtype=float)
        coord_btb_z = np.array([11.0, 12.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub_1_trial001")


def test_get_channel_metadata_requires_included_field(tmp_path: Path, monkeypatch):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")

    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        coord_btb_lip_l = np.array([1.0, 2.0], dtype=float)
        coord_btb_lip_i = np.array([3.0, 4.0], dtype=float)
        coord_btb_lip_p = np.array([5.0, 6.0], dtype=float)
        coord_btb_x = np.array([7.0, 8.0], dtype=float)
        coord_btb_y = np.array([9.0, 10.0], dtype=float)
        coord_btb_z = np.array([11.0, 12.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    with pytest.raises(AttributeError):
        ds.get_channel_metadata("sub_1_trial001")


def test_get_neural_signal_metadata_reads_saved_seeg_attrs(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")
    ds = _make_dataset(tmp_path)

    assert ds.get_neural_signal_metadata("sub_1_trial001") == {
        "unit": "uV",
        "scale_to_uV": 1.0,
    }


def test_get_neural_signal_metadata_requires_101_attrs(tmp_path: Path):
    _write_mock_recordings(tmp_path, ("sub_1_trial001",), subset_tier="full")
    with h5py.File(_mock_dataset_dir(tmp_path) / "sub_1_trial001.h5", "a") as h5:
        del h5["seeg_data"].attrs["scale_to_uV"]
    ds = _make_dataset(tmp_path)

    with pytest.raises(ValueError, match="brainsets 1.1.0 neural signal metadata"):
        ds.get_neural_signal_metadata("sub_1_trial001")
