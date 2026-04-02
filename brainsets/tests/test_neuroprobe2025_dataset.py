from pathlib import Path

import h5py
import numpy as np
import pytest
from temporaldata import Interval

from brainsets.datasets.Neuroprobe2025 import (
    Neuroprobe2025,
    _from_recording_id,
    _to_recording_id,
)


def _mock_dataset_dir(tmp_path: Path) -> Path:
    """Return the expected Neuroprobe dataset directory under tmp_path."""
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
    label_mode: str,
    h5_regime: str,
    task: str = "speech",
    fold: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        channels = h5.create_group("channels")
        channels.create_dataset("id", data=np.array(["ch0"], dtype="S8"))
        channel_splits = h5.create_group("channel_splits")
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
    h5_regime: str,
    label_mode: str = "binary",
    task: str = "speech",
    fold: int = 0,
) -> None:
    """Create compatible mock H5 recordings for the given ids."""
    dataset_dir = _mock_dataset_dir(tmp_path)
    for recording_id in recording_ids:
        _write_mock_h5(
            dataset_dir / f"{recording_id}.h5",
            subset_tier=subset_tier,
            label_mode=label_mode,
            h5_regime=h5_regime,
            task=task,
            fold=fold,
        )


def _write_default_recordings(
    tmp_path: Path, recording_ids: tuple[str, ...] = ("sub_1_trial001",)
) -> None:
    _write_mock_recordings(
        tmp_path,
        recording_ids,
        subset_tier="full",
        h5_regime="within_session",
    )


def _make_dataset(tmp_path: Path, **overrides) -> Neuroprobe2025:
    """Build a Neuroprobe2025 instance with concise defaults for tests."""
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
                "regime": "SS-SM",
                "test_subject": 1,
                "test_session": 1,
                "split": "train",
            }
        )
    kwargs.update(overrides)
    return Neuroprobe2025(**kwargs)


def test_recording_id_roundtrip():
    recording_id = _to_recording_id(2, 4)
    assert recording_id == "sub_2_trial004"
    assert _from_recording_id(recording_id) == (2, 4)


def test_to_recording_id_accepts_numpy_integer_inputs():
    assert _to_recording_id(np.int64(2), np.int32(4)) == "sub_2_trial004"


@pytest.mark.parametrize(
    ("subject", "session"),
    [
        (-1, 0),
        (1, 1000),
        ("1", 1),
        (1, "1"),
        (True, 1),
        (1, False),
    ],
)
def test_to_recording_id_invalid_subject_session_raise(subject, session):
    with pytest.raises(ValueError, match="_to_recording_id") as excinfo:
        _to_recording_id(subject, session)

    message = str(excinfo.value)
    assert f"subject={subject!r}" in message
    assert f"session={session!r}" in message


@pytest.mark.parametrize(
    ("recording_id", "error_fragment"),
    [
        ("subject2_trial4", "Invalid recording_id"),
        ("sub_1_trial4", "zero-padded 3-digit session"),
    ],
)
def test_recording_id_invalid_inputs_raise(recording_id: str, error_fragment: str):
    with pytest.raises(ValueError, match=error_fragment):
        _from_recording_id(recording_id)


def test_nano_cross_x_rejected_fast(tmp_path):
    with pytest.raises(ValueError, match="not compatible with cross_x"):
        Neuroprobe2025(
            root=str(tmp_path),
            subset_tier="nano",
            label_mode="binary",
            task="speech",
            regime="SS-DM",
            test_subject=1,
            test_session=1,
            split="train",
            keep_files_open=False,
        )


def test_cross_x_fold_must_be_zero(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="cross_x",
    )

    with pytest.raises(ValueError, match="must be 0"):
        _make_dataset(
            tmp_path,
            subset_tier="full",
            regime="SS-DM",
            split="test",
            fold=1,
        )


def test_split_selection_accepts_fold1_for_ss_sm(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, regime="SS-SM", fold=1)
    assert ds.fold == 1


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("SS-SM", 2),
        ("SS-DM", 1),
        ("DS-DM", 1),
    ],
)
def test_num_folds_for_regime_reports_expected_counts(regime: str, expected: int):
    assert Neuroprobe2025.num_folds_for_regime(regime) == expected


@pytest.mark.parametrize("fold", [False, ""])
def test_split_selection_defaults_falsy_fold_to_zero(tmp_path, fold):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, fold=fold)
    assert ds.fold == 0


def test_split_selection_rejects_invalid_regime(tmp_path):
    _write_default_recordings(tmp_path)

    with pytest.raises(ValueError, match="Invalid regime"):
        _make_dataset(tmp_path, regime="BAD")


def test_ss_dm_lite_train_selects_other_trial(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002"),
        subset_tier="lite",
        h5_regime="cross_x",
    )

    ds = _make_dataset(
        tmp_path,
        subset_tier="lite",
        regime="SS-DM",
        split="train",
    )
    assert ds.recording_ids == ["sub_1_trial002"]


def test_ds_dm_train_selects_fixed_anchor(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_3_trial001", "sub_2_trial004"),
        subset_tier="full",
        h5_regime="cross_x",
    )

    ds = _make_dataset(
        tmp_path,
        regime="DS-DM",
        test_subject=3,
        test_session=1,
        split="train",
    )
    assert ds.recording_ids == ["sub_2_trial004"]


def test_active_recording_ids_default_to_split_selection(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)
    selection = ds.describe_selection()
    assert selection["active_recording_ids"] == ["sub_1_trial001"]


def test_active_recording_ids_use_explicit_recording_ids(tmp_path):
    _write_default_recordings(tmp_path, ("sub_1_trial001", "sub_2_trial004"))

    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub_2_trial004", "sub_1_trial001"],
    )
    selection = ds.describe_selection()
    assert selection["active_recording_ids"] == ["sub_1_trial001", "sub_2_trial004"]


def test_explicit_recording_ids_allow_construction_without_split_args(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="nano",
        h5_regime="cross_x",
    )

    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub_1_trial001"],
    )
    assert ds.recording_ids == ["sub_1_trial001"]


def test_explicit_recording_ids_reject_split_selection_args(tmp_path):
    _write_default_recordings(tmp_path)

    with pytest.raises(ValueError, match="Unexpected args: split"):
        _make_dataset(
            tmp_path,
            recording_ids=["sub_1_trial001"],
            split="train",
        )


def test_explicit_recording_ids_reject_fold_arg(tmp_path):
    _write_default_recordings(tmp_path)

    with pytest.raises(ValueError, match="Unexpected args: fold"):
        _make_dataset(
            tmp_path,
            recording_ids=["sub_1_trial001"],
            fold=0,
        )


def test_split_selection_defaults_fold_to_zero(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)
    assert ds.fold == 0
    assert ds.describe_selection()["fold"] == 0


@pytest.mark.parametrize("label_mode", ["", False])
def test_split_selection_defaults_falsy_label_mode_to_binary(tmp_path, label_mode):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, label_mode=label_mode)
    assert ds.label_mode == "binary"


@pytest.mark.parametrize("task", ["", False])
def test_split_selection_defaults_falsy_task_to_speech(tmp_path, task):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, task=task)
    assert ds.task == "speech"


@pytest.mark.parametrize("regime", ["", False])
def test_split_selection_defaults_falsy_regime_to_ss_sm(tmp_path, regime):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, regime=regime)
    assert ds.regime == "SS-SM"


def test_describe_selection_excludes_selected_recording_ids(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)
    selection = ds.describe_selection()
    assert "selected_recording_ids" not in selection


def test_prune_to_split_is_no_longer_supported(tmp_path):
    _write_default_recordings(tmp_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'prune_to_split'"):
        _make_dataset(tmp_path, prune_to_split=True)


def test_get_sampling_rate_is_fixed_constant(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)
    assert ds.sampling_rate == 2048.0


def test_uniquify_channel_ids_option_sets_multichannel_mixin_components(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject=True)
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject is True
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_uniquify_channel_ids_option_accepts_non_boolean_passthrough(tmp_path):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path, uniquify_channel_ids_with_subject="yes")
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject == "yes"
    assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


def test_get_sampling_intervals_reads_current_splits_field(tmp_path, monkeypatch):
    _write_default_recordings(tmp_path)

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


def test_get_recording_hook_sets_active_split_interval_on_data(tmp_path):
    _write_default_recordings(tmp_path)

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


def test_get_recording_hook_does_not_read_legacy_channel_split_path(tmp_path):
    _write_default_recordings(tmp_path)

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


def test_get_recording_hook_requires_channel_split_and_interval_paths(tmp_path):
    _write_default_recordings(tmp_path)

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


def test_get_domain_intervals_uses_selected_recording_domains(tmp_path, monkeypatch):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)

    class _FakeRecording:
        domain = Interval(
            start=np.array([10.0]),
            end=np.array([20.0]),
            label=np.array([1]),
        )

    seen_recording_ids = []
    fake_recording = _FakeRecording()

    def _fake_get_recording(recording_id: str):
        seen_recording_ids.append(recording_id)
        return fake_recording

    monkeypatch.setattr(ds, "get_recording", _fake_get_recording)

    intervals = ds.get_domain_intervals()
    assert list(intervals.keys()) == ["sub_1_trial001"]
    assert intervals["sub_1_trial001"] is fake_recording.domain
    assert seen_recording_ids == ["sub_1_trial001"]


def test_domain_intervals_use_active_recording_ids_and_sampling_requires_split_mode(
    tmp_path, monkeypatch
):
    _write_default_recordings(tmp_path, ("sub_1_trial001", "sub_2_trial004"))
    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub_2_trial004", "sub_1_trial001"],
    )

    class _FakeRecording:
        def __init__(self, recording_id: str):
            self.recording_id = recording_id

        def get_nested_attribute(self, _path: str):
            return Interval(
                start=np.array([0.0]),
                end=np.array([1.0]),
                label=np.array([1]),
            )

        @property
        def domain(self):
            rid_value = int(self.recording_id.split("_trial")[1])
            return Interval(
                start=np.array([float(rid_value)]),
                end=np.array([float(rid_value + 1)]),
                label=np.array([1]),
            )

    monkeypatch.setattr(ds, "get_recording", lambda rid: _FakeRecording(rid))

    with pytest.raises(RuntimeError, match="benchmark mode"):
        ds.get_sampling_intervals()

    domain_intervals = ds.get_domain_intervals()
    assert list(domain_intervals.keys()) == ["sub_1_trial001", "sub_2_trial004"]


def test_get_channel_metadata_returns_full_arrays_with_included_mask(
    tmp_path, monkeypatch
):
    _write_default_recordings(tmp_path)

    ds = _make_dataset(tmp_path)

    class _FakeChannels:
        id = np.array(["c0", "c1"])
        name = np.array(["A", "B"])
        included = np.array([True, False])
        localization_L = np.array([1.0, 2.0], dtype=float)
        localization_I = np.array([3.0, 4.0], dtype=float)
        localization_P = np.array([5.0, 6.0], dtype=float)

    class _FakeRecording:
        channels = _FakeChannels()

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())
    channel_metadata = ds.get_channel_metadata("sub_1_trial001")
    assert channel_metadata["ids"].tolist() == ["c0", "c1"]
    assert channel_metadata["names"].tolist() == ["A", "B"]
    assert channel_metadata["included_mask"].tolist() == [True, False]
    assert channel_metadata["coords_type"] == "lip"
    assert channel_metadata["indices"].tolist() == [0, 1]
    assert channel_metadata["coords"] is not None
    assert channel_metadata["coords"].shape == (2, 3)


def test_get_channel_ids_use_recording_view_ids_without_suffix(tmp_path, monkeypatch):
    _write_default_recordings(tmp_path, ("sub_1_trial001", "sub_2_trial004"))

    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub_2_trial004", "sub_1_trial001"],
    )

    class _FakeRecording:
        class channels:
            id = np.array(["ch0", "ch1"])
            included = np.array([True, False])

    monkeypatch.setattr(ds, "get_recording", lambda _rid: _FakeRecording())

    assert ds.get_channel_ids() == ["ch0", "ch0", "ch1", "ch1"]


def test_get_channel_ids_return_prefixed_ids_from_recording_view(tmp_path, monkeypatch):
    _write_default_recordings(tmp_path, ("sub_1_trial001", "sub_2_trial004"))

    ds = _make_dataset(
        tmp_path,
        recording_ids=["sub_2_trial004", "sub_1_trial001"],
        uniquify_channel_ids_with_subject=True,
        uniquify_channel_ids_with_session=True,
    )

    class _Recording1:
        class channels:
            id = np.array(["1/001/ch0"])
            included = np.array([True])

    class _Recording2:
        class channels:
            id = np.array(["2/004/ch0"])
            included = np.array([True])

    views = {
        "sub_1_trial001": _Recording1(),
        "sub_2_trial004": _Recording2(),
    }
    monkeypatch.setattr(ds, "get_recording", lambda rid: views[rid])

    assert ds.get_channel_ids() == ["1/001/ch0", "2/004/ch0"]
