import io
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path
from torch.utils.data import Subset

from torch_brain.data import ArrayDict, Data, RegularTimeSeries

add_pipelines_to_path()
from neuroprobe_2025 import (  # noqa: E402  # ty:ignore[unresolved-import]
    pipeline as neuroprobe_pipeline,
)


@pytest.mark.parametrize("empty", [False, True])
def test_interval_export_is_chronological_and_stable(monkeypatch, empty):
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "neuroprobe_config",
        SimpleNamespace(SAMPLING_RATE=2),
        raising=False,
    )
    starts = np.array([] if empty else [6, 0, 2, 2], dtype=np.int64)
    labels = np.array([] if empty else [1, 0, 2, 3], dtype=np.int64)
    dataset = SimpleNamespace(get_window_bounds=lambda: (starts, starts + 2, labels))
    interval = neuroprobe_pipeline._intervals_from_dataset(dataset)
    np.testing.assert_array_equal(interval.start, [] if empty else [0, 1, 1, 3])
    np.testing.assert_array_equal(interval.end, [] if empty else [1, 2, 2, 4])
    np.testing.assert_array_equal(interval.label, [] if empty else [0, 2, 3, 1])


def test_extract_neural_data_uses_sample_index_timestamps(tmp_path):
    input_file = tmp_path / "sub_1_trial001.h5"
    with h5py.File(input_file, "w") as handle:
        data_group = handle.create_group("data")
        data_group.create_dataset("electrode_1", data=np.array([1.0, 2.0, 3.0]))
        data_group.create_dataset("electrode_2", data=np.array([4.0, 5.0, 6.0]))

    class _Channels:
        h5_label = np.array(["electrode_1", "electrode_2"], dtype=np.str_)

        def __len__(self):
            return len(self.h5_label)

    neuroprobe_pipeline.neuroprobe_config = SimpleNamespace(SAMPLING_RATE=2.0)
    seeg_data = neuroprobe_pipeline._extract_neural_data(input_file, _Channels())

    assert isinstance(seeg_data, RegularTimeSeries)
    np.testing.assert_allclose(
        np.asarray(seeg_data.data), np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    )
    np.testing.assert_allclose(
        np.asarray(seeg_data.timestamps), np.array([0.0, 0.5, 1.0], dtype=np.float64)
    )


def test_download_file_applies_timeout_and_retries(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.bin"
    payload = b"ok"
    seen_timeouts = []
    calls = {"count": 0}

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _fake_urlopen(url, timeout=None):
        seen_timeouts.append(timeout)
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("stalled")
        return _FakeResponse(payload)

    monkeypatch.setattr(neuroprobe_pipeline.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_TIMEOUT_SECONDS", 7)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_MAX_RETRIES", 2)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(neuroprobe_pipeline.time, "sleep", lambda _seconds: None)

    neuroprobe_pipeline._download_file(
        "https://example.com/artifact.bin",
        destination,
        overwrite=True,
    )

    assert destination.read_bytes() == payload
    assert calls["count"] == 3
    assert seen_timeouts == [7, 7, 7]
    assert not destination.with_suffix(".bin.tmp").exists()


def test_process_prepares_worker_runtime_even_when_processing_is_skipped(
    tmp_path, monkeypatch
):
    pipeline_instance = neuroprobe_pipeline.Pipeline.__new__(
        neuroprobe_pipeline.Pipeline
    )
    pipeline_instance.raw_dir = tmp_path / "raw"
    pipeline_instance.processed_dir = tmp_path / "processed"
    pipeline_instance.processed_dir.mkdir(parents=True, exist_ok=True)
    pipeline_instance.args = SimpleNamespace(reprocess=False, no_splits=True)
    pipeline_instance.update_status = lambda _status: None

    input_file = tmp_path / "already_downloaded_asset.h5"
    input_file.touch()
    with h5py.File(pipeline_instance.processed_dir / input_file.name, "w") as handle:
        handle.create_group("brainset").attrs["derived_version"] = "1.2.0"
        for key in ("subject", "session", "domain", "channels", "seeg_data"):
            handle.create_group(key)
        handle["seeg_data/data"] = np.zeros((4, 2), dtype=np.float32)
        handle["channels/id"] = np.array(["a", "b"], dtype="S1")
        handle.attrs["preparation_settings_json"] = (
            '{"splits_enabled":false,"pre_offset_s":0.0,"post_offset_s":1.0,"label_files":{}}'
        )

    prepared_raw_dirs = []
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_prepare_neuroprobe_lib",
        lambda raw_dir: prepared_raw_dirs.append(raw_dir),
    )

    neuroprobe_pipeline.Pipeline.process(
        pipeline_instance,
        neuroprobe_pipeline.DownloadedAsset(
            path=input_file,
            subject_id=1,
            trial_id=1,
        ),
    )
    assert prepared_raw_dirs == [pipeline_instance.raw_dir]


@pytest.fixture
def neuroprobe_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(neuroprobe_pipeline, "version", lambda _: "0.2.0")
    input_file = tmp_path / "sub_1_trial001.h5"
    signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with h5py.File(input_file, "w") as handle:
        handle.create_dataset("data/electrode_1", data=signal)

    pipeline = neuroprobe_pipeline.Pipeline.__new__(neuroprobe_pipeline.Pipeline)
    pipeline.raw_dir = tmp_path
    pipeline.processed_dir = tmp_path / "processed"
    pipeline.processed_dir.mkdir()
    pipeline.args = SimpleNamespace(reprocess=False, no_splits=True)
    pipeline.update_status = lambda _status: None
    monkeypatch.setattr(neuroprobe_pipeline, "_prepare_neuroprobe_lib", lambda _: None)
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "neuroprobe",
        SimpleNamespace(BrainTreebankSubject=lambda **_: object()),
        raising=False,
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "neuroprobe_config",
        SimpleNamespace(SAMPLING_RATE=2.0),
        raising=False,
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_extract_channel_data",
        lambda _: ArrayDict(h5_label=np.array(["electrode_1"])),
    )

    return pipeline, neuroprobe_pipeline.DownloadedAsset(input_file, 1, 1), signal


def test_process_preserves_signal_and_units_when_saving(neuroprobe_preparation):
    pipeline, asset, signal = neuroprobe_preparation
    pipeline.process(asset)

    output_file = pipeline.processed_dir / asset.path.name
    with Data.load(output_file) as data:
        assert data.session.id == "sub_1_trial001"
        np.testing.assert_array_equal(data.seeg_data.data, signal[:, None])
        np.testing.assert_array_equal(data.seeg_data.timestamps, [0.0, 0.5, 1.0])
    with h5py.File(output_file, "r") as handle:
        assert handle["seeg_data"].attrs["unit"] == "uV"
        assert handle["seeg_data"].attrs["scale_to_uV"] == 1.0


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure_stage", ["save", "attributes"])
def test_failed_publication_preserves_previous_output(
    neuroprobe_preparation, monkeypatch, existing, failure_stage
):
    pipeline, asset, _ = neuroprobe_preparation
    pipeline.args.reprocess = True
    output = pipeline.processed_dir / asset.path.name
    if existing:
        with h5py.File(output, "w") as handle:
            handle["previous"] = np.arange(5, dtype=np.int16)
        previous_bytes = output.read_bytes()

    if failure_stage == "save":
        save = Data.save

        def interrupted_save(self, path):
            save(self, path)
            raise RuntimeError("interrupted publication")

        monkeypatch.setattr(Data, "save", interrupted_save)
    else:
        open_file = h5py.File

        def interrupted_attributes(path, mode="r", *args, **kwargs):
            if mode == "a":
                raise RuntimeError("interrupted publication")
            return open_file(path, mode, *args, **kwargs)

        monkeypatch.setattr(h5py, "File", interrupted_attributes)

    with pytest.raises(RuntimeError, match="interrupted publication"):
        pipeline.process(asset)

    assert output.exists() is existing
    if existing:
        assert output.read_bytes() == previous_bytes
    assert not list(pipeline.processed_dir.glob("*.tmp"))


def test_get_brainset_description_records_dataset_and_neuroprobe_versions(monkeypatch):
    monkeypatch.setattr(neuroprobe_pipeline, "version", lambda package: "0.2.0")
    description = neuroprobe_pipeline.get_brainset_description()
    assert (
        description.origin_version
        == "dataset=0.0.0; neuroprobe_distribution=0.2.0; temporal_split=shared-temporal-v1"
    )
    assert description.derived_version == "1.2.0"
    assert neuroprobe_pipeline.get_brainset_description(
        splits_enabled=False
    ).origin_version.endswith("temporal_split=none")


def test_extract_channel_data_adds_btb_lip_and_xyz_coordinate_fields(
    tmp_path, monkeypatch
):
    localization_dir = tmp_path / "localization"
    localization_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Subject": "sub_1",
                "Electrode": "A*",
                "X": 11.0,
                "Y": 22.0,
                "Z": 33.0,
            }
        ]
    ).to_csv(localization_dir / "elec_coords_full.csv", index=False)

    class _FakeSubject:
        subject_id = 1
        h5_neural_data_keys = {"A": "electrode_1", "B": "electrode_2"}
        localization_data = pd.DataFrame(
            {
                "Electrode": ["A", "B"],
                "L": [1.0, 2.0],
                "I": [3.0, 4.0],
                "P": [5.0, 6.0],
            }
        )
        electrode_labels = ["A", "B"]

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))

    channels = neuroprobe_pipeline._extract_channel_data(_FakeSubject())

    assert channels.type.tolist() == [31, 31]
    np.testing.assert_allclose(channels.coord_btb_lip_l, [1.0, 2.0])
    np.testing.assert_allclose(channels.coord_btb_lip_i, [3.0, 4.0])
    np.testing.assert_allclose(channels.coord_btb_lip_p, [5.0, 6.0])
    np.testing.assert_allclose(channels.coord_btb_x[:1], [11.0])
    np.testing.assert_allclose(channels.coord_btb_y[:1], [22.0])
    np.testing.assert_allclose(channels.coord_btb_z[:1], [33.0])
    assert np.isnan(channels.coord_btb_x[1])


def test_iterate_extract_splits_prepares_and_deduplicates_subject_initialization(
    tmp_path, monkeypatch
):
    pipeline_instance = neuroprobe_pipeline.Pipeline.__new__(
        neuroprobe_pipeline.Pipeline
    )
    pipeline_instance.raw_dir = tmp_path / "raw"
    pipeline_instance.args = SimpleNamespace(reprocess=False, no_splits=False)
    pipeline_instance.update_status = lambda _status: None

    prepared_raw_dirs = []
    created_subject_ids = []
    seen_subject_keys = []

    class _FakeNeuroprobe:
        class BrainTreebankSubject:
            def __init__(self, *, subject_id, **_kwargs):
                created_subject_ids.append(subject_id)
                self.subject_id = subject_id

    def _fake_prepare(raw_dir: Path) -> None:
        prepared_raw_dirs.append(raw_dir)
        neuroprobe_pipeline.neuroprobe = _FakeNeuroprobe
        neuroprobe_pipeline.neuroprobe_config = SimpleNamespace(
            NEUROPROBE_FULL_SUBJECT_TRIALS=[(1, 0), (1, 1), (2, 0)],
            NEUROPROBE_LITE_SUBJECT_TRIALS=set(),
            NEUROPROBE_NANO_SUBJECT_TRIALS=set(),
        )

    def _fake_extract_and_structure_splits(**kwargs):
        seen_subject_keys.append(sorted(kwargs["all_subjects"].keys()))
        return {"split_key": object()}, {"split_key": np.array([True], dtype=bool)}, {}

    monkeypatch.setattr(neuroprobe_pipeline, "_prepare_neuroprobe_lib", _fake_prepare)
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "ALL_EVAL_SETTINGS",
        {
            "lite": [False],
            "nano": [False],
            "binary_tasks": [True],
            "eval_setting": ["within_session"],
        },
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_extract_channel_data",
        lambda subject: f"channels-{subject.subject_id}",
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_extract_and_structure_splits",
        _fake_extract_and_structure_splits,
    )

    split_indices, split_channel_masks, temporal = (
        neuroprobe_pipeline.Pipeline.iterate_extract_splits(
            pipeline_instance,
            subject_id=1,
            trial_id=0,
        )
    )

    assert prepared_raw_dirs == [pipeline_instance.raw_dir]
    assert created_subject_ids == [1, 2]
    assert seen_subject_keys == [[1, 2]]
    assert "split_key" in split_indices
    assert "split_key" in split_channel_masks
    assert split_channel_masks["split_key"].tolist() == [True]


def test_extract_and_structure_splits_returns_masks_without_mutating_channels(
    monkeypatch,
):
    selector_key = neuroprobe_pipeline.split_selector_key(
        lite=False,
        nano=False,
        binary_tasks=True,
        eval_setting="within_session",
        eval_name="speech",
        fold_idx=0,
        split_type="train",
    )

    channels = SimpleNamespace(name=np.array(["A", "B"], dtype=np.str_))
    train_dataset = SimpleNamespace(electrode_labels=np.array(["A"], dtype=np.str_))
    val_dataset = SimpleNamespace(electrode_labels=np.array(["B"], dtype=np.str_))
    test_dataset = SimpleNamespace(electrode_labels=np.array(["A", "B"], dtype=np.str_))
    fold = {
        "temporal_split": {
            "protocol": "shared-temporal-v1",
            "sampling_rate": 2,
            "blocks_samples": {"train": [100, 200], "val": [0, 50], "test": [50, 100]},
        },
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }

    monkeypatch.setattr(
        neuroprobe_pipeline,
        "neuroprobe_config",
        SimpleNamespace(NEUROPROBE_TASKS_MAPPING={"speech": object()}),
    )
    monkeypatch.setattr(
        neuroprobe_pipeline, "_extract_splits", lambda **_kwargs: [fold]
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_intervals_from_dataset",
        lambda dataset: f"interval-{dataset.electrode_labels[0]}",
    )

    split_indices, split_channel_masks, temporal = (
        neuroprobe_pipeline._extract_and_structure_splits(
            all_subjects={1: object()},
            all_channels={1: channels},
            subject_id=1,
            trial_id=1,
            lite=False,
            nano=False,
            binary_tasks=True,
            eval_setting="within_session",
        )
    )

    assert selector_key in split_indices
    assert selector_key in split_channel_masks
    assert split_channel_masks[selector_key].tolist() == [True, False]
    assert not hasattr(channels, selector_key)


@pytest.mark.parametrize("indices", [[4, 1, 4], []])
def test_native_interval_export_preserves_nested_subset_membership(
    monkeypatch, indices
):
    starts = np.arange(6, dtype=np.int64) * 10
    ends = starts + 4
    labels = np.arange(6, dtype=np.int64) % 2

    class NativeCandidates:
        def get_window_bounds(self):
            return starts, ends, labels

        def __getitem__(self, index):
            raise AssertionError(
                "Native interval export must not read individual items"
            )

    monkeypatch.setattr(
        neuroprobe_pipeline, "neuroprobe_config", SimpleNamespace(SAMPLING_RATE=2)
    )
    base = NativeCandidates()
    nested = Subset(Subset(base, [5, 3, 0, 2, 4]), indices)
    interval = neuroprobe_pipeline._intervals_from_dataset(nested)
    selected = np.array([5, 3, 0, 2, 4], dtype=np.int64)[
        np.asarray(indices, dtype=np.int64)
    ]
    selected = selected[np.argsort(starts[selected], kind="stable")]
    np.testing.assert_array_equal(interval.start, starts[selected] / 2)
    np.testing.assert_array_equal(interval.end, ends[selected] / 2)
    np.testing.assert_array_equal(interval.label, labels[selected])
    assert interval.start.dtype == interval.end.dtype == np.float64
    assert interval.label.dtype == np.int64


def test_xyz_coordinates_reflect_replaced_source(tmp_path):
    localization = tmp_path / "localization"
    localization.mkdir()
    path = localization / "elec_coords_full.csv"
    coordinates = pd.DataFrame(
        {"Subject": ["sub_1"], "Electrode": ["A*"], "X": [1.0], "Y": [2.0], "Z": [3.0]}
    )
    coordinates.to_csv(path, index=False)
    first = neuroprobe_pipeline._load_btb_xyz_coordinates(str(tmp_path))
    coordinates["X"] = 9.0
    coordinates.to_csv(path, index=False)
    second = neuroprobe_pipeline._load_btb_xyz_coordinates(str(tmp_path))
    assert first.loc[("sub_1", "A"), "X"] == 1.0
    assert second.loc[("sub_1", "A"), "X"] == 9.0
