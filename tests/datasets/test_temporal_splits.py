import gc
import json

import h5py
import numpy as np
import pytest

import torch_brain.datasets.mixins as mixin_module
from torch_brain.data import ArrayDict, Data, Interval, RegularTimeSeries
from torch_brain.datasets import BerezutskayaPippi2022, KelesBYD2024, NeuroprobeV2
from torch_brain.datasets.mixins import (
    TemporalBlockDatasetMixin,
    recording_fingerprint,
)
from torch_brain.utils.split import generate_temporal_folds, temporal_metadata


@pytest.fixture(
    params=[
        (NeuroprobeV2, "neuroprobe_2025", "sub_1_trial001", 1, 1),
        (KelesBYD2024, "keles_byd_2024", "sub-CS41_ses-P41CSR1", 41, 1),
        (
            BerezutskayaPippi2022,
            "berezutskaya_pippi_2022",
            "sub-01_ses-iemu_task-film_acq-clinical_run-1",
            1,
            1,
        ),
    ],
    ids=["neuroprobe", "byd", "pippi"],
)
def native_recording(tmp_path, request):
    """Small real H5 recording shared by provider and immutable-selection checks."""
    provider, dirname, rid, subject, session = request.param
    domain = Interval(start=[0.0], end=[100.0])
    starts = np.arange(0.1, 99, 0.5)
    labels = np.arange(len(starts)) % 2
    metadata = temporal_metadata([0, 100], context_width_s=9, gap_s=1)
    folds = generate_temporal_folds(
        starts,
        starts + 1,
        labels,
        domain=[0, 100],
        context_width_s=9,
        gap_s=1,
        balance=True,
        rng=np.random.default_rng(42),
    )
    splits, masks = Data(domain=domain), Data()
    for task in ("speech", "volume"):
        for fold in (0, 1):
            for split, interval in folds[fold].items():
                key = f"full$binary$within_session${task}$fold{fold}${split}"
                setattr(splits, key, interval)
                setattr(masks, key, np.ones(2, dtype=bool))
    data = Data(
        domain=domain,
        brainset=Data(id=dirname, derived_version="1.2.0"),
        session=Data(id=rid),
        subject=Data(id=str(subject)),
        seeg_data=RegularTimeSeries(
            sampling_rate=100.0, data=np.zeros((10000, 2), dtype=np.float32)
        ),
        channels=ArrayDict(id=np.array(["a", "b"]), name=np.array(["a", "b"])),
        splits=splits,
        channel_splits=masks,
        temporal_splits_json=json.dumps(metadata),
    )
    path = tmp_path / dirname / f"{rid}.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        data.to_hdf5(handle)
        handle["seeg_data"].attrs.update(unit="uV", scale_to_uV=1.0)
        assert "signal_identity" not in handle.attrs
    return provider, path, rid, subject, session, folds, metadata


def test_real_provider_roundtrip(native_recording):
    provider, path, rid, subject, session, folds, metadata = native_recording
    for fold in (0, 1):
        for split in ("train", "val", "test"):
            ds = provider(
                root=str(path.parent.parent),
                subset_tier="full",
                test_subject=subject,
                test_session=session,
                task="speech",
                split=split,
                fold=fold,
            )
            selected = ds.get_sampling_intervals()[rid]
            np.testing.assert_array_equal(selected.start, folds[fold][split].start)
            summary = ds.describe_selection()
            assert summary["temporal_protocol"] == "shared-temporal-v1"
            assert summary["temporal_recording_fingerprints"] == {
                rid: recording_fingerprint(path)
            }
            block = ds.get_temporal_block(rid)
            assert "signal_identity" not in block
            assert block["recording_path"] == str(path)
            assert block["recording_fingerprint"] == recording_fingerprint(path)
            assert block["bounds"] == list(metadata["blocks"][str(fold)][split])
    key = "full$binary$within_session$speech$fold0$train"
    first = TemporalBlockDatasetMixin.read_temporal_selection(path, key)
    del ds, selected
    gc.collect()
    # Older files may carry stale hashes; neither hashes nor signal values define selection.
    with h5py.File(path, "r+") as handle:
        handle.attrs["signal_identity"] = "legacy-signal-hash"
        handle["seeg_data/data"][0, 0] = 123.0
    legacy = TemporalBlockDatasetMixin.read_temporal_selection(path, key)
    assert legacy["selection_digest"] == first["selection_digest"]
    assert legacy["recording_fingerprint"] != first["recording_fingerprint"]
    assert "signal_identity" not in legacy
    with h5py.File(path, "r+") as handle:
        handle["splits"][key]["label"][0] = 1 - handle["splits"][key]["label"][0]
    second = TemporalBlockDatasetMixin.read_temporal_selection(path, key)
    assert first["selection_digest"] != second["selection_digest"]
    with h5py.File(path, "r+") as handle:
        handle["splits"][key]["start"][0] = 0
    with pytest.raises(ValueError, match="cross temporal"):
        TemporalBlockDatasetMixin.read_temporal_selection(path, key)


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("first_read", ["describe", "block"])
def test_provider_reuses_immutable_temporal_selection(
    native_recording, monkeypatch, historical, first_read
):
    provider, path, rid, subject, session, _, _ = native_recording
    if historical:
        with h5py.File(path, "r+") as handle:
            del handle.attrs["temporal_splits_json"]
            handle["brainset"].attrs["derived_version"] = "1.1.1"
    ds = provider(
        root=str(path.parent.parent),
        subset_tier="full",
        test_subject=subject,
        test_session=session,
        task="speech",
        split="train",
        fold=0,
        keep_files_open=False,
    )
    key = "full$binary$within_session$speech$fold0$train"
    expected = TemporalBlockDatasetMixin.read_temporal_selection(path, key)
    opens = []
    original = h5py.File

    def counted(*args, **kwargs):
        opens.append(args[0])
        return original(*args, **kwargs)

    with monkeypatch.context() as capture:
        capture.setattr(mixin_module.h5py, "File", counted)
        if first_read == "block":
            if historical:
                with pytest.raises(
                    ValueError, match="Native temporal splits are missing"
                ):
                    ds.get_temporal_block(rid)
            else:
                assert ds.get_temporal_block(rid) == expected
        first = ds.describe_selection()
        assert ds.describe_selection() == first
        if historical:
            assert "temporal_protocol" not in first
            with pytest.raises(ValueError, match="Native temporal splits are missing"):
                ds.get_temporal_block(rid)
        else:
            block = ds.get_temporal_block(rid)
            assert block == expected
            # Returned mappings remain independent of the captured selection.
            block["bounds"][0] = -999
            block["recording_fingerprint"]["size"] = -1
            first["temporal_recording_fingerprints"][rid]["size"] = -2
            assert ds.get_temporal_block(rid) == expected
            assert ds.describe_selection()["temporal_recording_fingerprints"] == {
                rid: recording_fingerprint(path)
            }
        assert opens == [path]

    with h5py.File(path, "r+") as handle:
        handle.attrs["revision"] = 1
    with pytest.raises(RuntimeError, match="changed after capturing temporal metadata"):
        ds.describe_selection()
    with pytest.raises(RuntimeError, match="changed after capturing temporal metadata"):
        ds.get_temporal_block(rid)


def test_cached_temporal_selection_rejects_deleted_source(native_recording):
    _, path, _, _, _, _, _ = native_recording
    cache = {}
    key = "full$binary$within_session$speech$fold0$train"
    TemporalBlockDatasetMixin.read_temporal_selection(path, key, cache=cache)
    path.unlink()
    with pytest.raises(RuntimeError, match="changed after capturing temporal metadata"):
        TemporalBlockDatasetMixin.read_temporal_selection(path, key, cache=cache)


def test_explicit_recording_mode_preserves_channel_hooks_without_temporal_selection(
    native_recording,
):
    provider, path, rid, subject, _, _, _ = native_recording
    ds = provider(
        root=str(path.parent.parent),
        recording_ids=[rid],
        keep_files_open=False,
    )
    assert isinstance(ds, TemporalBlockDatasetMixin)
    assert ds.get_channel_ids() == [f"{subject}/a", f"{subject}/b"]
    assert ds.describe_temporal_selection() == {}
    assert "temporal_protocol" not in ds.describe_selection()
    with pytest.raises(ValueError, match="active split-selected recording"):
        ds.get_temporal_block(rid)


def test_canonical_base_writer_format_retains_sampling_without_native_metadata(
    native_recording,
):
    provider, path, rid, subject, session, folds, _ = native_recording
    with h5py.File(path, "r+") as handle:
        del handle.attrs["temporal_splits_json"]
        handle["brainset"].attrs["derived_version"] = "1.1.1"
    ds = provider(
        root=str(path.parent.parent),
        subset_tier="full",
        test_subject=subject,
        test_session=session,
        task="speech",
        split="train",
        fold=0,
        keep_files_open=False,
    )
    assert ds.recording_ids == [rid]
    selected = ds.get_sampling_intervals()[rid]
    np.testing.assert_array_equal(selected.start, folds[0]["train"].start)
    np.testing.assert_array_equal(selected.label, folds[0]["train"].label)
    summary = ds.describe_selection()
    assert summary["split_key"] == "full$binary$within_session$speech$fold0$train"
    assert "temporal_protocol" not in summary
    with pytest.raises(ValueError, match="Native temporal splits are missing"):
        ds.get_temporal_block(rid)


@pytest.mark.parametrize(
    "native_recording",
    [(KelesBYD2024, "keles_byd_2024", "sub-CS41_ses-P41CSR1", 41, 1)],
    indirect=True,
    ids=["byd"],
)
@pytest.mark.parametrize(
    "task,retired_task",
    [("speech", "speech"), ("onset", "sentence_onset"), ("word_head_pos", "word_head")],
)
def test_byd_rejects_retired_root_selectors_on_canonical_file(
    native_recording, task, retired_task
):
    provider, path, _, subject, session, _, _ = native_recording
    with h5py.File(path, "r+") as handle:
        del handle.attrs["temporal_splits_json"]
        handle["brainset"].attrs["derived_version"] = "1.1.1"
        for fold in (0, 1):
            for split in ("train", "val", "test"):
                selector = f"full$binary$within_session$speech$fold{fold}${split}"
                retired = f"binary_{retired_task}_fold{fold}_{split}"
                handle.move(f"splits/{selector}", retired)
                handle.move(
                    f"channel_splits/{selector}", f"channels/included_{retired}"
                )
        del handle["splits"]
        del handle["channel_splits"]
    ds = provider(
        root=str(path.parent.parent),
        subset_tier="full",
        test_subject=subject,
        test_session=session,
        task=task,
        split="train",
        fold=0,
        keep_files_open=False,
    )
    with pytest.raises(
        KeyError, match="Missing required benchmark selector attributes"
    ):
        ds.get_sampling_intervals()


@pytest.mark.parametrize(
    "native_recording",
    [(KelesBYD2024, "keles_byd_2024", "sub-CS41_ses-P41CSR1", 41, 1)],
    indirect=True,
    ids=["byd"],
)
@pytest.mark.parametrize("corruption", ["reversed_window", "label_length"])
def test_native_reader_rejects_malformed_window_arrays(native_recording, corruption):
    _, path, _, _, _, _, _ = native_recording
    key = "full$binary$within_session$speech$fold0$val"
    with h5py.File(path, "r+") as handle:
        group = handle["splits"][key]
        if corruption == "reversed_window":
            # Keep the malformed target inside its block, so containment alone passes.
            index = len(group["start"]) // 2
            group["end"][index] = group["start"][index] - 1
        else:
            labels = group["label"][:-1]
            del group["label"]
            group["label"] = labels
    with pytest.raises(ValueError, match="finite positive-duration windows"):
        TemporalBlockDatasetMixin.read_temporal_selection(path, key)
