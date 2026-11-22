"""Resume checks run through native entrypoints without reading raw waveforms."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from _utils import add_pipelines_to_path

from torch_brain.data import ArrayDict, Data, Interval, RegularTimeSeries
from torch_brain.pipeline.temporal_preparation import (
    MOVIE_DERIVED_VERSION,
    preparation_settings,
    validate_prepared_temporal_settings,
)
from torch_brain.utils.split import generate_temporal_folds, temporal_metadata

add_pipelines_to_path()
from berezutskaya_pippi_2022 import pipeline as pippi  # noqa: E402
from keles_byd_2024 import pipeline as byd  # noqa: E402
from neuroprobe_2025 import pipeline as neuroprobe  # noqa: E402


@pytest.fixture
def prepared_movie(tmp_path):
    annotation = tmp_path / "speech_binary_labels.csv"
    annotation.write_text("start_time_s,label\n0.5,0\n1.5,1\n")
    settings = preparation_settings(
        splits_enabled=True,
        pre_offset_s=0.0,
        post_offset_s=1.0,
        label_paths={annotation.name: annotation},
    )
    return tmp_path, annotation, settings


def write_prepared(path, settings, *, mask_domain=False):
    domain = Interval(start=[0.0], end=[40.0])
    metadata = temporal_metadata([0, 40])
    metadata.update(balance_seed=7, balanced=True)
    starts = np.arange(0.5, 39.0, 0.5)
    folds = generate_temporal_folds(
        starts,
        starts + 1,
        np.arange(len(starts)) % 2,
        domain=[0, 40],
        balance=True,
        rng=np.random.default_rng(7),
    )
    splits = Data(domain=domain)
    masks = Data(domain=domain) if mask_domain else Data()
    for fold, parts in folds.items():
        for split, interval in parts.items():
            key = f"full$binary$within_session$speech$fold{fold}${split}"
            setattr(splits, key, interval)
            setattr(masks, key, np.array([True, True]))
    data = Data(
        brainset=Data(derived_version=MOVIE_DERIVED_VERSION),
        subject=Data(id="subject"),
        session=Data(id="session"),
        domain=domain,
        channels=ArrayDict(id=np.array(["a", "b"], dtype="S1")),
        seeg_data=RegularTimeSeries(
            data=np.arange(160, dtype=np.float32).reshape(80, 2), sampling_rate=2
        ),
        splits=splits,
        channel_splits=masks,
        preparation_settings_json=json.dumps(settings),
        temporal_splits_json=json.dumps(metadata),
    )
    with h5py.File(path, "w") as handle:
        data.to_hdf5(handle)
        handle["seeg_data"].attrs.update(unit="V", scale_to_uV=1e6)


def validate(path, **changes):
    expected = dict(
        context_width_s=1.0,
        gap_s=0.0,
        balance_seed=7,
        balanced=True,
        splits_enabled=True,
    )
    expected.update(changes)
    validate_prepared_temporal_settings(path, **expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_width_s", 15),
        ("gap_s", 2),
        ("balance_seed", 9),
        ("balanced", False),
        ("splits_enabled", False),
    ],
)
def test_resume_rejects_requested_temporal_policy_changes(prepared_movie, field, value):
    root, annotation, settings = prepared_movie
    path = root / "movie.h5"
    write_prepared(path, settings)
    validate(path, preparation_settings=settings)
    with pytest.raises(ValueError, match="--reprocess"):
        validate(path, **{field: value}, preparation_settings=settings)


@pytest.mark.parametrize(
    "corruption", ["old_version", "missing_provenance", "bad_settings", "missing_seed"]
)
def test_resume_rejects_unattributed_or_malformed_outputs(prepared_movie, corruption):
    root, annotation, settings = prepared_movie
    path = root / "movie.h5"
    write_prepared(path, settings)
    with h5py.File(path, "r+") as handle:
        if corruption == "old_version":
            handle["brainset"].attrs["derived_version"] = "1.1.1"
        elif corruption == "missing_provenance":
            del handle.attrs["preparation_settings_json"]
        elif corruption == "bad_settings":
            handle.attrs["preparation_settings_json"] = json.dumps(
                dict(settings, label_files=[])
            )
        else:
            metadata = json.loads(handle.attrs["temporal_splits_json"])
            del metadata["balance_seed"]
            handle.attrs["temporal_splits_json"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="--reprocess"):
        validate(path)


def test_resume_compares_offsets_and_annotation_content(prepared_movie):
    root, annotation, settings = prepared_movie
    path = root / "movie.h5"
    write_prepared(path, settings)
    annotation.write_text("start_time_s,label\n0.5,1\n1.5,0\n")
    changed = preparation_settings(
        splits_enabled=True,
        pre_offset_s=0.0,
        post_offset_s=1.0,
        label_paths={annotation.name: annotation},
    )
    with pytest.raises(ValueError, match="source annotations differ"):
        validate(path, preparation_settings=changed)
    with pytest.raises(ValueError, match="target offsets"):
        validate(
            path,
            preparation_settings=dict(settings, pre_offset_s=0.25, post_offset_s=0.75),
        )


@pytest.mark.parametrize(
    "dataset,changed_seed,corruption",
    [
        *[
            (dataset, changed_seed, None)
            for dataset in ("byd", "pippi", "neuroprobe")
            for changed_seed in (False, True)
        ],
        ("pippi", False, "missing_signal"),
        ("pippi", False, "empty_splits"),
        ("neuroprobe", False, "missing_signal"),
        ("neuroprobe", False, "empty_splits"),
    ],
)
def test_all_entrypoints_check_resume_before_raw_reads(
    prepared_movie, monkeypatch, dataset, changed_seed, corruption
):
    root, annotation, settings = prepared_movie
    recording_id = {
        "byd": "sub-CS41_ses-P41CSR1",
        "pippi": "sub-01_ses-iemu_task-film_acq-clinical_run-1",
        "neuroprobe": "sub_1_trial001",
    }[dataset]
    output = root / f"{recording_id}.h5"
    write_prepared(output, settings)
    if corruption:
        with h5py.File(output, "r+") as handle:
            if corruption == "missing_signal":
                del handle["seeg_data/data"]
            else:
                for group in ("splits", "channel_splits"):
                    del handle[group]
                    handle.create_group(group)
    before = output.read_bytes()
    seed = 9 if changed_seed else 7
    common = dict(
        labels_dir=root,
        label_files=[annotation.name],
        pre_offset_s=0.0,
        post_offset_s=1.0,
        balance_splits=True,
        balance_seed=seed,
    )
    if dataset == "byd":

        def execute():
            return byd.process_file(
                str(root / f"{recording_id}_behavior+ecephys.nwb"), str(root), **common
            )
    elif dataset == "pippi":

        def execute():
            return pippi.process_file(
                root / f"{recording_id}_ieeg.vhdr", root, **common
            )
    else:
        monkeypatch.setattr(neuroprobe, "_prepare_neuroprobe_lib", lambda _: None)
        monkeypatch.setattr(neuroprobe, "_validate_native_neuroprobe_api", lambda: None)
        monkeypatch.setattr(
            neuroprobe,
            "_neuroprobe_label_paths",
            lambda *_: {annotation.name: annotation},
        )
        instance = neuroprobe.Pipeline(
            root, root, neuroprobe.parser.parse_args(["--balance-seed", str(seed)])
        )
        instance.update_status = lambda _: None

        def execute():
            return instance.process(
                neuroprobe.DownloadedAsset(root / output.name, 1, 1)
            )

    if changed_seed:
        with pytest.raises(ValueError, match="balance_seed"):
            execute()
    elif corruption:
        with pytest.raises(ValueError, match="--reprocess"):
            execute()
    else:
        execute()
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "pipeline,filename",
    [
        (byd, "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb"),
        (pippi, "sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr"),
    ],
)
def test_direct_preparation_validates_config_before_missing_raw_inputs(
    tmp_path, pipeline, filename
):
    with pytest.raises(ValueError, match="gap_s"):
        pipeline.process_file(
            tmp_path / filename,
            tmp_path,
            labels_dir=tmp_path,
            label_files=(),
            pre_offset_s=0.0,
            post_offset_s=1.0,
            temporal_gap_s=-1,
        )


def test_neuroprobe_rejects_released_api_before_annotations(monkeypatch):
    def released_splitter(*args, **kwargs):
        raise AssertionError("Must not call the splitter during capability validation")

    monkeypatch.setattr(
        neuroprobe, "neuroprobe", SimpleNamespace(temporal_splits=None), raising=False
    )
    monkeypatch.setattr(
        neuroprobe,
        "neuroprobe_train_test_splits",
        SimpleNamespace(generate_splits_within_session=released_splitter),
        raising=False,
    )
    with pytest.raises(ValueError, match="--use-active-env"):
        neuroprobe._validate_native_neuroprobe_api()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_signal",
        "empty_signal",
        "wrong_channel_count",
        "empty_splits",
        "missing_mask",
        "incomplete_fold",
        "missing_fold",
        "bad_annotation_shapes",
        "bad_mask_length",
    ],
)
def test_resume_rejects_partial_recording_structure(prepared_movie, corruption):
    root, annotation, settings = prepared_movie
    path = root / "movie.h5"
    write_prepared(path, settings)
    key = "full$binary$within_session$speech$fold0$train"
    with h5py.File(path, "r+") as handle:
        if corruption in {"missing_signal", "empty_signal"}:
            del handle["seeg_data/data"]
            if corruption == "empty_signal":
                handle["seeg_data/data"] = np.empty((0, 2), dtype=np.float32)
        elif corruption == "wrong_channel_count":
            del handle["channels/id"]
            handle["channels/id"] = np.array(["a"], dtype="S1")
        elif corruption == "empty_splits":
            del handle["splits"]
            del handle["channel_splits"]
            handle.create_group("splits")
            handle.create_group("channel_splits")
        elif corruption == "missing_mask":
            del handle["channel_splits"][key]
        elif corruption == "incomplete_fold":
            del handle["splits"][key]
            del handle["channel_splits"][key]
        elif corruption == "missing_fold":
            for selector in list(handle["splits"]):
                if "$fold1$" in selector:
                    del handle["splits"][selector]
                    del handle["channel_splits"][selector]
        elif corruption == "bad_annotation_shapes":
            del handle["splits"][key]["label"]
            handle["splits"][key]["label"] = np.array([0])
        else:
            del handle["channel_splits"][key]
            handle["channel_splits"][key] = np.array([True])
    with pytest.raises(ValueError, match="--reprocess"):
        validate(path)


def test_resume_validation_never_reads_waveform_arrays(prepared_movie, monkeypatch):
    root, annotation, settings = prepared_movie
    path = root / "movie.h5"
    write_prepared(path, settings)
    original = h5py.Dataset.__getitem__

    def read_annotation_only(array, index):
        if array.name == "/seeg_data/data":
            raise AssertionError("Resume must not load waveform arrays")
        return original(array, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read_annotation_only)
    validate(path)


@pytest.mark.parametrize("mask_domain", [False, True])
def test_resume_distinguishes_native_data_domain_from_selectors(
    prepared_movie, mask_domain
):
    root, annotation, settings = prepared_movie
    path = root / "native_movie.h5"
    write_prepared(path, settings, mask_domain=mask_domain)
    with h5py.File(path) as handle:
        assert "domain" in handle["splits"]
        assert ("domain" in handle["channel_splits"]) == mask_domain
    validate(path)
    with h5py.File(path, "r+") as handle:
        handle["splits"].create_group("unexpected")
        handle["channel_splits"]["unexpected"] = np.array([True, True])
    with pytest.raises(ValueError, match="invalid split selector"):
        validate(path)
