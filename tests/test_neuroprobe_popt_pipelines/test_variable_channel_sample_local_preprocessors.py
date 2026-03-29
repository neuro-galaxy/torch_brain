import os
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.preprocessors.channel_subselect_preprocessor import (
    ChannelSubselectPreprocessor,
)
from neuroprobe_eval.preprocessors.laplacian_stft_preprocessor import (
    LaplacianRereferencePreprocessor,
)
from neuroprobe_eval.preprocessors.raw_preprocessor import RawPreprocessor
from neuroprobe_eval.preprocessors.stft_preprocessor import STFTPreprocessor
from neuroprobe_eval.preprocessors.time_domain_filter_preprocessor import (
    TimeDomainFilterPreprocessor,
)


def _sample(channel_ids, *, channel_names=None):
    n_channels = len(channel_ids)
    if channel_names is None:
        channel_names = list(channel_ids)
    if len(channel_names) != n_channels:
        raise ValueError("channel_names must match channel_ids length.")
    x = np.arange(n_channels * 64, dtype=np.float32).reshape(n_channels, 64)
    coords = np.arange(n_channels * 3, dtype=np.float32).reshape(n_channels, 3)
    seq_id = np.arange(n_channels, dtype=np.int64)
    brain_areas = [f"region_{i}" for i in range(n_channels)]
    return {
        "x": x,
        "y": 1,
        "channel_ids": list(channel_ids),
        "channel_names": list(channel_names),
        "channel_coords_lip": coords,
        "brain_areas": brain_areas,
        "seq_id": seq_id,
        "recording_id": "rec_0",
        "split": "train",
        "sample_idx": 0,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
    }


def test_channel_subselect_transform_sample_filters_prefixed_labels_and_metadata():
    cfg = OmegaConf.create(
        {"name": "channel_subselect", "allowed_electrodes": ["T1b2"]}
    )
    pre = ChannelSubselectPreprocessor(cfg)

    sample = _sample(
        ["id_1", "id_2", "id_3"],
        channel_names=["sub_2/sess_0/T1b1", "sub_2/sess_0/T1b2", "sub_2/sess_0/T1b3"],
    )
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape[0] == 1
    assert out["channel_ids"] == ["id_2"]
    assert out["channel_names"] == ["sub_2/sess_0/T1b2"]
    np.testing.assert_array_equal(
        out["channel_coords_lip"][0], sample["channel_coords_lip"][1]
    )
    assert int(out["seq_id"][0]) == int(sample["seq_id"][1])
    assert out["brain_areas"] == [sample["brain_areas"][1]]


def test_channel_subselect_uses_subject_prefix_for_dict_allowlist(tmp_path):
    allow_path = tmp_path / "allow.json"
    allow_path.write_text(
        json.dumps(
            {
                "sub_1": ["T1b1"],
                "sub_2": ["T2b2"],
            }
        )
    )
    cfg = OmegaConf.create(
        {
            "name": "channel_subselect",
            "allowed_electrodes": str(allow_path),
            "subject": "sub_1",
        }
    )
    pre = ChannelSubselectPreprocessor(cfg)

    sample = _sample(
        ["id_1", "id_2", "id_3"],
        channel_names=["sub_2/sess_0/T2b1", "sub_2/sess_0/T2b2", "sub_2/sess_0/T2b3"],
    )
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape[0] == 1
    assert out["channel_ids"] == ["id_2"]
    assert out["channel_names"] == ["sub_2/sess_0/T2b2"]
    assert out["brain_areas"] == [sample["brain_areas"][1]]


def test_laplacian_transform_sample_preserves_prefixed_labels():
    cfg = OmegaConf.create(
        {"name": "laplacian_rereference", "remove_non_laplacian": True}
    )
    pre = LaplacianRereferencePreprocessor(cfg)

    sample = _sample(
        ["id_1", "id_2", "id_3"],
        channel_names=["sub_2/sess_0/T1b1", "sub_2/sess_0/T1b2", "sub_2/sess_0/P1b1"],
    )
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape[0] == 2
    assert out["channel_ids"] == ["id_1", "id_2"]
    assert out["channel_names"] == ["sub_2/sess_0/T1b1", "sub_2/sess_0/T1b2"]
    assert out["channel_coords_lip"].shape == (2, 3)
    assert out["seq_id"].shape == (2,)
    assert out["brain_areas"] == [sample["brain_areas"][0], sample["brain_areas"][1]]


def test_stft_transform_sample_updates_feature_shape_and_keeps_channel_alignment():
    cfg = OmegaConf.create(
        {
            "name": "stft",
            "nperseg": 32,
            "poverlap": 0.5,
            "sampling_rate": 2048,
            "max_frequency": 150,
            "min_frequency": 0,
            "normalizing": "none",
            "clip_k": 0,
            "use_scipy": False,
        }
    )
    pre = STFTPreprocessor(cfg)

    sample = _sample(["A1", "A2"])
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape[0] == 2
    assert out["x"].ndim == 3  # (channels, time_bins, freq_bins)
    assert out["channel_ids"] == ["A1", "A2"]
    assert out["channel_names"] == ["A1", "A2"]
    assert out["channel_coords_lip"].shape == (2, 3)
    assert out["seq_id"].shape == (2,)
    assert out["brain_areas"] == sample["brain_areas"]


def test_time_domain_filter_transform_sample_preserves_shape():
    cfg = OmegaConf.create(
        {
            "name": "time_domain_filter",
            "sampling_rate": 2048,
            "high_gamma": False,
        }
    )
    pre = TimeDomainFilterPreprocessor(cfg)
    sample = _sample(["B1", "B2"])
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape == sample["x"].shape
    assert out["channel_ids"] == sample["channel_ids"]
    assert out["channel_names"] == sample["channel_names"]
    assert out["channel_coords_lip"].shape == sample["channel_coords_lip"].shape
    assert out["brain_areas"] == sample["brain_areas"]


def test_raw_transform_sample_preserves_channel_alignment():
    cfg = OmegaConf.create({"name": "raw"})
    pre = RawPreprocessor(cfg)
    sample = _sample(["R1", "R2"])
    out = pre.transform_samples([sample])[0]

    assert out["x"].shape == sample["x"].shape
    assert out["channel_ids"] == sample["channel_ids"]
    assert out["channel_names"] == sample["channel_names"]
    np.testing.assert_array_equal(
        out["channel_coords_lip"], sample["channel_coords_lip"]
    )
    np.testing.assert_array_equal(out["seq_id"], sample["seq_id"])
    assert out["brain_areas"] == sample["brain_areas"]


def test_raw_transform_samples_matches_transform_sample_output():
    cfg = OmegaConf.create({"name": "raw"})
    pre = RawPreprocessor(cfg)
    sample = _sample(["R1", "R2"])

    out_single = pre.transform_samples([sample])[0]
    out_batch = pre.transform_samples([sample])[0]

    np.testing.assert_allclose(out_batch["x"], out_single["x"], atol=1e-6)
    assert out_batch["channel_ids"] == out_single["channel_ids"]
    assert out_batch["channel_names"] == out_single["channel_names"]


def test_time_domain_filter_transform_samples_matches_transform_sample_output():
    cfg = OmegaConf.create(
        {
            "name": "time_domain_filter",
            "sampling_rate": 2048,
            "high_gamma": False,
        }
    )
    pre = TimeDomainFilterPreprocessor(cfg)
    sample = _sample(["B1", "B2"])

    out_single = pre.transform_samples([sample])[0]
    out_batch = pre.transform_samples([sample])[0]

    np.testing.assert_allclose(out_batch["x"], out_single["x"], atol=1e-6)
    assert out_batch["channel_ids"] == out_single["channel_ids"]


def test_stft_transform_samples_matches_transform_sample_output():
    cfg = OmegaConf.create(
        {
            "name": "stft",
            "nperseg": 32,
            "poverlap": 0.5,
            "sampling_rate": 2048,
            "max_frequency": 150,
            "min_frequency": 0,
            "normalizing": "none",
            "clip_k": 0,
            "use_scipy": False,
        }
    )
    pre = STFTPreprocessor(cfg)
    sample = _sample(["A1", "A2"])

    out_single = pre.transform_samples([sample])[0]
    out_batch = pre.transform_samples([sample])[0]

    np.testing.assert_allclose(out_batch["x"], out_single["x"], atol=1e-6)
    assert out_batch["x"].shape[0] == len(sample["channel_ids"])
    assert out_batch["channel_ids"] == out_single["channel_ids"]


def test_channel_subselect_transform_samples_matches_transform_sample_and_metadata():
    cfg = OmegaConf.create(
        {"name": "channel_subselect", "allowed_electrodes": ["T1b2"]}
    )
    pre = ChannelSubselectPreprocessor(cfg)
    sample = _sample(
        ["id_1", "id_2", "id_3"],
        channel_names=["sub_2/sess_0/T1b1", "sub_2/sess_0/T1b2", "sub_2/sess_0/T1b3"],
    )

    out_single = pre.transform_samples([sample])[0]
    out_batch = pre.transform_samples([sample])[0]

    np.testing.assert_allclose(out_batch["x"], out_single["x"], atol=1e-6)
    assert out_batch["channel_names"] == out_single["channel_names"]
    idx = [sample["channel_names"].index(name) for name in out_single["channel_names"]]
    assert out_batch["channel_ids"] == [sample["channel_ids"][i] for i in idx]
    np.testing.assert_array_equal(
        out_batch["channel_coords_lip"], sample["channel_coords_lip"][idx]
    )
    np.testing.assert_array_equal(out_batch["seq_id"], sample["seq_id"][idx])
    assert out_batch["brain_areas"] == [sample["brain_areas"][i] for i in idx]


def test_laplacian_transform_samples_matches_transform_sample_and_metadata():
    cfg = OmegaConf.create(
        {"name": "laplacian_rereference", "remove_non_laplacian": True}
    )
    pre = LaplacianRereferencePreprocessor(cfg)
    sample = _sample(
        ["id_1", "id_2", "id_3"],
        channel_names=["sub_2/sess_0/T1b1", "sub_2/sess_0/T1b2", "sub_2/sess_0/P1b1"],
    )

    out_single = pre.transform_samples([sample])[0]
    out_batch = pre.transform_samples([sample])[0]

    np.testing.assert_allclose(out_batch["x"], out_single["x"], atol=1e-6)
    assert out_batch["channel_names"] == out_single["channel_names"]
    idx = [sample["channel_names"].index(name) for name in out_single["channel_names"]]
    assert out_batch["channel_ids"] == [sample["channel_ids"][i] for i in idx]
    np.testing.assert_array_equal(
        out_batch["channel_coords_lip"], sample["channel_coords_lip"][idx]
    )
    np.testing.assert_array_equal(out_batch["seq_id"], sample["seq_id"][idx])
    assert out_batch["brain_areas"] == [sample["brain_areas"][i] for i in idx]


def test_transform_samples_preserves_input_order_for_multiple_samples():
    cfg = OmegaConf.create({"name": "raw"})
    pre = RawPreprocessor(cfg)
    s0 = _sample(["R1", "R2"])
    s1 = _sample(["R1", "R2"])
    s1["x"] = s1["x"] + 1000.0

    out = pre.transform_samples([s0, s1])

    assert len(out) == 2
    np.testing.assert_allclose(out[0]["x"], s0["x"], atol=1e-6)
    np.testing.assert_allclose(out[1]["x"], s1["x"], atol=1e-6)


def test_channel_subselect_transform_sample_requires_channel_names():
    cfg = OmegaConf.create(
        {"name": "channel_subselect", "allowed_electrodes": ["T1b1"]}
    )
    pre = ChannelSubselectPreprocessor(cfg)
    sample = _sample(["id_1", "id_2"])
    sample["channel_names"] = None

    with pytest.raises(ValueError, match="channel_names"):
        pre.transform_samples([sample])[0]


def test_laplacian_transform_sample_requires_channel_names():
    cfg = OmegaConf.create({"name": "laplacian_rereference"})
    pre = LaplacianRereferencePreprocessor(cfg)
    sample = _sample(["id_1", "id_2"])
    sample["channel_names"] = None

    with pytest.raises(ValueError, match="channel_names"):
        pre.transform_samples([sample])[0]
