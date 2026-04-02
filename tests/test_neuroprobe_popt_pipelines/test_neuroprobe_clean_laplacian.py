import json
import os
from types import SimpleNamespace

import torch

os.environ["ROOT_DIR_BRAINTREEBANK"] = "/tmp"

from neuroprobe_eval.preprocessors.channel_subselect_preprocessor import (
    ChannelSubselectPreprocessor,
)
from neuroprobe_eval.preprocessors.laplacian_stft_preprocessor import (
    LaplacianRereferencePreprocessor,
    LaplacianSTFTPreprocessor,
    laplacian_rereference_neural_data,
)


def test_laplacian_stft_preprocessor_keeps_channels_by_default():
    cfg = SimpleNamespace(
        name="laplacian_stft",
        get=lambda key, default=None: default,
    )
    preprocessor = LaplacianSTFTPreprocessor(cfg)

    sample = {
        "x": torch.randn(3, 512).numpy(),
        "channel_ids": ["id1", "id2", "id3"],
        "channel_names": ["T1b1", "T1b2", "T1b3"],
        "channel_coords_lip": torch.randn(3, 3).numpy(),
        "seq_id": torch.zeros((3,), dtype=torch.int64).numpy(),
        "brain_areas": ["r1", "r2", "r3"],
        "split": "train",
    }
    out = preprocessor.transform_samples([sample])[0]

    assert out["x"].shape[0] == 3
    assert out["channel_names"] == sample["channel_names"]


def test_laplacian_rereference_preprocessor_can_drop_non_laplacian_channels():
    cfg = SimpleNamespace(
        name="laplacian_rereference",
        get=lambda key, default=None: {"remove_non_laplacian": True}.get(key, default),
    )
    preprocessor = LaplacianRereferencePreprocessor(cfg)

    sample = {
        "x": torch.randn(3, 32).numpy(),
        "channel_ids": ["id1", "id2", "id3"],
        "channel_names": ["T1b1", "T1b2", "P1b1"],
        "channel_coords_lip": torch.randn(3, 3).numpy(),
        "seq_id": torch.zeros((3,), dtype=torch.int64).numpy(),
        "brain_areas": ["r1", "r2", "r3"],
        "split": "train",
    }
    out = preprocessor.transform_samples([sample])[0]

    assert out["x"].shape[0] == 2
    assert out["channel_names"] == ["T1b1", "T1b2"]


def test_channel_subselect_preprocessor_filters_channels():
    cfg = SimpleNamespace(
        name="channel_subselect",
        get=lambda key, default=None: {"allowed_electrodes": ["T1b2"]}.get(
            key, default
        ),
    )
    preprocessor = ChannelSubselectPreprocessor(cfg)

    sample = {
        "x": torch.randn(3, 10).numpy(),
        "channel_ids": ["id1", "id2", "id3"],
        "channel_names": ["T1b1", "T1b2", "T1b3"],
        "channel_coords_lip": torch.randn(3, 3).numpy(),
        "seq_id": torch.zeros((3,), dtype=torch.int64).numpy(),
        "brain_areas": ["r1", "r2", "r3"],
        "split": "train",
    }
    out = preprocessor.transform_samples([sample])[0]

    assert out["x"].shape[0] == 1
    assert out["channel_names"] == ["T1b2"]


def test_channel_subselect_preprocessor_accepts_json_path(tmp_path):
    json_path = tmp_path / "allowed.json"
    json_path.write_text(json.dumps(["T1b2"]))

    cfg = SimpleNamespace(
        name="channel_subselect",
        get=lambda key, default=None: {"allowed_electrodes": str(json_path)}.get(
            key, default
        ),
    )
    preprocessor = ChannelSubselectPreprocessor(cfg)

    sample = {
        "x": torch.randn(3, 10).numpy(),
        "channel_ids": ["id1", "id2", "id3"],
        "channel_names": ["T1b1", "T1b2", "T1b3"],
        "channel_coords_lip": torch.randn(3, 3).numpy(),
        "seq_id": torch.zeros((3,), dtype=torch.int64).numpy(),
        "brain_areas": ["r1", "r2", "r3"],
        "split": "train",
    }
    out = preprocessor.transform_samples([sample])[0]

    assert out["x"].shape[0] == 1
    assert out["channel_names"] == ["T1b2"]


def test_laplacian_rereference_torch_input_matches_expected_neighbor_mean():
    labels = ["T1b1", "T1b2", "T1b3"]
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]]],
        dtype=torch.float32,
    )

    out_x, out_labels, out_indices = laplacian_rereference_neural_data(
        x,
        labels,
        remove_non_laplacian=False,
    )

    expected = torch.tensor(
        [[[-2.0, -2.0], [-1.0, -1.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(out_x, expected)
    assert out_labels == labels
    assert out_indices == [0, 1, 2]
