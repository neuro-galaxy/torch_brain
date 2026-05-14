"""Variable-channel collate and loader helpers for Neuroprobe torch evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from neuroprobe_eval.utils.data_adapter import (
    validate_sample_dict,
)


def variable_channel_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-channel samples into one batch dict."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    split = samples[0]["split"]
    feature_shape = tuple(np.asarray(samples[0]["x"]).shape[1:])
    if not feature_shape:
        raise ValueError(
            "sample['x'] must include at least one non-channel feature dimension."
        )

    for sample in samples:
        # Batch-level collate accepts coord-optional samples; model-specific
        # requirements are enforced later in model prepare/forward code.
        validate_sample_dict(sample, expected_split=split, require_coords=False)
        sample_feature_shape = tuple(np.asarray(sample["x"]).shape[1:])
        if sample_feature_shape != feature_shape:
            raise ValueError(
                "All samples in a batch must have the same per-channel feature shape, "
                f"got {feature_shape} and {sample_feature_shape}."
            )

    batch_size = len(samples)
    n_channels = [np.asarray(sample["x"]).shape[0] for sample in samples]
    max_channels = max(n_channels)
    # Batch tensors are padded to max_channels; `channel_mask` preserves the
    # real channel extents per sample.

    x = torch.zeros((batch_size, max_channels, *feature_shape), dtype=torch.float32)
    y = torch.zeros((batch_size,), dtype=torch.long)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    n_channels_tensor = torch.as_tensor(n_channels, dtype=torch.long)

    coords_present = [sample["channel_coords_lip"] is not None for sample in samples]
    # Keep coords representation uniform within a batch to avoid ambiguous
    # downstream tensor contracts.
    if any(coords_present) and not all(coords_present):
        raise ValueError(
            "channel_coords_lip must be either present for all samples in a batch or "
            "None for all samples."
        )
    coords_tensor = (
        torch.zeros((batch_size, max_channels, 3), dtype=torch.float32)
        if all(coords_present)
        else None
    )

    seq_present = [sample["seq_id"] is not None for sample in samples]
    if any(seq_present) and not all(seq_present):
        raise ValueError(
            "seq_id must be either present for all samples in a batch or None for "
            "all samples."
        )
    seq_tensor = (
        torch.zeros((batch_size, max_channels), dtype=torch.long)
        if all(seq_present)
        else None
    )

    channel_ids: list[list[str]] = []
    recording_ids: list[str] = []
    for idx, sample in enumerate(samples):
        sample_x = np.asarray(sample["x"], dtype=np.float32)
        sample_y = sample["y"]
        sample_n = sample_x.shape[0]

        # Left-aligned copy keeps sample channel order intact.
        x[idx, :sample_n] = torch.as_tensor(sample_x, dtype=torch.float32)
        y[idx] = sample_y
        channel_mask[idx, :sample_n] = True

        if coords_tensor is not None:
            coords = np.asarray(sample["channel_coords_lip"], dtype=np.float32)
            coords_tensor[idx, :sample_n] = torch.as_tensor(coords, dtype=torch.float32)

        if seq_tensor is not None:
            seq = np.asarray(sample["seq_id"], dtype=np.int64)
            seq_tensor[idx, :sample_n] = torch.as_tensor(seq, dtype=torch.long)

        channel_ids.append(list(sample["channel_ids"]))
        recording_ids.append(sample["recording_id"])

    return {
        "x": x,
        "y": y,
        "channel_mask": channel_mask,
        "channel_coords_lip": coords_tensor,
        "seq_id": seq_tensor,
        "channel_ids": channel_ids,
        "n_channels": n_channels_tensor,
        "recording_ids": recording_ids,
        "split": split,
    }
