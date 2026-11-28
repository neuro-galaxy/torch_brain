"""Preparation provenance and strict resume checks for naturalistic movie splits."""

import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import h5py
import numpy as np

from torch_brain.utils.split import PROTOCOL_VERSION

MOVIE_DERIVED_VERSION = "1.2.0"


def validate_temporal_preparation_config(
    *,
    context_width_s,
    gap_s,
    balance_seed,
    balanced,
    no_splits,
    pre_offset_s=0.0,
    post_offset_s=1.0,
):
    """Reject invalid selection settings before annotation or waveform reads."""
    for name, value in (("balanced", balanced), ("no_splits", no_splits)):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    for name, value in (
        ("context_width_s", context_width_s),
        ("gap_s", gap_s),
        ("pre_offset_s", pre_offset_s),
        ("post_offset_s", post_offset_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    if context_width_s < 1:
        raise ValueError("context_width_s must be at least 1 second")
    if pre_offset_s + post_offset_s <= 0:
        raise ValueError("Target offsets must define a positive-duration window")
    if not no_splits and context_width_s < pre_offset_s + post_offset_s:
        raise ValueError("context_width_s cannot be shorter than target windows")
    if (
        isinstance(balance_seed, bool)
        or not isinstance(balance_seed, Integral)
        or balance_seed < 0
    ):
        raise ValueError("balance_seed must be a nonnegative integer")


def preparation_settings(*, splits_enabled, pre_offset_s, post_offset_s, label_paths):
    """Record extra selection inputs; temporal settings remain in native metadata.

    Hash source annotations, never neural signals. Stream large feature tables;
    logical source names keep identities portable across different data roots.
    """
    labels = {}
    for name, path in sorted(label_paths.items()):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        labels[name] = digest.hexdigest()
    return {
        "splits_enabled": splits_enabled,
        "pre_offset_s": float(pre_offset_s),
        "post_offset_s": float(post_offset_s),
        "label_files": labels,
    }


def _validate_prepared_structure(handle, *, splits_enabled):
    """Check H5 layout and annotation shapes without reading neural samples."""
    for key in ("brainset", "subject", "session", "domain", "channels", "seeg_data"):
        if key not in handle or not isinstance(handle[key], h5py.Group):
            raise ValueError(f"missing required recording group {key}")
    signal = handle["seeg_data/data"]
    if (
        not isinstance(signal, h5py.Dataset)
        or signal.ndim != 2
        or any(size <= 0 for size in signal.shape)
    ):
        raise ValueError("seeg_data/data must be a nonempty two-dimensional array")
    channel_ids = handle["channels/id"]
    if not isinstance(channel_ids, h5py.Dataset) or channel_ids.shape != (
        signal.shape[1],
    ):
        raise ValueError("channels/id length must match the signal channel count")
    if any((key in handle) != splits_enabled for key in ("splits", "channel_splits")):
        raise ValueError("stored split/channel groups disagree with requested policy")
    if not splits_enabled:
        return
    splits, masks = handle["splits"], handle["channel_splits"]
    if not isinstance(splits, h5py.Group) or not isinstance(masks, h5py.Group):
        raise ValueError("splits and channel_splits must be groups")
    # Data(domain=...) serializes its domain alongside the actual selectors.
    # Mask containers usually omit it, but either container may carry it.
    keys = set(splits) - {"domain"}
    mask_keys = set(masks) - {"domain"}
    if not keys or keys != mask_keys:
        raise ValueError(
            "split and channel-mask selector keys must be nonempty and match"
        )
    split_sets, native_folds = {}, {}
    for key in keys:
        fields = key.split("$")
        if len(fields) != 6 or not all(fields):
            raise ValueError(f"invalid split selector {key}")
        tier, mode, regime, task, fold, split = fields
        if (
            not fold.startswith("fold")
            or not fold[4:].isdigit()
            or split not in {"train", "val", "test"}
        ):
            raise ValueError(f"invalid split selector {key}")
        split_sets.setdefault((tier, mode, regime, task, fold), set()).add(split)
        if regime == "within_session":
            native_folds.setdefault((tier, mode, regime, task), set()).add(fold)
        group = splits[key]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"split selector must contain annotation arrays: {key}")
        arrays = [group[field] for field in ("start", "end", "label")]
        if (
            any(
                not isinstance(array, h5py.Dataset) or array.ndim != 1
                for array in arrays
            )
            or len({array.shape for array in arrays}) != 1
        ):
            raise ValueError(
                f"start/end/label arrays must be matching one-dimensional arrays: {key}"
            )
        mask = masks[key]
        if (
            not isinstance(mask, h5py.Dataset)
            or mask.shape != (signal.shape[1],)
            or mask.dtype != np.dtype(bool)
        ):
            raise ValueError(
                f"channel mask must be boolean with one value per signal channel: {key}"
            )
    if any(parts != {"train", "val", "test"} for parts in split_sets.values()):
        raise ValueError("each saved fold must contain train, val, and test selectors")
    if not native_folds or any(
        folds != {"fold0", "fold1"} for folds in native_folds.values()
    ):
        raise ValueError("native temporal selections must contain both fold0 and fold1")


def validate_prepared_temporal_settings(
    path,
    *,
    context_width_s,
    gap_s,
    balance_seed,
    balanced,
    splits_enabled,
    preparation_settings=None,
):
    """Require current provenance before reusing a prepared movie recording.

    Callers may omit the extra-settings comparison when inspecting an inventory;
    pipeline resume supplies it to check target offsets and source annotations too.
    """
    path = Path(path)
    hint = (
        f"{path}: incompatible preparation; use --reprocess or a new output directory"
    )
    try:
        with h5py.File(path, "r") as handle:
            if handle["brainset"].attrs["derived_version"] != MOVIE_DERIVED_VERSION:
                raise ValueError(
                    "derived_version does not match the current movie release"
                )
            _validate_prepared_structure(handle, splits_enabled=splits_enabled)
            settings = json.loads(handle.attrs["preparation_settings_json"])
            if not isinstance(settings, dict) or set(settings) != {
                "splits_enabled",
                "pre_offset_s",
                "post_offset_s",
                "label_files",
            }:
                raise ValueError("missing or unknown preparation settings")
            if (
                type(settings["splits_enabled"]) is not bool
                or settings["splits_enabled"] != splits_enabled
            ):
                raise ValueError("split-enabled policy differs")
            for key in ("pre_offset_s", "post_offset_s"):
                value = settings[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"invalid saved {key}")
            if settings["pre_offset_s"] + settings["post_offset_s"] <= 0:
                raise ValueError("invalid saved target duration")
            labels = settings["label_files"]
            if not isinstance(labels, dict) or any(
                not isinstance(name, str)
                or not name
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                for name, digest in labels.items()
            ):
                raise ValueError("invalid saved label identities")
            if bool(labels) != splits_enabled:
                raise ValueError("label identities disagree with split-enabled policy")
            if preparation_settings is not None and settings != preparation_settings:
                raise ValueError("target offsets or source annotations differ")
            if splits_enabled:
                metadata = json.loads(handle.attrs["temporal_splits_json"])
                if not isinstance(metadata, dict):
                    raise ValueError("invalid saved temporal metadata")
                expected = {
                    "protocol": PROTOCOL_VERSION,
                    "context_width_s": context_width_s,
                    "gap_s": gap_s,
                    "balance_seed": balance_seed,
                    "balanced": balanced,
                }
                if (
                    type(metadata.get("balanced")) is not bool
                    or type(metadata.get("balance_seed")) is not int
                    or metadata["balance_seed"] < 0
                ):
                    raise ValueError("invalid saved balancing policy")
                for key in ("context_width_s", "gap_s"):
                    value = metadata.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not np.isfinite(value)
                    ):
                        raise ValueError(f"invalid saved {key}")
                if (
                    metadata["context_width_s"]
                    < max(1.0, settings["pre_offset_s"] + settings["post_offset_s"])
                    or metadata["gap_s"] < 0
                ):
                    raise ValueError("invalid saved temporal support")
                for key, value in expected.items():
                    if key not in metadata or metadata[key] != value:
                        raise ValueError(
                            f"temporal setting {key} differs or is missing"
                        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{hint}: {exc}") from exc
