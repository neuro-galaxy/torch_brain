"""
Preprocessor that drops channels not included in an allowed list.
"""

import json
import os
import re
import numpy as np

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("channel_subselect")
class ChannelSubselectPreprocessor(BasePreprocessor):
    """Selects a subset of channels based on allowed electrode labels."""

    def __init__(self, cfg):
        super().__init__(cfg)
        allowed = cfg.get("allowed_electrodes")
        self.subject = cfg.get("subject")
        self.allowed_electrodes_by_subject = None
        self.allowed_electrodes = self._normalize_allowed(allowed)
        self._initial_allowed = self.allowed_electrodes

    def set_allowed_electrodes(self, electrode_labels):
        if electrode_labels is None:
            self.allowed_electrodes = self._initial_allowed
        else:
            self.allowed_electrodes = set(electrode_labels)

    def _normalize_subject_key(self, value):
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    def _infer_subject_from_label(self, label):
        if not isinstance(label, str) or "/" not in label:
            return None
        if not self.allowed_electrodes_by_subject:
            return None
        parts = [p.strip() for p in label.split("/") if p.strip()]
        for part in parts:
            if part in self.allowed_electrodes_by_subject:
                return part
        for part in parts:
            match = re.search(r"(sub_\d+)", part)
            if match:
                subject = match.group(1)
                if subject in self.allowed_electrodes_by_subject:
                    return subject
        return None

    def _is_allowed_label(self, label):
        if self.allowed_electrodes is None:
            return True
        if label in self.allowed_electrodes:
            return True
        if isinstance(label, str) and "/" in label:
            # Variable-channel IDs may be prefixed as subject/session/channel.
            # Accept by raw channel-id suffix when clean lists are unprefixed.
            suffix = label.split("/")[-1]
            if suffix in self.allowed_electrodes:
                return True
            inferred_subject = self._infer_subject_from_label(label)
            if inferred_subject is not None:
                subject_allowed = self.allowed_electrodes_by_subject.get(
                    inferred_subject, set()
                )
                if label in subject_allowed:
                    return True
                if suffix in subject_allowed:
                    return True
        return False

    def _normalize_allowed(self, allowed):
        if allowed is None:
            return None
        if isinstance(allowed, str):
            if not os.path.exists(allowed):
                raise FileNotFoundError(f"Allowed electrodes file not found: {allowed}")
            with open(allowed, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.allowed_electrodes_by_subject = {
                    str(key): set(values) for key, values in loaded.items()
                }
                subject_key = self._normalize_subject_key(self.subject)
                if subject_key is not None:
                    if subject_key in loaded:
                        loaded = loaded[subject_key]
                    else:
                        raise ValueError(
                            f"Subject {subject_key} not found in allowed electrodes JSON."
                        )
                else:
                    # No explicit subject selected: use union as default fallback and
                    # resolve subject-specific lists dynamically when prefixes exist.
                    merged = set()
                    for values in loaded.values():
                        merged.update(values)
                    loaded = list(merged)
            allowed = loaded
        return set(allowed)

    def _transform_one(self, sample):
        """Filter one sample by channel_names while preserving channel_ids alignment."""
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample or "channel_names" not in sample:
            raise KeyError(
                "channel_subselect requires sample['x'] and sample['channel_names']."
            )
        if "channel_ids" not in sample:
            raise KeyError("channel_subselect requires sample['channel_ids'].")

        channel_names_in = sample["channel_names"]
        if channel_names_in is None:
            raise ValueError(
                "channel_subselect requires sample['channel_names'] to be set."
            )
        if not isinstance(channel_names_in, list):
            raise TypeError("sample['channel_names'] must be a list[str].")
        if any(not isinstance(name, str) for name in channel_names_in):
            raise TypeError("sample['channel_names'] must contain only str values.")

        channel_ids_in = list(sample["channel_ids"])
        if len(channel_ids_in) != len(channel_names_in):
            raise ValueError(
                "sample['channel_ids'] and sample['channel_names'] length mismatch: "
                f"{len(channel_ids_in)} vs {len(channel_names_in)}."
            )

        x_in = np.asarray(sample["x"], dtype=np.float32)
        if x_in.ndim < 2:
            raise ValueError(
                "channel_subselect expects sample['x'] to be at least 2D "
                f"(channels, features...), got {x_in.shape}."
            )
        if x_in.shape[0] != len(channel_names_in):
            raise ValueError(
                "sample['x'] channel axis must match channel_names length: "
                f"{x_in.shape[0]} vs {len(channel_names_in)}."
            )

        if self.allowed_electrodes is None:
            aligned_old_indices = list(range(len(channel_names_in)))
        else:
            keep_mask = [self._is_allowed_label(name) for name in channel_names_in]
            aligned_old_indices = [i for i, keep in enumerate(keep_mask) if keep]
            if not aligned_old_indices:
                raise ValueError(
                    "No electrodes remain after channel_subselect preprocessing."
                )

        x_out_np = x_in[aligned_old_indices]
        channel_names_out = [channel_names_in[i] for i in aligned_old_indices]

        if x_out_np.ndim < 2:
            raise ValueError(
                "channel_subselect output must be at least 2D (channels, features...), "
                f"got shape {x_out_np.shape}."
            )
        if x_out_np.shape[0] != len(channel_names_out):
            raise ValueError(
                "channel_subselect output channel axis does not match returned labels: "
                f"{x_out_np.shape[0]} vs {len(channel_names_out)}."
            )

        out = dict(sample)
        out["x"] = np.asarray(x_out_np, dtype=np.float32)
        out["channel_names"] = list(channel_names_out)
        out["channel_ids"] = [channel_ids_in[i] for i in aligned_old_indices]

        coords = sample.get("channel_coords_lip")
        if coords is not None:
            coords_arr = np.asarray(coords)
            if coords_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "channel_coords_lip first dimension mismatch: "
                    f"{coords_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["channel_coords_lip"] = coords_arr[aligned_old_indices]

        seq_id = sample.get("seq_id")
        if seq_id is not None:
            seq_arr = np.asarray(seq_id)
            if seq_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "seq_id first dimension mismatch: "
                    f"{seq_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["seq_id"] = seq_arr[aligned_old_indices]

        brain_areas = sample.get("brain_areas")
        if brain_areas is not None:
            brain_areas_arr = np.asarray(brain_areas, dtype=object).reshape(-1)
            if brain_areas_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "brain_areas first dimension mismatch: "
                    f"{brain_areas_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["brain_areas"] = brain_areas_arr[aligned_old_indices].tolist()

        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
