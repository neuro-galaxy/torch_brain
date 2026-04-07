"""Processed dataset adapter for variable-channel runs."""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import numpy as np
import torch

from neuroprobe_eval.utils.pipeline_contracts import (
    build_processed_split_provider,
    needs_region_intersection_pool,
)
from neuroprobe_eval.utils.logging_utils import log, log_fold_split_sample_counts


def _normalize_brain_area_array(
    values: Any,
    *,
    expected_len: int,
    context: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{context} must have length {expected_len}, got {len(arr)}.")
    out: list[str] = []
    for value in arr:
        if value is None:
            raise ValueError(f"{context} contains None labels.")
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            raise ValueError(f"{context} contains NaN labels.")
        label = str(value).strip()
        if not label:
            raise ValueError(f"{context} contains empty labels.")
        out.append(label)
    return np.asarray(out, dtype=object)


def _iter_preprocessor_stages(preprocessor: Any) -> list[Any]:
    """Return preprocessors to profile for variable-channel materialization."""
    stages = getattr(preprocessor, "preprocessors", None)
    # Composite preprocessors expose an explicit stage list; legacy preprocessors
    # are treated as a single stage.
    if isinstance(stages, (list, tuple)) and stages:
        return list(stages)
    return [preprocessor]


def _resolve_dataset_brain_area_key(
    dataset_cfg: Any, *, needs_pool: bool
) -> str | None:
    """Resolve brain-area source key from dataset config."""
    raw_key = getattr(dataset_cfg, "brain_area_key", None)
    if raw_key is None:
        if needs_pool:
            raise ValueError(
                "dataset.brain_area_key is required when region-intersection "
                "pooling is enabled."
            )
        return None
    if not isinstance(raw_key, str):
        raise TypeError(
            "dataset.brain_area_key must be a str when set, got "
            f"{type(raw_key).__name__}."
        )
    if raw_key.strip() != raw_key:
        raise ValueError(
            "dataset.brain_area_key must not include leading/trailing whitespace. "
            f"Got '{raw_key}'."
        )
    if raw_key == "":
        raise ValueError("dataset.brain_area_key must be non-empty when set.")
    return raw_key


def _collect_split_brain_areas(split_dataset: Any) -> list[str]:
    # DS-DM pooling computes the train/test region intersection; this helper
    # collects candidate labels from one split in canonical sorted order.
    regions: set[str] = set()
    for sample in split_dataset.iter_preprocess_fit_samples():
        x = np.asarray(sample.get("x"))
        if x.ndim < 2:
            raise ValueError(
                "Region-intersection pooling requires sample['x'] with shape "
                f"(channels, ...), got {x.shape}."
            )
        n_channels = x.shape[0]
        brain_areas = sample.get("brain_areas")
        if brain_areas is None:
            raise ValueError(
                "Region-intersection pooling requires non-empty test-side brain-area "
                "metadata; sample['brain_areas'] is missing."
            )
        try:
            labels = _normalize_brain_area_array(
                brain_areas,
                expected_len=n_channels,
                context="sample['brain_areas']",
            )
        except ValueError as exc:
            raise ValueError(
                "Region-intersection pooling requires non-empty test-side brain-area "
                "metadata; sample['brain_areas'] is invalid."
            ) from exc
        regions.update(labels.tolist())
    return sorted(regions)


def validate_provider_interface(provider: Any) -> None:
    # Fail once with a complete missing-method list instead of surfacing
    # one AttributeError at a later call site.
    required = (
        "get_sampling_intervals",
        "get_recording",
        "get_channel_metadata",
        "describe_selection",
    )
    missing = [
        name
        for name in required
        if not hasattr(provider, name) or not callable(getattr(provider, name))
    ]
    if missing:
        raise TypeError(
            "Provider is missing required methods for variable-channel adapter: "
            + ", ".join(missing)
        )


def validate_sample_dict(
    sample: dict[str, Any],
    *,
    expected_split: str | None = None,
    require_coords: bool = False,
) -> None:
    required_keys = {
        "x",
        "y",
        "channel_ids",
        "channel_coords_lip",
        "seq_id",
        "recording_id",
        "split",
        "sample_idx",
        "window_start_sec",
        "window_end_sec",
    }
    missing = sorted(required_keys - set(sample.keys()))
    if missing:
        raise KeyError(f"Sample dict missing required keys: {missing}")

    x = np.asarray(sample["x"])
    if x.ndim < 2:
        raise ValueError(f"sample['x'] must be at least 2D, got shape {x.shape}.")
    n_channels = x.shape[0]

    channel_ids = sample["channel_ids"]
    if not isinstance(channel_ids, list):
        raise TypeError("sample['channel_ids'] must be a list[str].")
    if len(channel_ids) != n_channels:
        raise ValueError(
            "sample['channel_ids'] length must match x.shape[0], got "
            f"{len(channel_ids)} vs {n_channels}."
        )

    brain_areas = sample.get("brain_areas")
    if brain_areas is not None:
        _normalize_brain_area_array(
            brain_areas,
            expected_len=n_channels,
            context="sample['brain_areas']",
        )

    coords = sample["channel_coords_lip"]
    if coords is None:
        # Some model paths can run without coords, but coord-dependent models
        # (for example PopT) require them and opt in via require_coords=True.
        if require_coords:
            raise ValueError(
                "sample['channel_coords_lip'] is required for this configuration."
            )
    else:
        coords = np.asarray(coords)
        if coords.shape != (n_channels, 3):
            raise ValueError(
                "sample['channel_coords_lip'] must have shape "
                f"({n_channels}, 3), got {coords.shape}."
            )

    seq_id = sample["seq_id"]
    if seq_id is not None:
        seq_id = np.asarray(seq_id)
        if seq_id.shape != (n_channels,):
            raise ValueError(
                f"sample['seq_id'] must have shape ({n_channels},), got {seq_id.shape}."
            )

    if not isinstance(sample["split"], str):
        raise TypeError(
            f"sample['split'] must be a str, got {type(sample['split']).__name__}."
        )
    if expected_split is not None and sample["split"] != expected_split:
        raise ValueError(
            f"sample['split'] mismatch: expected '{expected_split}', got '{sample['split']}'."
        )
    if not isinstance(sample["recording_id"], str):
        raise TypeError(
            "sample['recording_id'] must be a str, got "
            f"{type(sample['recording_id']).__name__}."
        )
    if not isinstance(sample["y"], (int, np.integer)) or isinstance(sample["y"], bool):
        raise TypeError(
            f"sample['y'] must be an int, got {type(sample['y']).__name__}."
        )
    if any(not isinstance(cid, str) for cid in channel_ids):
        raise TypeError("sample['channel_ids'] must contain only str values.")

    channel_names = sample.get("channel_names")
    if channel_names is not None:
        if not isinstance(channel_names, list):
            raise TypeError("sample['channel_names'] must be a list[str] when set.")
        if len(channel_names) != n_channels:
            raise ValueError(
                "sample['channel_names'] length must match x.shape[0], got "
                f"{len(channel_names)} vs {n_channels}."
            )
        if any(not isinstance(name, str) for name in channel_names):
            raise TypeError("sample['channel_names'] must contain only str values.")


def validate_fold_dict(fold: dict[str, Any]) -> None:
    required = {
        "fold_idx",
        "train_split",
        "val_split",
        "test_split",
        "preprocess_state",
        "metadata",
    }
    missing = sorted(required - set(fold.keys()))
    if missing:
        raise KeyError(f"Fold dict missing required keys: {missing}")
    for split_key in ("train_split", "val_split", "test_split"):
        split_obj = fold[split_key]
        if not hasattr(split_obj, "__len__") or not hasattr(split_obj, "__getitem__"):
            raise TypeError(
                f"Fold key '{split_key}' must be dataset-like (__len__/__getitem__)."
            )


class WindowedNeuroprobeSplitDataset(torch.utils.data.Dataset):
    """Split-local sample dataset with optional in-memory materialization."""

    def __init__(
        self,
        provider: Any,
        *,
        split: str,
        require_coords: bool = False,
        brain_area_key: str | None = None,
    ):
        self.provider = provider
        self.split = split
        if not isinstance(require_coords, bool):
            raise TypeError(
                "require_coords must be a bool, got "
                f"{type(require_coords).__name__}."
            )
        if brain_area_key is not None and not isinstance(brain_area_key, str):
            raise TypeError(
                "brain_area_key must be a str when provided, got "
                f"{type(brain_area_key).__name__}."
            )
        if isinstance(brain_area_key, str) and not brain_area_key.strip():
            raise ValueError("brain_area_key must be non-empty when provided.")
        self.require_coords = require_coords
        self._brain_area_key = (
            brain_area_key.strip() if isinstance(brain_area_key, str) else None
        )
        # Validate the provider contract up front so dataset construction fails
        # before any per-sample processing starts.
        validate_provider_interface(self.provider)

        self._selection = self.provider.describe_selection()
        self._interval_map = self.provider.get_sampling_intervals()
        if not isinstance(self._interval_map, dict) or not self._interval_map:
            raise ValueError(
                "Provider get_sampling_intervals() must return a non-empty dict."
            )

        self._flat_index: list[tuple[str, int]] = []
        # Keep deterministic sample ordering across runs by sorting recording ids.
        for recording_id in sorted(self._interval_map.keys()):
            interval = self._interval_map[recording_id]
            starts = np.asarray(interval.start)
            ends = np.asarray(interval.end)
            labels = np.asarray(interval.label)
            if not (len(starts) == len(ends) == len(labels)):
                raise ValueError(
                    f"Interval array length mismatch for recording '{recording_id}'."
                )
            for i in range(len(starts)):
                self._flat_index.append((recording_id, i))

        self._recording_cache: dict[str, Any] = {}
        self._channel_cache: dict[str, dict[str, Any]] = {}
        self._materialized: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        if self._materialized is not None:
            return len(self._materialized)
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._materialized is not None:
            return self._materialized[idx]
        return self._build_raw_sample(idx)

    def is_materialized(self) -> bool:
        return self._materialized is not None

    def get_split_summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_samples": len(self),
            "n_recordings": len(self._interval_map),
            "recording_ids": sorted(self._interval_map.keys()),
            "selection": self._selection,
            "is_materialized": self.is_materialized(),
        }

    def _iter_raw_samples(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self._flat_index)):
            yield self._build_raw_sample(i)

    def iter_preprocess_fit_samples(self) -> Iterable[dict[str, Any]]:
        if self._materialized is not None:
            return iter(self._materialized)
        return self._iter_raw_samples()

    def set_materialized_samples(
        self, samples: Iterable[dict[str, Any]]
    ) -> "WindowedNeuroprobeSplitDataset":
        out_samples = []
        for sample in samples:
            validate_sample_dict(
                sample,
                expected_split=self.split,
                require_coords=self.require_coords,
            )
            out_samples.append(sample)
        self._materialized = out_samples
        return self

    def materialize(
        self,
        preprocessor=None,
        *,
        profile_preprocessor: bool = False,
        log_prefix: str = "",
    ) -> "WindowedNeuroprobeSplitDataset":
        if self._materialized is not None:
            return self

        stage_chain = None
        stage_stats = None
        if preprocessor is not None and profile_preprocessor:
            # Optional stage-level telemetry helps diagnose shape/channel changes
            # introduced by chained preprocessors.
            stage_chain = _iter_preprocessor_stages(preprocessor)
            stage_stats = [
                {
                    "name": getattr(stage, "cfg", {}).get("name", "unknown"),
                    "elapsed": 0.0,
                    "shape": None,
                    "shape_varies": False,
                    "min_electrodes": None,
                    "max_electrodes": None,
                }
                for stage in stage_chain
            ]

        out_samples = list(self._iter_raw_samples())
        if preprocessor is None:
            # Raw-materialization path: keep provider samples as-is.
            pass
        elif stage_chain is None:
            # Fast path for one preprocessor object. Use the batch-capable hook
            # so heavy stages can process multiple samples in one call.
            out_samples = list(preprocessor.transform_samples(out_samples))
        else:
            # Profiling path: run each stage over the full split so stages with
            # custom transform_samples(...) can batch internally.
            for stage_idx, stage in enumerate(stage_chain):
                start_time = time.time()
                out_samples = list(stage.transform_samples(out_samples))
                elapsed = time.time() - start_time

                stats = stage_stats[stage_idx]
                stats["elapsed"] += elapsed
                for transformed in out_samples:
                    x_out = (
                        transformed.get("x") if isinstance(transformed, dict) else None
                    )
                    if x_out is not None:
                        # Track whether stage output shape is stable or sample-dependent.
                        shape = tuple(np.asarray(x_out).shape)
                        if stats["shape"] is None:
                            stats["shape"] = shape
                        elif stats["shape"] != shape:
                            stats["shape_varies"] = True

                    channel_ids = (
                        transformed.get("channel_ids")
                        if isinstance(transformed, dict)
                        else None
                    )
                    if isinstance(channel_ids, list):
                        # Record output channel-count range for quick stage diagnostics.
                        n_electrodes = len(channel_ids)
                        if (
                            stats["min_electrodes"] is None
                            or n_electrodes < stats["min_electrodes"]
                        ):
                            stats["min_electrodes"] = n_electrodes
                        if (
                            stats["max_electrodes"] is None
                            or n_electrodes > stats["max_electrodes"]
                        ):
                            stats["max_electrodes"] = n_electrodes

        for transformed in out_samples:
            # Enforce adapter sample contract after all preprocessing stages.
            validate_sample_dict(
                transformed,
                expected_split=self.split,
                require_coords=self.require_coords,
            )

        if stage_stats is not None:
            prefix = f"{log_prefix}" if log_prefix else ""
            # Emit one consolidated telemetry line per stage for this split.
            for stats in stage_stats:
                if stats["shape"] is None:
                    shape_str = "unknown"
                elif stats["shape_varies"]:
                    shape_str = f"variable (example {stats['shape']})"
                else:
                    shape_str = str(stats["shape"])

                min_e = stats["min_electrodes"]
                max_e = stats["max_electrodes"]
                if min_e is None:
                    electrode_str = "N/A"
                elif min_e == max_e:
                    electrode_str = str(min_e)
                else:
                    electrode_str = f"{min_e}-{max_e}"

                log(
                    f"{prefix}Preprocessor '{stats['name']}' completed: "
                    f"output shape {shape_str}, {electrode_str} electrodes, "
                    f"{stats['elapsed']:.3f}s",
                    priority=2,
                    indent=1,
                )

        self._materialized = out_samples
        return self

    def _get_recording(self, recording_id: str):
        cached = self._recording_cache.get(recording_id)
        if cached is not None:
            return cached
        # Cache recordings because each recording contributes multiple windows.
        rec = self.provider.get_recording(recording_id)
        self._recording_cache[recording_id] = rec
        return rec

    def _get_channel_meta(self, recording_id: str) -> dict[str, Any]:
        cached = self._channel_cache.get(recording_id)
        if cached is not None:
            return cached

        channel_metadata = dict(self.provider.get_channel_metadata(recording_id))
        for key in ("indices", "ids", "included_mask"):
            if key not in channel_metadata:
                raise KeyError(
                    "Provider channel metadata missing key "
                    f"'{key}' for recording '{recording_id}'."
                )

        indices_all = np.asarray(channel_metadata["indices"], dtype=int).reshape(-1)
        ids_all = np.asarray(channel_metadata["ids"]).astype(str).reshape(-1)
        included_mask = np.asarray(
            channel_metadata["included_mask"], dtype=bool
        ).reshape(-1)
        if len(ids_all) != len(included_mask):
            raise ValueError(
                "ids/included_mask length mismatch for recording "
                f"'{recording_id}': {len(ids_all)} vs {len(included_mask)}."
            )
        if len(indices_all) != len(ids_all):
            raise ValueError(
                "indices/ids length mismatch for recording "
                f"'{recording_id}': {len(indices_all)} vs {len(ids_all)}."
            )

        indices = indices_all[included_mask]
        ids = ids_all[included_mask]
        # Channel identity should come from provider ids. The dataset/mixin path
        # is responsible for any subject/session uniqueness policy.
        channel_ids = ids.tolist()

        names_arr = channel_metadata.get("names")
        channel_names: list[str] | None = None
        if names_arr is not None:
            names = np.asarray(names_arr).astype(str).reshape(-1)
            if len(names) != len(ids_all):
                raise ValueError(
                    f"ids/names length mismatch for recording '{recording_id}': "
                    f"{len(ids_all)} vs {len(names)}."
                )
            channel_names = names[included_mask].tolist()

        coords = channel_metadata.get("coords")
        coords_type = channel_metadata.get("coords_type", "lip")
        if not isinstance(coords_type, str):
            raise TypeError(
                "coords_type must be a str for recording "
                f"'{recording_id}', got {type(coords_type).__name__}."
            )
        coords_type = coords_type.strip().lower()
        if coords is not None and coords_type != "lip":
            raise ValueError(
                "Unsupported channel coordinate type "
                f"'{channel_metadata.get('coords_type')}' for recording '{recording_id}'."
            )
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
            if len(coords) != len(ids_all):
                raise ValueError(
                    f"coords must have length {len(ids_all)}, got {len(coords)}."
                )
            coords = coords[included_mask]
            if coords.shape != (len(ids), 3):
                raise ValueError(
                    f"coords must have shape ({len(ids)}, 3), got {coords.shape}."
                )

        rec = self._get_recording(recording_id)

        def _maybe_extract_brain_areas(source, *, source_name: str):
            if source is None:
                return None
            arr = np.asarray(source, dtype=object).reshape(-1)
            # Accept either full-channel metadata arrays, already-filtered arrays,
            # or indexable arrays that can be projected to selected channels.
            if len(arr) == len(ids_all):
                arr = arr[included_mask]
                return _normalize_brain_area_array(
                    arr,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            if len(arr) == len(indices):
                return _normalize_brain_area_array(
                    arr,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            max_index = np.max(indices) if len(indices) else -1
            if len(arr) > max_index:
                selected = arr[indices]
                return _normalize_brain_area_array(
                    selected,
                    expected_len=len(indices),
                    context=f"{source_name} brain-area labels",
                )
            return None

        brain_areas = None
        key = self._brain_area_key
        if key is not None:
            # Source precedence for brain_area_key:
            # 1) provider.get_channel_metadata(recording_id)[key]
            # 2) recording.channels.<key> fallback when metadata does not expose key
            sources_seen: list[str] = []
            if key in channel_metadata:
                sources_seen.append(f"channel metadata '{key}'")
                brain_areas = _maybe_extract_brain_areas(
                    channel_metadata.get(key),
                    source_name=f"channel metadata '{key}'",
                )
                if brain_areas is None:
                    raise ValueError(
                        f"Configured brain_area_key '{key}' in channel metadata for "
                        f"recording '{recording_id}' could not be aligned to selected "
                        "channels."
                    )
            elif hasattr(rec, "channels") and hasattr(rec.channels, key):
                sources_seen.append(f"recording channels '{key}'")
                brain_areas = _maybe_extract_brain_areas(
                    getattr(rec.channels, key),
                    source_name=f"recording channels '{key}'",
                )
                if brain_areas is None:
                    raise ValueError(
                        f"Configured brain_area_key '{key}' in recording "
                        f"'{recording_id}' could not be aligned to selected channels."
                    )
            if brain_areas is None:
                source_msg = (
                    f"found in {', '.join(sources_seen)} but invalid"
                    if sources_seen
                    else "missing"
                )
                raise ValueError(
                    f"Configured brain_area_key '{key}' is {source_msg} for "
                    f"recording '{recording_id}'."
                )

        meta = {
            "indices": indices,
            "channel_ids": channel_ids,
            "channel_names": (None if channel_names is None else list(channel_names)),
            "channel_coords_lip": coords,
            "brain_areas": (
                None
                if brain_areas is None
                else np.asarray(brain_areas, dtype=object).copy()
            ),
            # V1: seq_id is all zeros (single sequence per sample), matching
            # the existing _create_seq_id behavior.  Richer grouping semantics
            # are deferred (see plan: Out of Scope).
            "seq_id": np.zeros(len(channel_ids), dtype=np.int64),
        }
        self._channel_cache[recording_id] = meta
        return meta

    def _build_raw_sample(self, idx: int) -> dict[str, Any]:
        recording_id, interval_idx = self._flat_index[idx]
        interval = self._interval_map[recording_id]

        start = float(np.asarray(interval.start, dtype=np.float64)[interval_idx])
        end = float(np.asarray(interval.end, dtype=np.float64)[interval_idx])
        label = np.asarray(interval.label)[interval_idx]

        rec = self._get_recording(recording_id)
        meta = self._get_channel_meta(recording_id)
        window = rec.slice(start, end)
        window_data = np.asarray(window.seeg_data.data)
        if window_data.ndim != 2:
            raise ValueError(
                f"Expected window seeg_data.data to be 2D, got shape {window_data.shape}."
            )
        # Assumes seeg_data.data is (time, all_channels).  Transpose after
        # channel selection gives (selected_channels, time) = (channels, *feature_shape).
        x = np.asarray(window_data[:, meta["indices"]].T, dtype=np.float32)
        expected_channels = len(meta["channel_ids"])
        if x.shape[0] != expected_channels:
            raise ValueError(
                "Channel count mismatch after selection: "
                f"x has {x.shape[0]} channels but channel_ids has "
                f"{expected_channels} entries."
            )

        sample = {
            "x": x,
            "y": label,
            "channel_ids": list(meta["channel_ids"]),
            "channel_names": (
                None if meta["channel_names"] is None else list(meta["channel_names"])
            ),
            "channel_coords_lip": (
                None
                if meta["channel_coords_lip"] is None
                else np.asarray(meta["channel_coords_lip"], dtype=np.float32).copy()
            ),
            "brain_areas": (
                None
                if meta["brain_areas"] is None
                else np.asarray(meta["brain_areas"], dtype=object).astype(str).tolist()
            ),
            "seq_id": np.asarray(meta["seq_id"], dtype=np.int64).copy(),
            "recording_id": recording_id,
            "split": self.split,
            "sample_idx": idx,
            "window_start_sec": start,
            "window_end_sec": end,
        }
        validate_sample_dict(
            sample,
            expected_split=self.split,
            require_coords=self.require_coords,
        )
        return sample


def build_neuroprobe_torch_fold(
    dataset_cfg: Any,
    preprocessor,
    *,
    fold_idx: int,
    seed: int,
    require_coords: bool = True,
    needs_pool: bool | None = None,
) -> dict[str, Any]:
    fold_seed = seed
    dataset_provider = dataset_cfg.provider
    regime = dataset_cfg.regime
    if needs_pool is None:
        requires_aligned = getattr(dataset_cfg, "requires_aligned_channels", True)
        needs_pool = needs_region_intersection_pool(
            dataset_provider, regime, requires_aligned
        )

    split_providers = {}
    split_ctor_seconds = {}
    for split in ("train", "val", "test"):
        split_ctor_start = time.time()
        # Build one provider instance per split so split-specific sampling
        # boundaries remain isolated.
        split_providers[split] = build_processed_split_provider(
            dataset_provider=dataset_provider,
            dataset_cfg=dataset_cfg,
            split=split,
            fold_idx=fold_idx,
            regime=regime,
        )
        split_ctor_seconds[split] = time.time() - split_ctor_start

    log(
        f"Fold {fold_idx} dataset provider '{dataset_provider}' "
        "construction timings: "
        f"train={split_ctor_seconds['train']:.2f}s "
        f"val={split_ctor_seconds['val']:.2f}s "
        f"test={split_ctor_seconds['test']:.2f}s",
        priority=0,
    )

    brain_area_key = _resolve_dataset_brain_area_key(
        dataset_cfg,
        needs_pool=needs_pool,
    )

    split_datasets = {
        split: WindowedNeuroprobeSplitDataset(
            split_providers[split],
            split=split,
            require_coords=require_coords,
            brain_area_key=brain_area_key,
        )
        for split in ("train", "val", "test")
    }
    log_fold_split_sample_counts(split_datasets, fold_idx=fold_idx, phase="raw")

    preprocess_state = None
    if preprocessor is not None:
        if hasattr(preprocessor, "reset_state"):
            preprocessor.reset_state()
        fold_context = {
            "dataset_provider": dataset_provider,
            "dataset_regime": regime,
            "needs_region_intersection_pool": needs_pool,
        }
        if needs_pool:
            # RegionIntersectionPoolPreprocessor needs test-side labels to
            # compute the train/test intersection contract.
            test_brain_areas = _collect_split_brain_areas(split_datasets["test"])
            if not test_brain_areas:
                raise ValueError(
                    "Region-intersection pooling requires non-empty test-side brain-area "
                    "metadata to compute train/test intersection."
                )
            fold_context["test_brain_areas"] = test_brain_areas
        if hasattr(preprocessor, "set_fold_context"):
            preprocessor.set_fold_context(fold_context)

        if preprocessor.requires_fit():
            fit_start = time.time()
            preprocess_state = preprocessor.fit_split(
                split_datasets["train"].iter_preprocess_fit_samples()
            )
            preprocessor.set_state(preprocess_state)
            log(
                f"Fold {fold_idx} preprocessor fit completed in "
                f"{time.time() - fit_start:.2f}s",
                priority=0,
            )
            consume_fit_cache = getattr(
                preprocessor, "consume_fit_transformed_samples", None
            )
            if callable(consume_fit_cache):
                cached_train_samples = consume_fit_cache()
                if cached_train_samples is not None:
                    # Reuse transformed train samples from fit pass to avoid
                    # a second identical transform traversal.
                    split_datasets["train"].set_materialized_samples(
                        cached_train_samples
                    )
                    log(
                        f"Fold {fold_idx} train split reused "
                        f"{len(cached_train_samples)} preprocessed samples from fit pass",
                        priority=0,
                    )
        else:
            preprocess_state = preprocessor.get_state()

        for split in ("train", "val", "test"):
            if split_datasets[split].is_materialized():
                log(
                    f"Fold {fold_idx} {split} split already materialized; "
                    "skipping duplicate preprocessing.",
                    priority=0,
                )
                continue
            split_start = time.time()
            split_datasets[split].materialize(
                preprocessor=preprocessor,
                profile_preprocessor=True,
                log_prefix=f"Fold {fold_idx} {split} split: ",
            )
            log(
                f"Fold {fold_idx} {split} split materialization + preprocessing "
                f"completed in {time.time() - split_start:.2f}s",
                priority=0,
            )
    else:
        for split in ("train", "val", "test"):
            split_start = time.time()
            split_datasets[split].materialize(preprocessor=None)
            log(
                f"Fold {fold_idx} {split} split materialization completed in "
                f"{time.time() - split_start:.2f}s",
                priority=0,
            )
    log_fold_split_sample_counts(
        split_datasets, fold_idx=fold_idx, phase="materialized"
    )

    metadata = {
        "task": dataset_cfg.task,
        "label_mode": dataset_cfg.label_mode,
        "regime": regime,
        "needs_region_intersection_pool": needs_pool,
        "dataset_provider": dataset_provider,
        "fold_seed": fold_seed,
        "brain_area_key": brain_area_key,
        "test_subject": dataset_cfg.test_subject,
        "test_session": dataset_cfg.test_session,
        "split_summaries": {
            split: split_datasets[split].get_split_summary()
            for split in ("train", "val", "test")
        },
    }

    fold = {
        "fold_idx": fold_idx,
        "train_split": split_datasets["train"],
        "val_split": split_datasets["val"],
        "test_split": split_datasets["test"],
        "preprocess_state": preprocess_state,
        "metadata": metadata,
    }
    validate_fold_dict(fold)
    return fold
