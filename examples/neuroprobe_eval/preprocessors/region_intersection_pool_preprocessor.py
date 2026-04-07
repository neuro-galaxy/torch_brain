"""Region-intersection pooling for multi-subject aligned-channel folds."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


def _normalize_region_labels(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        raise ValueError("brain-area metadata is empty.")
    out: list[str] = []
    for value in arr:
        if value is None:
            raise ValueError("brain-area metadata contains None labels.")
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            raise ValueError("brain-area metadata contains NaN labels.")
        label = str(value).strip()
        if not label:
            raise ValueError("brain-area metadata contains empty labels.")
        out.append(label)
    return np.asarray(out, dtype=object)


@register_preprocessor("region_intersection_pool")
class RegionIntersectionPoolPreprocessor(BasePreprocessor):
    """Mean-pool channels by common train/test brain regions."""

    execution_type = "fold_fit_transform"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._common_regions: np.ndarray | None = None
        self._test_regions: np.ndarray | None = None

    def set_fold_context(self, context: dict[str, Any] | None) -> None:
        if context is None:
            self._test_regions = None
            return
        test_regions = context.get("test_brain_areas")
        if test_regions is None:
            self._test_regions = None
            return
        self._test_regions = _normalize_region_labels(test_regions)

    def fit_split(self, sample_iter: Iterable[dict[str, Any]]) -> dict[str, Any]:
        if self._test_regions is None:
            raise RuntimeError(
                "region_intersection_pool requires fold context with test_brain_areas "
                "before fit_split(...)."
            )

        # Build the region set that is present in every train sample so the
        # fitted state is always transformable per sample.
        train_regions_intersection: set[str] | None = None
        saw_sample = False
        for sample in sample_iter:
            saw_sample = True
            brain_areas = sample.get("brain_areas")
            if brain_areas is None:
                raise ValueError(
                    "region_intersection_pool requires sample['brain_areas'] metadata."
                )
            labels = _normalize_region_labels(brain_areas)
            sample_regions = {str(label) for label in labels.tolist()}
            if train_regions_intersection is None:
                train_regions_intersection = sample_regions
            else:
                train_regions_intersection.intersection_update(sample_regions)
                if not train_regions_intersection:
                    break

        if not saw_sample:
            raise ValueError(
                "region_intersection_pool fit_split received zero samples."
            )
        if not train_regions_intersection:
            raise ValueError(
                "region_intersection_pool found empty train-side region intersection "
                "across fit samples."
            )

        common_regions = np.intersect1d(
            np.asarray(sorted(train_regions_intersection), dtype=object),
            np.asarray(self._test_regions, dtype=object),
        )
        if common_regions.size == 0:
            raise ValueError(
                "region_intersection_pool found empty intersection between "
                "train-side common regions and test-side regions."
            )

        self._common_regions = np.asarray(common_regions, dtype=object)
        return self.get_state() or {"common_regions": []}

    def set_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            self._common_regions = None
            return
        common = state.get("common_regions")
        self._common_regions = _normalize_region_labels(common)

    def get_state(self) -> dict[str, Any] | None:
        if self._common_regions is None:
            return None
        return {"common_regions": [str(x) for x in self._common_regions.tolist()]}

    def _transform_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._common_regions is None:
            raise RuntimeError(
                "region_intersection_pool state is unset. Run fit_split(...) first."
            )

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                f"sample['x'] must be at least 2D (channels,...), got shape {x.shape}."
            )
        n_channels = int(x.shape[0])

        brain_areas = sample.get("brain_areas")
        if brain_areas is None:
            raise ValueError(
                "region_intersection_pool requires sample['brain_areas'] metadata."
            )
        labels = _normalize_region_labels(brain_areas)
        if len(labels) != n_channels:
            raise ValueError(
                "sample['brain_areas'] length must match x.shape[0], got "
                f"{len(labels)} vs {n_channels}."
            )

        coords = sample.get("channel_coords_lip")
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.shape != (n_channels, 3):
                raise ValueError(
                    "sample['channel_coords_lip'] must have shape "
                    f"({n_channels}, 3), got {coords.shape}."
                )

        pooled_x = []
        pooled_coords = [] if coords is not None else None
        for region in self._common_regions:
            idx = np.flatnonzero(labels == region)
            if idx.size == 0:
                raise ValueError(
                    f"Sample missing common region '{region}' required by fit state."
                )
            pooled_x.append(x[idx].mean(axis=0))
            if pooled_coords is not None:
                pooled_coords.append(coords[idx].mean(axis=0))

        out = dict(sample)
        out["x"] = np.asarray(pooled_x, dtype=np.float32)
        region_labels = [str(region) for region in self._common_regions.tolist()]
        # Region pooling emits one channel per common region; keep ids/names
        # synchronized to the pooled region labels for downstream alignment.
        out["channel_ids"] = list(region_labels)
        out["channel_names"] = list(region_labels)
        out["brain_areas"] = list(region_labels)
        if pooled_coords is None:
            out["channel_coords_lip"] = None
        else:
            out["channel_coords_lip"] = np.asarray(pooled_coords, dtype=np.float32)
        out["seq_id"] = np.zeros((len(region_labels),), dtype=np.int64)
        return out

    def transform_samples(
        self, samples: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [self._transform_one(sample) for sample in samples]
