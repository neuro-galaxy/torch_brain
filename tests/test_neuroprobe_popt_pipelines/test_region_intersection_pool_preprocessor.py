import numpy as np
import pytest
from omegaconf import DictConfig

from neuroprobe_eval.preprocessors.region_intersection_pool_preprocessor import (
    RegionIntersectionPoolPreprocessor,
)


def _sample(x, brain_areas):
    n_channels = len(brain_areas)
    return {
        "x": np.asarray(x, dtype=np.float32),
        "brain_areas": list(brain_areas),
        "channel_ids": [f"ch{i}" for i in range(n_channels)],
        "channel_coords_lip": np.zeros((n_channels, 3), dtype=np.float32),
        "seq_id": np.zeros((n_channels,), dtype=np.int64),
        "recording_id": "r",
        "split": "train",
        "sample_idx": 0,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
        "y": 1,
    }


def test_region_intersection_pool_rejects_invalid_brain_area_labels():
    pre = RegionIntersectionPoolPreprocessor(
        DictConfig({"name": "region_intersection_pool"})
    )
    pre.set_fold_context({"test_brain_areas": ["A"]})
    with pytest.raises(ValueError, match="empty labels"):
        pre.fit_split(
            iter(
                [
                    _sample(
                        x=[[1.0, 2.0]],
                        brain_areas=[""],
                    )
                ]
            )
        )


def test_region_intersection_pool_rejects_missing_brain_areas_on_transform():
    pre = RegionIntersectionPoolPreprocessor(
        DictConfig({"name": "region_intersection_pool"})
    )
    pre.set_fold_context({"test_brain_areas": ["A"]})
    pre.fit_split(iter([_sample(x=[[1.0, 2.0]], brain_areas=["A"])]))
    bad = _sample(x=[[1.0, 2.0]], brain_areas=["A"])
    bad["brain_areas"] = None
    with pytest.raises(ValueError, match="requires sample\\['brain_areas'\\]"):
        pre.transform_samples([bad])[0]


def test_region_intersection_pool_rejects_nan_region_labels():
    pre = RegionIntersectionPoolPreprocessor(
        DictConfig({"name": "region_intersection_pool"})
    )
    with pytest.raises(ValueError, match="NaN labels"):
        pre.set_fold_context({"test_brain_areas": [np.nan]})


def test_region_intersection_pool_enforces_common_region_presence_per_sample():
    pre = RegionIntersectionPoolPreprocessor(
        DictConfig({"name": "region_intersection_pool"})
    )
    pre.set_fold_context({"test_brain_areas": ["region_b"]})
    train_samples = [
        _sample(x=[[1.0, 2.0], [3.0, 4.0]], brain_areas=["region_a", "region_b"]),
        _sample(x=[[5.0, 6.0]], brain_areas=["region_a"]),
    ]
    pre.fit_split(iter(train_samples))

    with pytest.raises(ValueError, match="missing common region 'region_b'"):
        pre.transform_samples(train_samples)
