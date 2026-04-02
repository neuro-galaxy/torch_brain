"""Tests for the variable-channel-only StandardizationPreprocessor contract."""

import numpy as np
import pytest
from omegaconf import DictConfig

from neuroprobe_eval.preprocessors.standardization_preprocessor import (
    StandardizationPreprocessor,
)


def _sample(x, channel_ids):
    return {
        "x": np.asarray(x, dtype=np.float32),
        "channel_ids": list(channel_ids),
        "recording_id": "rec_0",
        "split": "train",
        "sample_idx": 0,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
        "y": 1,
    }


@pytest.mark.parametrize(
    "mode",
    ["global_feature", "per_channel_feature", "per_channel_samples_time_pooled"],
)
def test_mode_validation_accepts_variable_modes(mode):
    pre = StandardizationPreprocessor(DictConfig({"mode": mode}))
    assert pre.execution_type == "fold_fit_transform"


@pytest.mark.parametrize("mode", ["per_feature", "samples_time_pooled", "invalid_mode"])
def test_mode_validation_rejects_unsupported_modes(mode):
    with pytest.raises(ValueError, match="Invalid mode"):
        StandardizationPreprocessor(DictConfig({"mode": mode}))


def test_dense_preprocess_entrypoints_are_removed():
    pre = StandardizationPreprocessor(DictConfig({"mode": "per_channel_feature"}))
    assert not hasattr(pre, "preprocess")
    assert not hasattr(pre, "preprocess_batch")


def test_transform_sample_requires_fitted_state():
    pre = StandardizationPreprocessor(DictConfig({"mode": "global_feature"}))
    sample = _sample([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])

    with pytest.raises(RuntimeError, match="state is unset"):
        pre.transform_samples([sample])[0]


def test_reset_state_clears_fitted_state():
    pre = StandardizationPreprocessor(DictConfig({"mode": "global_feature"}))
    fit_samples = [_sample([[1.0, 2.0]], ["a"]), _sample([[3.0, 4.0]], ["b"])]
    pre.fit_split(iter(fit_samples))
    pre.transform_samples([_sample([[2.0, 3.0]], ["a"])])[0]

    pre.reset_state()

    with pytest.raises(RuntimeError, match="state is unset"):
        pre.transform_samples([_sample([[2.0, 3.0]], ["a"])])[0]
