import os

import numpy as np
import pytest
from omegaconf import DictConfig

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.preprocessors import CompositePreprocessor
from neuroprobe_eval.preprocessors.base_preprocessor import BasePreprocessor
from neuroprobe_eval.preprocessors.standardization_preprocessor import (
    StandardizationPreprocessor,
)


def _sample(x, channel_ids, **extra):
    sample = {
        "x": np.asarray(x, dtype=np.float32),
        "channel_ids": list(channel_ids),
        "recording_id": "rec_0",
        "split": "train",
        "sample_idx": 0,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
        "y": 1,
    }
    sample.update(extra)
    return sample


def _fit_samples():
    return [
        _sample([[1.0, 3.0], [2.0, 6.0]], ["a", "b"], sample_idx=1),
        _sample([[3.0, 9.0]], ["a"], sample_idx=2),
    ]


@pytest.mark.parametrize("mode", ["per_feature", "samples_time_pooled"])
def test_legacy_dense_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Invalid mode"):
        StandardizationPreprocessor(DictConfig({"mode": mode}))


def test_global_feature_fit_transform_and_state_round_trip():
    cfg = DictConfig({"mode": "global_feature"})
    pre = StandardizationPreprocessor(cfg)
    fit_samples = _fit_samples()

    state = pre.fit_split(iter(fit_samples))
    assert state is not None

    eval_sample = _sample([[2.0, 6.0], [4.0, 12.0]], ["a", "c"], split="eval")
    transformed = pre.transform_samples([eval_sample])[0]

    all_rows = np.vstack([s["x"] for s in fit_samples]).astype(np.float32)
    mean = all_rows.mean(axis=0)
    std = all_rows.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected = (eval_sample["x"] - mean[None, :]) / std[None, :]

    np.testing.assert_allclose(transformed["x"], expected.astype(np.float32), atol=1e-6)
    assert transformed["recording_id"] == eval_sample["recording_id"]
    assert transformed["split"] == "eval"
    assert transformed["channel_ids"] == eval_sample["channel_ids"]

    pre2 = StandardizationPreprocessor(cfg)
    pre2.set_state(state)
    transformed2 = pre2.transform_samples([eval_sample])[0]
    np.testing.assert_allclose(transformed2["x"], transformed["x"], atol=1e-6)


def test_per_channel_feature_keeps_separate_stats_and_unseen_falls_back_to_global():
    cfg = DictConfig(
        {"mode": "per_channel_feature", "unseen_channel_policy": "global_fallback"}
    )
    pre = StandardizationPreprocessor(cfg)
    fit_samples = _fit_samples()
    pre.fit_split(iter(fit_samples))

    a_rows = np.array([[1.0, 3.0], [3.0, 9.0]], dtype=np.float32)
    a_mean = a_rows.mean(axis=0)
    a_std = a_rows.std(axis=0, ddof=0)
    a_std[a_std == 0] = 1.0

    all_rows = np.array([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0]], dtype=np.float32)
    global_mean = all_rows.mean(axis=0)
    global_std = all_rows.std(axis=0, ddof=0)
    global_std[global_std == 0] = 1.0

    eval_sample = _sample([[5.0, 15.0], [6.0, 18.0]], ["a", "z"], split="eval")
    transformed = pre.transform_samples([eval_sample])[0]

    expected_a = (np.array([5.0, 15.0]) - a_mean) / a_std
    np.testing.assert_allclose(transformed["x"][0], expected_a, atol=1e-5)

    expected_z = (np.array([6.0, 18.0]) - global_mean) / global_std
    np.testing.assert_allclose(transformed["x"][1], expected_z, atol=1e-5)


def test_per_channel_feature_unseen_error():
    cfg = DictConfig({"mode": "per_channel_feature", "unseen_channel_policy": "error"})
    pre = StandardizationPreprocessor(cfg)
    pre.fit_split(iter([_sample([[1.0, 2.0]], ["a"])]))

    with pytest.raises(KeyError, match="Unseen channel_id"):
        pre.transform_samples([_sample([[3.0, 6.0]], ["missing"], split="eval")])[0]


def test_set_state_get_state_round_trip():
    cfg = DictConfig(
        {"mode": "per_channel_feature", "unseen_channel_policy": "global_fallback"}
    )
    pre = StandardizationPreprocessor(cfg)
    pre.fit_split(iter(_fit_samples()))
    state = pre.get_state()

    pre_copy = StandardizationPreprocessor(cfg)
    pre_copy.set_state(state)

    eval_sample = _sample([[7.0, 21.0], [8.0, 24.0]], ["a", "b"], split="eval")
    out_a = pre.transform_samples([eval_sample])[0]
    out_b = pre_copy.transform_samples([eval_sample])[0]
    np.testing.assert_allclose(out_a["x"], out_b["x"], atol=1e-6)


def test_set_state_uses_state_policy_and_eps_over_local_config():
    fit_cfg = DictConfig(
        {"mode": "per_channel_feature", "unseen_channel_policy": "error", "eps": 1e-2}
    )
    fitted = StandardizationPreprocessor(fit_cfg)
    fitted.fit_split(iter(_fit_samples()))
    state = fitted.get_state()

    load_cfg = DictConfig(
        {
            "mode": "per_channel_feature",
            "unseen_channel_policy": "global_fallback",
            "eps": 1e-8,
        }
    )
    loaded = StandardizationPreprocessor(load_cfg)
    loaded.set_state(state)

    with pytest.raises(KeyError, match="Unseen channel_id"):
        loaded.transform_samples([_sample([[10.0, 20.0]], ["unseen"], split="eval")])[0]

    assert loaded.eps == pytest.approx(1e-2)


def test_transform_sample_raises_before_fit():
    pre = StandardizationPreprocessor(DictConfig({"mode": "per_channel_feature"}))
    sample = _sample([[1.0, 2.0], [3.0, 4.0]], ["a", "b"], split="eval")

    with pytest.raises(RuntimeError, match="state is unset"):
        pre.transform_samples([sample])[0]


def test_reset_state_clears_variable_channel_state():
    for mode in ("global_feature", "per_channel_feature"):
        cfg = DictConfig({"mode": mode, "unseen_channel_policy": "global_fallback"})
        pre = StandardizationPreprocessor(cfg)
        pre.fit_split(iter(_fit_samples()))

        s = _sample([[1.0, 2.0]], ["a"], split="eval")
        pre.transform_samples([s])[0]

        pre.reset_state()
        with pytest.raises(RuntimeError, match="state is unset"):
            pre.transform_samples([s])[0]


def test_global_feature_matches_manual_zscore():
    rng = np.random.default_rng(42)
    n_samples, n_channels, n_features = 20, 4, 8
    data = rng.standard_normal((n_samples, n_channels, n_features)).astype(np.float32)
    labels = [f"ch{i}" for i in range(n_channels)]

    pre = StandardizationPreprocessor(DictConfig({"mode": "global_feature"}))
    samples = [_sample(data[i], labels, sample_idx=i) for i in range(n_samples)]
    pre.fit_split(iter(samples))
    out = np.stack([pre.transform_samples([s])[0]["x"] for s in samples], axis=0)

    pooled = data.reshape(n_samples * n_channels, n_features)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected = (data - mean[None, None, :]) / std[None, None, :]
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_per_channel_feature_matches_manual():
    rng = np.random.default_rng(99)
    n_samples, n_channels, n_features = 15, 3, 6
    data = rng.standard_normal((n_samples, n_channels, n_features)).astype(np.float32)
    labels = [f"ch{i}" for i in range(n_channels)]

    cfg = DictConfig(
        {"mode": "per_channel_feature", "unseen_channel_policy": "global_fallback"}
    )
    pre = StandardizationPreprocessor(cfg)
    samples = [_sample(data[i], labels, sample_idx=i) for i in range(n_samples)]
    pre.fit_split(iter(samples))

    vc_out = np.stack([pre.transform_samples([s])[0]["x"] for s in samples], axis=0)

    for ch_idx in range(n_channels):
        ch_data = data[:, ch_idx, :]
        ch_mean = ch_data.mean(axis=0)
        ch_std = ch_data.std(axis=0, ddof=0)
        ch_std[ch_std == 0] = 1.0
        expected = (ch_data - ch_mean[None, :]) / ch_std[None, :]
        np.testing.assert_allclose(
            vc_out[:, ch_idx, :],
            expected,
            atol=1e-4,
            err_msg=f"Channel {labels[ch_idx]} normalization mismatch",
        )


def test_per_channel_samples_time_pooled_matches_manual():
    rng = np.random.default_rng(321)
    n_samples, n_channels, n_time, n_freq = 12, 4, 9, 6
    data = rng.standard_normal((n_samples, n_channels, n_time, n_freq)).astype(
        np.float32
    )
    labels = [f"ch{i}" for i in range(n_channels)]

    pre = StandardizationPreprocessor(
        DictConfig(
            {
                "mode": "per_channel_samples_time_pooled",
                "unseen_channel_policy": "global_fallback",
            }
        )
    )
    samples = [_sample(data[i], labels, sample_idx=i) for i in range(n_samples)]
    pre.fit_split(iter(samples))
    out = np.stack([pre.transform_samples([s])[0]["x"] for s in samples], axis=0)

    expected = np.empty_like(data, dtype=np.float32)
    for ch_idx in range(n_channels):
        pooled = data[:, ch_idx, :, :].reshape(n_samples * n_time, n_freq)
        ch_mean = pooled.mean(axis=0)
        ch_std = pooled.std(axis=0, ddof=0)
        ch_std[ch_std == 0] = 1.0
        expected[:, ch_idx, :, :] = (
            data[:, ch_idx, :, :] - ch_mean[None, None, :]
        ) / ch_std[None, None, :]

    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)


class _SampleScalePreprocessor(BasePreprocessor):
    execution_type = "sample_local"

    def transform_samples(self, samples):
        out_samples = []
        for sample in samples:
            out = dict(sample)
            out["x"] = np.asarray(sample["x"], dtype=np.float32) * 2.0
            out_samples.append(out)
        return out_samples


class _SampleMeanCenterPreprocessor(BasePreprocessor):
    execution_type = "fold_fit_transform"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._mean = None

    def fit_split(self, sample_iter):
        rows = [
            np.asarray(sample["x"], dtype=np.float32).reshape(-1, sample["x"].shape[-1])
            for sample in sample_iter
        ]
        stacked = np.vstack(rows)
        self._mean = stacked.mean(axis=0)
        return {"mean": self._mean.tolist()}

    def set_state(self, state):
        if state is None:
            self._mean = None
        else:
            self._mean = np.asarray(state["mean"], dtype=np.float32)

    def get_state(self):
        if self._mean is None:
            return None
        return {"mean": self._mean.tolist()}

    def transform_samples(self, samples):
        if self._mean is None:
            raise RuntimeError("Mean state is not set.")
        out_samples = []
        for sample in samples:
            out = dict(sample)
            out["x"] = np.asarray(sample["x"], dtype=np.float32) - self._mean
            out_samples.append(out)
        return out_samples


def test_composite_preprocessor_fit_split_and_transform_sample_chain():
    cfg = DictConfig({"name": "composite"})
    composite = CompositePreprocessor(
        cfg,
        [_SampleScalePreprocessor(cfg), _SampleMeanCenterPreprocessor(cfg)],
    )
    assert composite.requires_fit() is True

    samples = [
        _sample([[1.0, 2.0], [3.0, 4.0]], ["a", "b"]),
        _sample([[5.0, 6.0]], ["a"]),
    ]
    state = composite.fit_split(iter(samples))
    assert isinstance(state, list)
    assert len(state) == 2

    eval_sample = _sample([[2.0, 4.0]], ["a"], split="eval")
    out = composite.transform_samples([eval_sample])[0]
    assert out["x"].shape == eval_sample["x"].shape
    expected = np.array([[-2.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(out["x"], expected, atol=1e-6)


def test_composite_preprocessor_unload_model_forwards_to_nested_stages():
    cfg = DictConfig({"name": "composite"})
    unloaded = []

    class _UnloadSpy(BasePreprocessor):
        def __init__(self, cfg, tag):
            super().__init__(cfg)
            self.tag = tag

        def transform_samples(self, samples):
            return list(samples)

        def unload_model(self):
            unloaded.append(self.tag)

    inner = CompositePreprocessor(cfg, [_UnloadSpy(cfg, "inner")])
    outer = CompositePreprocessor(cfg, [_UnloadSpy(cfg, "outer"), inner])

    outer.unload_model()

    assert unloaded == ["outer", "inner"]
