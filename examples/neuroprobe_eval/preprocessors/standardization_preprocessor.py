"""Standardization preprocessor for variable-channel pipelines."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("standardize")
class StandardizationPreprocessor(BasePreprocessor):
    """
    Preprocessor that standardizes sample dictionaries with configurable modes.

    Modes:
        - 'global_feature': one shared feature-wise mean/std across all channels
          observed in train split.
        - 'per_channel_feature': feature-wise mean/std tracked per channel_id,
          with optional unseen-channel fallback policy.
        - 'per_channel_samples_time_pooled': per-channel-id stats pooled across
          sample/time observations.
    """

    VARIABLE_MODES = {
        "global_feature",
        "per_channel_feature",
        "per_channel_samples_time_pooled",
    }
    CHANNEL_ID_VARIABLE_MODES = {
        "per_channel_feature",
        "per_channel_samples_time_pooled",
    }
    VALID_MODES = VARIABLE_MODES

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mode = getattr(cfg, "mode", "per_channel_feature")
        self._state = None
        self.eps = float(getattr(cfg, "eps", 1e-8))
        self.unseen_channel_policy = str(
            getattr(cfg, "unseen_channel_policy", "global_fallback")
        )

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {sorted(self.VALID_MODES)}"
            )

        if self.mode in self.CHANNEL_ID_VARIABLE_MODES:
            valid_policies = {"global_fallback", "error"}
            if self.unseen_channel_policy not in valid_policies:
                raise ValueError(
                    "Invalid unseen_channel_policy "
                    f"'{self.unseen_channel_policy}'. Expected one of {sorted(valid_policies)}."
                )

        self.execution_type = "fold_fit_transform"

    def fit_split(self, sample_iter):
        state = self._new_variable_state()
        saw_sample = False
        for sample in sample_iter:
            saw_sample = True
            if self.mode == "global_feature":
                x_matrix, feature_shape = self._extract_feature_matrix(sample)
                self._ensure_feature_shape(state, feature_shape)
                self._update_stats_matrix(state["global"], x_matrix)
                continue

            if self.mode == "per_channel_feature":
                x_matrix, feature_shape = self._extract_feature_matrix(sample)
                self._ensure_feature_shape(state, feature_shape)
                channel_ids = self._require_channel_ids(
                    sample, expected_n=x_matrix.shape[0]
                )
                for channel_id, row in zip(channel_ids, x_matrix):
                    ch_key = str(channel_id)
                    ch_stats = state["per_channel"].setdefault(
                        ch_key,
                        self._new_stats_accumulator(),
                    )
                    self._update_stats_row(ch_stats, row)
                    self._update_stats_row(state["global"], row)
                continue

            if self.mode == "per_channel_samples_time_pooled":
                x_matrix, feature_shape = self._extract_channel_time_feature_matrix(
                    sample
                )
                self._ensure_feature_shape(state, feature_shape)
                channel_ids = self._require_channel_ids(
                    sample, expected_n=x_matrix.shape[0]
                )
                for channel_id, channel_matrix in zip(channel_ids, x_matrix):
                    ch_key = str(channel_id)
                    ch_stats = state["per_channel"].setdefault(
                        ch_key,
                        self._new_stats_accumulator(),
                    )
                    self._update_stats_matrix(ch_stats, channel_matrix)
                    self._update_stats_matrix(state["global"], channel_matrix)
                continue

            raise ValueError(f"Unsupported variable mode: {self.mode}")

        if not saw_sample:
            raise ValueError("fit_split received zero samples.")

        if state["global"]["count"] == 0:
            raise ValueError("fit_split produced empty statistics.")

        self._state = state
        return self.get_state()

    def set_state(self, state):
        if state is None:
            self._state = None
            return
        decoded = self._decode_state(state)
        # Loaded state is authoritative for transform behavior.
        self._state = decoded
        self.eps = float(decoded["eps"])
        self.unseen_channel_policy = str(decoded["unseen_channel_policy"])

    def get_state(self):
        if self._state is None:
            return None
        return self._encode_state(self._state)

    def _transform_one(self, sample):
        if self._state is None:
            raise RuntimeError(
                "Variable-channel standardization state is unset. "
                "Call fit_split(...) then set_state(...)/use fitted instance."
            )

        if self.mode == "global_feature":
            x_matrix, feature_shape = self._extract_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            mean, std = self._stats_mean_std(self._state["global"])
            normalized = (x_matrix - mean[None, :]) / std[None, :]
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        if self.mode == "per_channel_feature":
            x_matrix, feature_shape = self._extract_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            channel_ids = self._require_channel_ids(
                sample, expected_n=x_matrix.shape[0]
            )
            normalized = np.empty_like(x_matrix, dtype=np.float32)
            global_mean, global_std = self._stats_mean_std(self._state["global"])
            for i, channel_id in enumerate(channel_ids):
                ch_stats = self._state["per_channel"].get(str(channel_id))
                if ch_stats is None:
                    if self.unseen_channel_policy == "error":
                        raise KeyError(
                            "Unseen channel_id during eval: "
                            f"'{channel_id}'. Set unseen_channel_policy=global_fallback to allow fallback."
                        )
                    mean, std = global_mean, global_std
                else:
                    mean, std = self._stats_mean_std(ch_stats)
                normalized[i] = (x_matrix[i] - mean) / std
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        if self.mode == "per_channel_samples_time_pooled":
            x_matrix, feature_shape = self._extract_channel_time_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            channel_ids = self._require_channel_ids(
                sample, expected_n=x_matrix.shape[0]
            )
            normalized = np.empty_like(x_matrix, dtype=np.float32)
            global_mean, global_std = self._stats_mean_std(self._state["global"])
            for i, channel_id in enumerate(channel_ids):
                ch_stats = self._state["per_channel"].get(str(channel_id))
                if ch_stats is None:
                    if self.unseen_channel_policy == "error":
                        raise KeyError(
                            "Unseen channel_id during eval: "
                            f"'{channel_id}'. Set unseen_channel_policy=global_fallback to allow fallback."
                        )
                    mean, std = global_mean, global_std
                else:
                    mean, std = self._stats_mean_std(ch_stats)
                normalized[i] = (x_matrix[i] - mean[None, :]) / std[None, :]
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        raise ValueError(f"Unsupported variable mode: {self.mode}")

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]

    def reset_state(self):
        """Reset fitted variable-channel state."""
        self._state = None

    @staticmethod
    def _new_stats_accumulator():
        return {"count": 0, "mean": None, "m2": None}

    def _new_variable_state(self):
        return {
            "mode": self.mode,
            "feature_shape": None,
            "global": self._new_stats_accumulator(),
            "per_channel": {},
            "eps": self.eps,
            "unseen_channel_policy": self.unseen_channel_policy,
        }

    @staticmethod
    def _extract_feature_matrix(sample):
        if "x" not in sample:
            raise KeyError("sample dict must contain key 'x'.")
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                f"sample['x'] must be at least 2D (channels, features...), got {x.shape}."
            )
        channels = x.shape[0]
        feature_shape = tuple(x.shape[1:])
        return x.reshape(channels, -1), feature_shape

    @staticmethod
    def _extract_channel_time_feature_matrix(sample):
        if "x" not in sample:
            raise KeyError("sample dict must contain key 'x'.")
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                "sample['x'] must be at least 2D (channels, time, ...), "
                f"got {x.shape}."
            )
        channels = int(x.shape[0])
        time_steps = int(x.shape[1])
        pooled_feature_shape = tuple(x.shape[2:]) or (1,)
        return x.reshape(channels, time_steps, -1), pooled_feature_shape

    @staticmethod
    def _require_channel_ids(sample, *, expected_n):
        channel_ids = sample.get("channel_ids")
        if channel_ids is None:
            raise KeyError("Channel-aware mode requires sample['channel_ids'].")
        if len(channel_ids) != expected_n:
            raise ValueError(
                "channel_ids length must match x channels: "
                f"{len(channel_ids)} vs {expected_n}."
            )
        return channel_ids

    @staticmethod
    def _ensure_feature_shape(state, feature_shape):
        if state["feature_shape"] is None:
            state["feature_shape"] = tuple(feature_shape)
            return
        if tuple(state["feature_shape"]) != tuple(feature_shape):
            raise ValueError(
                "Feature shape mismatch across samples: "
                f"expected {tuple(state['feature_shape'])}, got {tuple(feature_shape)}."
            )

    def _update_stats_matrix(self, acc, matrix):
        for row in matrix:
            self._update_stats_row(acc, row)

    @staticmethod
    def _update_stats_row(acc, row):
        row = np.asarray(row, dtype=np.float64)
        if acc["mean"] is None:
            acc["mean"] = np.zeros_like(row, dtype=np.float64)
            acc["m2"] = np.zeros_like(row, dtype=np.float64)
        acc["count"] += 1
        delta = row - acc["mean"]
        acc["mean"] += delta / acc["count"]
        delta2 = row - acc["mean"]
        acc["m2"] += delta * delta2

    def _stats_mean_std(self, acc):
        count = int(acc["count"])
        mean = np.asarray(acc["mean"], dtype=np.float64)
        if count <= 1:
            std = np.ones_like(mean, dtype=np.float64)
            return mean.astype(np.float32), std.astype(np.float32)
        m2 = np.asarray(acc["m2"], dtype=np.float64)
        # Match sklearn StandardScaler variance convention (population variance, ddof=0).
        var = m2 / float(max(count, 1))
        std = np.sqrt(np.maximum(var, 0.0))
        if self.eps > 0.0:
            std = np.maximum(std, float(self.eps))
        std[std == 0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _encode_acc(acc):
        mean = acc["mean"]
        m2 = acc["m2"]
        if mean is None:
            return {"count": int(acc["count"]), "mean": [], "m2": []}
        return {
            "count": int(acc["count"]),
            "mean": np.asarray(mean, dtype=np.float32).tolist(),
            "m2": np.asarray(m2, dtype=np.float32).tolist(),
        }

    @staticmethod
    def _decode_acc(encoded):
        count = int(encoded.get("count", 0))
        mean_list = encoded.get("mean", [])
        m2_list = encoded.get("m2", [])
        if count <= 0 or len(mean_list) == 0:
            return {"count": 0, "mean": None, "m2": None}
        mean = np.asarray(mean_list, dtype=np.float64)
        m2 = np.asarray(m2_list, dtype=np.float64)
        if mean.shape != m2.shape:
            raise ValueError("Encoded stats mean/m2 shape mismatch.")
        return {"count": count, "mean": mean, "m2": m2}

    def _encode_state(self, state):
        return {
            "mode": state["mode"],
            "feature_shape": list(state["feature_shape"]),
            "global": self._encode_acc(state["global"]),
            "per_channel": {
                key: self._encode_acc(acc) for key, acc in state["per_channel"].items()
            },
            "eps": float(state["eps"]),
            "unseen_channel_policy": state["unseen_channel_policy"],
        }

    def _decode_state(self, state):
        mode = state.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(
                "Standardization state mode mismatch: "
                f"state='{mode}', instance='{self.mode}'."
            )
        feature_shape = tuple(state.get("feature_shape", ()))
        if not feature_shape:
            raise ValueError("Standardization state missing feature_shape.")
        decoded = {
            "mode": mode,
            "feature_shape": feature_shape,
            "global": self._decode_acc(state["global"]),
            "per_channel": {
                str(key): self._decode_acc(acc)
                for key, acc in state.get("per_channel", {}).items()
            },
            "eps": float(state.get("eps", self.eps)),
            "unseen_channel_policy": str(
                state.get("unseen_channel_policy", self.unseen_channel_policy)
            ),
        }
        return deepcopy(decoded)
