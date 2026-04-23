"""
Time-domain filtering preprocessor (notch + optional bandpass).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("time_domain_filter")
class TimeDomainFilterPreprocessor(BasePreprocessor):
    """Apply notch filtering and optional high-gamma bandpass."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.freqs_to_filter = [60, 120, 180, 240, 300, 360]

    def _filter_array(self, array: np.ndarray) -> np.ndarray:
        sampling_rate = float(self.cfg.get("sampling_rate", 2048))
        high_gamma = bool(self.cfg.get("high_gamma", False))
        notch_q = float(self.cfg.get("notch_q", 30))
        bandpass_q = float(self.cfg.get("bandpass_q", 5))
        bandpass_low = float(self.cfg.get("bandpass_low", 70))
        bandpass_high = float(self.cfg.get("bandpass_high", 250))

        filtered = np.asarray(array, dtype=np.float32, order="C")
        nyquist = sampling_rate / 2.0
        for freq in self.freqs_to_filter:
            w0 = freq / nyquist
            if w0 >= 1.0:
                continue
            b, a = signal.iirnotch(w0, notch_q)
            filtered = signal.lfilter(b, a, filtered, axis=-1)

        if high_gamma:
            sos = signal.butter(
                bandpass_q,
                [bandpass_low, bandpass_high],
                btype="bandpass",
                analog=False,
                fs=sampling_rate,
                output="sos",
            )
            filtered = signal.sosfilt(sos, filtered, axis=-1)

        return filtered

    def _transform_one(self, sample):
        """Apply filtering to one sample dict while preserving aligned metadata."""
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("time_domain_filter requires sample['x'].")

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                "time_domain_filter expects sample['x'] to be at least 2D "
                f"(channels, time), got {x.shape}."
            )
        out = dict(sample)
        out["x"] = self._filter_array(x).astype(np.float32, copy=False)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
