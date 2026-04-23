"""
Raw preprocessor - no preprocessing applied.
"""

import numpy as np
import torch
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("raw")
class RawPreprocessor(BasePreprocessor):
    """Preprocessor that returns data as-is (no preprocessing)."""

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("raw preprocessor requires sample['x'].")
        out = dict(sample)
        x = sample["x"]
        if isinstance(x, torch.Tensor):
            out["x"] = x.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            out["x"] = np.asarray(x, dtype=np.float32)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
