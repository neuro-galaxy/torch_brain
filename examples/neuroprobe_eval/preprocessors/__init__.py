"""
Preprocessor registry for automatic preprocessor discovery and instantiation.
"""

import importlib
import os
from copy import deepcopy
from pathlib import Path
from omegaconf import DictConfig, ListConfig

from .base_preprocessor import BasePreprocessor

PREPROCESSOR_REGISTRY = {}


def register_preprocessor(name):
    """Decorator to register a preprocessor class."""

    def register_preprocessor_cls(cls):
        if name in PREPROCESSOR_REGISTRY:
            raise ValueError(f"{name} already in registry")
        PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return register_preprocessor_cls


class CompositePreprocessor(BasePreprocessor):
    def __init__(self, cfg, preprocessors):
        super().__init__(cfg)
        self.preprocessors = preprocessors
        # Cache of fully transformed train samples produced during fit_split.
        # This lets variable-channel fold builders reuse train preprocessing work
        # instead of re-running the same chain during train materialization.
        self._fit_transformed_samples = None

    def set_allowed_electrodes(self, electrode_labels):
        for pre in self.preprocessors:
            if hasattr(pre, "set_allowed_electrodes"):
                pre.set_allowed_electrodes(electrode_labels)

    def reset_state(self):
        """Reset state for all preprocessors in the chain."""
        self._fit_transformed_samples = None
        for pre in self.preprocessors:
            if hasattr(pre, "reset_state"):
                pre.reset_state()

    def unload_model(self):
        """Forward model-unload hooks to all stages in the chain."""
        for pre in self.preprocessors:
            unload = getattr(pre, "unload_model", None)
            if callable(unload):
                unload()

    def requires_fit(self) -> bool:
        return any(pre.requires_fit() for pre in self.preprocessors)

    def set_fold_context(self, context):
        for pre in self.preprocessors:
            if hasattr(pre, "set_fold_context"):
                pre.set_fold_context(context)

    def fit_split(self, sample_iter):
        # Materialize once so fit preprocessors later in the chain can consume
        # outputs from earlier transforms without re-reading a one-shot iterator.
        working_samples = [deepcopy(sample) for sample in sample_iter]
        states = []
        for pre in self.preprocessors:
            state = None
            if pre.requires_fit():
                state = pre.fit_split(iter(working_samples))
                pre.set_state(state)
            states.append(pre.get_state() if state is None else state)
            # Run per-stage sample transforms through the batch-capable hook so
            # heavy stages (for example encoders) can process multiple samples
            # in one call while preserving the canonical sample-dict contract.
            working_samples = pre.transform_samples(working_samples)
        self._fit_transformed_samples = working_samples
        return states

    def consume_fit_transformed_samples(self):
        """Return and clear cached transformed train samples from fit_split."""
        samples = self._fit_transformed_samples
        self._fit_transformed_samples = None
        return samples

    def set_state(self, state):
        if state is None:
            for pre in self.preprocessors:
                pre.set_state(None)
            return

        if not isinstance(state, (list, tuple)):
            raise TypeError("CompositePreprocessor state must be a list/tuple.")
        if len(state) != len(self.preprocessors):
            raise ValueError(
                "CompositePreprocessor state length mismatch: "
                f"expected {len(self.preprocessors)}, got {len(state)}."
            )

        for pre, pre_state in zip(self.preprocessors, state):
            pre.set_state(pre_state)

    def get_state(self):
        return [pre.get_state() for pre in self.preprocessors]

    def transform_samples(self, samples):
        out_samples = list(samples)
        for pre in self.preprocessors:
            out_samples = pre.transform_samples(out_samples)
        return out_samples


def _build_single(cfg: DictConfig):
    """
    Build a single preprocessor instance.
    """
    preprocessor_name = cfg.name
    if preprocessor_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Preprocessor {preprocessor_name} not found in registry. "
            f"Available: {list(PREPROCESSOR_REGISTRY.keys())}"
        )

    preprocessor_class = PREPROCESSOR_REGISTRY[preprocessor_name]
    return preprocessor_class(cfg)


def build_preprocessor(cfg):
    """Build a preprocessor or chain of preprocessors from configuration."""
    if isinstance(cfg, ListConfig) or isinstance(cfg, list):
        preprocessors = [build_preprocessor(stage_cfg) for stage_cfg in cfg]
        return CompositePreprocessor(cfg, preprocessors)

    chain_cfgs = cfg.get("chain") if hasattr(cfg, "get") else None
    if chain_cfgs:
        preprocessors = [build_preprocessor(stage_cfg) for stage_cfg in chain_cfgs]
        return CompositePreprocessor(cfg, preprocessors)

    return _build_single(cfg)


def import_preprocessors():
    """Auto-import all preprocessor files to register them."""
    preprocessors_dir = os.path.dirname(__file__)
    for file in os.listdir(preprocessors_dir):
        if (
            file.endswith(".py")
            and not file.startswith("_")
            and file != "base_preprocessor.py"
        ):
            module_name = str(Path(file).with_suffix(""))
            importlib.import_module(f"{__name__}.{module_name}")


# Import all preprocessors to register them
import_preprocessors()
