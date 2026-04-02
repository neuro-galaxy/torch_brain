"""
Base preprocessor interface that all preprocessors must implement.
"""

from abc import ABC
from typing import Any, Iterable, Literal


class BasePreprocessor(ABC):
    """Base class for all preprocessors."""

    # Default execution category for staged variable-channel preprocessing.
    execution_type: Literal["sample_local", "fold_fit_transform"] = "sample_local"

    def __init__(self, cfg):
        """
        Initialize preprocessor with configuration.

        Args:
            cfg: Preprocessor configuration (DictConfig from Hydra)
        """
        self.cfg = cfg

    def set_allowed_electrodes(self, electrode_labels):
        """Optional hook for preprocessors that support electrode filtering."""
        # Default implementation is a no-op
        return

    def reset_state(self):
        """
        Reset any stateful components (e.g., fitted scalers).
        Called before processing each fold to ensure clean state.
        """
        # Default: no-op for stateless preprocessors
        pass

    def requires_fit(self) -> bool:
        """Whether this preprocessor needs a fold-fit pass before transform."""
        return self.execution_type == "fold_fit_transform"

    def fit_split(self, sample_iter: Iterable[dict[str, Any]]) -> dict | None:
        """Optional fold-fit hook for sample-dict pipelines."""
        return None

    def set_fold_context(self, context: dict[str, Any] | None) -> None:
        """Optional hook to inject fold-specific metadata before fit_split."""
        _ = context
        return

    def set_state(self, state: dict | None) -> None:
        """Install previously fitted state (no-op by default)."""
        return

    def get_state(self) -> dict | None:
        """Export fitted state (no-op by default)."""
        return None

    def transform_samples(
        self, samples: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Transform a sequence of sample dicts.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.transform_samples(...) is not implemented."
        )
