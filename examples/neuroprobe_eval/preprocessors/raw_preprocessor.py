"""
Raw preprocessor - no preprocessing applied.
"""
from omegaconf import DictConfig
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("raw")
class RawPreprocessor(BasePreprocessor):
    """Preprocessor that returns data as-is (no preprocessing)."""
    
    def preprocess(self, data, electrode_labels):
        """
        Return data unchanged.
        
        Args:
            data: Input data (torch.Tensor or numpy.ndarray)
            electrode_labels: List of electrode labels (unused)
        
        Returns:
            Same data unchanged
        """
        return data

