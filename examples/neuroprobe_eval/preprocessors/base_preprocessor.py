"""
Base preprocessor interface that all preprocessors must implement.
"""
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Base class for all preprocessors."""
    
    def __init__(self, cfg):
        """
        Initialize preprocessor with configuration.
        
        Args:
            cfg: Preprocessor configuration (DictConfig from Hydra)
        """
        self.cfg = cfg
    
    @abstractmethod
    def preprocess(self, data, electrode_labels):
        """
        Apply preprocessing to neural data.
        
        Args:
            data: Input data (torch.Tensor or numpy.ndarray)
                Shape: (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
            electrode_labels: List of electrode labels
        
        Returns:
            Preprocessed data (same type as input)
        """
        pass

