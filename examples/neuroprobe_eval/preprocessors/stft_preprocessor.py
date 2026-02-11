"""
STFT (Short-Time Fourier Transform) preprocessor.
"""
import torch
import numpy as np
from omegaconf import DictConfig
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("stft")
class STFTPreprocessor(BasePreprocessor):
    """Preprocessor that applies STFT transform."""
    
    def preprocess(self, data, electrode_labels):
        """
        Apply STFT transform to neural data.
        
        Args:
            data: Input data (torch.Tensor or numpy.ndarray)
                Shape: (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
            electrode_labels: List of electrode labels (unused)
        
        Returns:
            Preprocessed data with shape (batch_size, n_electrodes, n_timebins, n_freqs)
        """
        was_tensor = isinstance(data, torch.Tensor)
        x = torch.from_numpy(data) if not was_tensor else data
        
        if len(x.shape) == 2:  # if it is only (n_electrodes, n_samples)
            x = x.unsqueeze(0)
        # data is of shape (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes, n_samples = x.shape
        
        # convert to float32 and reshape for STFT
        x = x.to(dtype=torch.float32)
        x = x.reshape(batch_size * n_electrodes, -1)
        
        # STFT parameters
        nperseg = self.cfg.get("nperseg", 512)
        poverlap = self.cfg.get("poverlap", 0.75)
        noverlap = int(nperseg * poverlap)
        hop_length = nperseg - noverlap
        window_type = self.cfg.get("window", "hann")
        sampling_rate = self.cfg.get("sampling_rate", 2048)
        max_frequency = self.cfg.get("max_frequency", 150)
        min_frequency = self.cfg.get("min_frequency", 0)
        
        if window_type == "hann":
            window = torch.hann_window(nperseg, device=x.device)
        elif window_type == "boxcar":
            window = torch.ones(nperseg, device=x.device)
        else:
            raise ValueError(f"Invalid window type: {window_type}")
        
        # Compute STFT
        x = torch.stft(
            x,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            return_complex=True,
            normalized=False,
            center=True
        )
        
        # Get frequency bins
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
        x = x[:, (freqs >= min_frequency) & (freqs <= max_frequency)]
        
        # Use magnitude (abs) for stft
        x = torch.abs(x)
        
        # Reshape back
        _, n_freqs, n_times = x.shape
        x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
        x = x.transpose(2, 3)  # (batch_size, n_electrodes, n_timebins, n_freqs)
        
        return x.numpy() if not was_tensor else x

