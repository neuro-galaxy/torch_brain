"""
Laplacian rereferencing + STFT preprocessor.
"""
import torch
import numpy as np
from omegaconf import DictConfig
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


def laplacian_rereference_neural_data(electrode_data, electrode_labels, remove_non_laplacian=True):
    """
    Rereference the neural data using the laplacian method.
    Copied from eval_utils.py.
    
    Args:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes
    
    Returns:
        rereferenced_data: torch tensor
        rereferenced_labels: list of electrode labels
        original_electrode_indices: list of indices
    """
    def get_all_laplacian_electrodes(electrode_labels):
        """Get all laplacian electrodes for a given subject."""
        def stem_electrode_name(name):
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return ''.join(reversed(stem)), int(''.join(reversed(num)))
        
        def has_neighbors(stem, stems):
            (x, y) = stem
            return ((x, y+1) in stems) or ((x, y-1) in stems)
        
        def get_neighbors(stem, stems):
            (x, y) = stem
            return [f'{x}{y}' for (x, y) in [(x, y+1), (x, y-1)] if (x, y) in stems]
        
        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x, y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes}
        return electrodes, neighbors
    
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)
    
    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]
    
    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(electrode_labels)
    laplacian_neighbor_indices = {
        laplacian_electrode_label: [
            electrode_labels.index(neighbor_label) for neighbor_label in neighbors
        ]
        for laplacian_electrode_label, neighbors in laplacian_neighbors.items()
    }
    
    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    if was_tensor:
        rereferenced_data = torch.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
            device=electrode_data.device
        )
    else:
        rereferenced_data = np.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype
        )
    
    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = (
                electrode_data[:, original_electrode_index] -
                electrode_data[:, laplacian_neighbor_indices[electrode_label]].mean(axis=1)
            )
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian:
                continue  # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, original_electrode_index]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1
    
    if batch_unsqueeze:
        if was_tensor:
            rereferenced_data = rereferenced_data.squeeze(0)
        else:
            rereferenced_data = rereferenced_data.squeeze(0)
    
    return rereferenced_data, (laplacian_electrodes if remove_non_laplacian else electrode_labels), original_electrode_indices


@register_preprocessor("laplacian_stft")
class LaplacianSTFTPreprocessor(BasePreprocessor):
    """Preprocessor that chains laplacian rereferencing + STFT."""
    
    def preprocess(self, data, electrode_labels):
        """
        Apply laplacian rereferencing followed by STFT.
        
        Args:
            data: Input data (torch.Tensor or numpy.ndarray)
            electrode_labels: List of electrode labels
        
        Returns:
            Preprocessed data with shape (batch_size, n_electrodes, n_timebins, n_freqs)
        """
        # First apply laplacian rereferencing
        remove_non_laplacian = self.cfg.get("remove_non_laplacian", False)
        data, new_electrode_labels, _ = laplacian_rereference_neural_data(
            data, electrode_labels, remove_non_laplacian=remove_non_laplacian
        )
        
        # Then apply STFT
        was_tensor = isinstance(data, torch.Tensor)
        x = torch.from_numpy(data) if not was_tensor else data
        
        if len(x.shape) == 2:  # if it is only (n_electrodes, n_samples)
            x = x.unsqueeze(0)
        batch_size, n_electrodes, n_samples = x.shape
        
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

