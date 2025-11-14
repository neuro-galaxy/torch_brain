import copy

import numpy as np

from temporaldata import IrregularTimeSeries, RegularTimeSeries, Interval, Data


class RegularPatching:
    r"""Patching transform that creates patches from temporal data along the time dimension.

    This transform takes a temporalData object and performs patching along the time dimension
    for all RegularTimeSeries objects. The data is reshaped from (time, channels, ...) to
    (num_patches, channels, patch_samples, ...).

    Args:
        patch_duration (float): Duration of each patch in seconds.
        stride (float, optional): Step size between patches in seconds. Defaults to 
            patch_duration (non-overlapping). Can be smaller than patch_duration to
            create overlapping patches.
        timestamp_mode (str, optional): How to assign timestamps to patches. Options:
            - "start": Use the start time of the patch (default)
            - "middle": Use the middle time of the patch
    
    Example:
        >>> # Non-overlapping patches
        >>> transform = RegularPatching(patch_duration=1.0, stride=1.0)
        >>> patched_data = transform(data)
        
        >>> # Overlapping patches (50% overlap)
        >>> transform = RegularPatching(patch_duration=2.0, stride=1.0)
        >>> patched_data = transform(data)
    """

    def __init__(self, patch_duration: float, stride: float = None, timestamp_mode: str = "start"):
        self.patch_duration = patch_duration
        self.stride = stride if stride is not None else patch_duration
        self.timestamp_mode = timestamp_mode

        if timestamp_mode not in ["start", "middle"]:
            raise ValueError(
                f"timestamp_mode must be 'start' or 'middle', got '{timestamp_mode}'"
            )

    def __call__(self, data: Data) -> Data:
        """Apply patching transform to the data.

        Args:
            data: The temporalData object to patch.

        Returns:
            A new Data object with patched RegularTimeSeries fields.
        """
        if data.domain is None:
            raise ValueError("Data object must have a domain to apply patching.")

        out = Data()

        # Process all fields
        for key, value in data.__dict__.items():
            if key in ["_domain", "_absolute_start"]:
                continue
            elif isinstance(value, RegularTimeSeries):
                out.__dict__[key] = self._patch_regular_time_series(value)
            elif isinstance(value, IrregularTimeSeries):
                # Leave IrregularTimeSeries unchanged
                out.__dict__[key] = copy.copy(value)
            elif isinstance(value, Interval):
                out.__dict__[key] = copy.copy(value)
            elif isinstance(value, Data) and value.domain is not None:
                out.__dict__[key] = self(value)
            else:
                # Preserve all other types (ArrayDict, scalars, etc.)
                out.__dict__[key] = copy.copy(value)

        # Update domain to reflect new time structure
        patched_domain = None
        for key, value in out.__dict__.items():
            if isinstance(value, RegularTimeSeries):
                patched_domain = value.domain
                break
        
        if patched_domain is not None:
            out._domain = patched_domain
        else:
            out._domain = copy.copy(data._domain)

        out._absolute_start = data._absolute_start

        return out

    def _patch_regular_time_series(self, ts: RegularTimeSeries) -> RegularTimeSeries:
        """Patch a RegularTimeSeries object using efficient vectorized operations.
        
        Args:
            ts: The RegularTimeSeries to patch.
            
        Returns:
            A new RegularTimeSeries with patched data.
        """
        # Get data shape and parameters
        data = ts.data
        time_samples = data.shape[0]
        sampling_rate = ts.sampling_rate
        
        # Calculate patch parameters in samples
        patch_samples = int(np.round(self.patch_duration * sampling_rate))
        stride_samples = int(np.round(self.stride * sampling_rate))
        
        # Calculate number of patches needed
        if time_samples <= patch_samples:
            num_patches = 1
        else:
            num_patches = int(np.ceil((time_samples - patch_samples) / stride_samples)) + 1
        
        # Calculate total samples needed after padding
        total_samples_needed = (num_patches - 1) * stride_samples + patch_samples
        
        # Pad data if necessary using efficient numpy pad
        if time_samples < total_samples_needed:
            pad_width = [(0, total_samples_needed - time_samples)] + [(0, 0)] * (data.ndim - 1)
            padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
        else:
            padded_data = data
        
        # Create index array: shape (num_patches, patch_samples)
        indices = np.arange(patch_samples)[None, :] + stride_samples * np.arange(num_patches)[:, None]
        
        patches = padded_data[indices]
        patches = np.moveaxis(patches, 2, 1)
        new_sampling_rate = 1.0 / self.stride
        
        if self.timestamp_mode == "start":
            domain_start = 0.0
            domain_end = (num_patches - 1) / new_sampling_rate
        elif self.timestamp_mode == "middle":
            domain_start = self.patch_duration / 2
            domain_end = domain_start + (num_patches - 1) / new_sampling_rate
        
        new_domain = Interval(start=domain_start, end=domain_end)
        
        # Create new RegularTimeSeries with patched data
        patched_ts = RegularTimeSeries.__new__(RegularTimeSeries)
        patched_ts.__dict__['data'] = patches
        patched_ts._sampling_rate = new_sampling_rate
        patched_ts._domain = new_domain
        
        return patched_ts
