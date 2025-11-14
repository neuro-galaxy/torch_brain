import copy

import numpy as np
from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


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
        >>> data = Data(
        ...     lfp_recording=RegularTimeSeries(
        ...         data=np.random.randn(500, 32),
        ...         sampling_rate=250.0,
        ...         domain=Interval(0.0, 2.0),
        ...     ),
        ...     domain=Interval(0.0, 2.0),
        ... )
        >>> # Non-overlapping patches
        >>> transform = RegularPatching(patch_duration=1.0, stride=1.0)
        >>> patched_data = transform(data)

        >>> # Overlapping patches (50% overlap)
        >>> transform = RegularPatching(patch_duration=2.0, stride=1.0)
        >>> patched_data = transform(data)
    """

    def __init__(
        self, patch_duration: float, stride: float = None, timestamp_mode: str = "start"
    ):
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
            A new Data object with patched time series fields.
        """
        if data.domain is None:
            raise ValueError("Data object must have a domain to apply patching.")

        out = Data()

        # Process all fields
        for key, value in data.__dict__.items():
            if key in ["_domain", "_absolute_start"]:
                continue
            elif isinstance(value, RegularTimeSeries):
                out.__dict__[key] = self._patch_time_series(value)
            elif isinstance(value, IrregularTimeSeries):
                # Check if regularly spaced
                if self._is_regularly_spaced(value):
                    out.__dict__[key] = self._patch_time_series(value)
                else:
                    # Skip irregularly spaced time series
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
            if isinstance(value, (RegularTimeSeries, IrregularTimeSeries)):
                patched_domain = value.domain
                break

        if patched_domain is not None:
            out._domain = patched_domain
        else:
            out._domain = copy.copy(data._domain)

        out._absolute_start = data._absolute_start

        return out

    def _patch_time_series(self, ts):
        """Patch a time series object (RegularTimeSeries or IrregularTimeSeries).

        Args:
            ts: The time series to patch (RegularTimeSeries or IrregularTimeSeries).

        Returns:
            A new time series of the same type with patched data.
        """
        # Get all array attributes (excluding timestamps for irregular)
        array_attrs = {}

        if isinstance(ts, RegularTimeSeries):
            sampling_rate = ts.sampling_rate
            start_time = ts.domain.start[0] if ts.domain else 0.0

            # Get all array attributes from RegularTimeSeries
            for key in ts.keys():
                attr = getattr(ts, key)
                if isinstance(attr, np.ndarray) and attr.ndim >= 2:
                    array_attrs[key] = attr

        elif isinstance(ts, IrregularTimeSeries):
            # Get all array attributes (excluding timestamps)
            for key in ts.keys():
                if key == "timestamps":
                    continue
                attr = getattr(ts, key)
                if isinstance(attr, np.ndarray) and attr.ndim >= 2:
                    array_attrs[key] = attr

            # If no suitable array attributes or < 2 timestamps, return unchanged
            if not array_attrs or len(ts.timestamps) < 2:
                return copy.copy(ts)

            # Calculate effective sampling rate from timestamps
            time_diffs = np.diff(ts.timestamps)
            sampling_rate = 1.0 / np.mean(time_diffs)
            start_time = ts.timestamps[0]
        else:
            return copy.copy(ts)

        # If no suitable array attributes found, return unchanged
        if not array_attrs:
            return copy.copy(ts)

        # Use the first array attribute to determine dimensions
        first_attr = array_attrs[list(array_attrs.keys())[0]]
        time_samples = first_attr.shape[0]

        # Calculate patch parameters in samples
        patch_samples = int(np.round(self.patch_duration * sampling_rate))
        stride_samples = int(np.round(self.stride * sampling_rate))

        # Calculate number of patches needed
        if time_samples <= patch_samples:
            num_patches = 1
        else:
            num_patches = (
                int(np.ceil((time_samples - patch_samples) / stride_samples)) + 1
            )

        # Calculate total samples needed after padding
        total_samples_needed = (num_patches - 1) * stride_samples + patch_samples

        # Create index array: shape (num_patches, patch_samples)
        indices = (
            np.arange(patch_samples)[None, :]
            + stride_samples * np.arange(num_patches)[:, None]
        )

        # Patch all array attributes
        patched_attrs = {}
        for key, attr_data in array_attrs.items():
            # Pad data if necessary
            if time_samples < total_samples_needed:
                pad_width = [(0, total_samples_needed - time_samples)] + [(0, 0)] * (
                    attr_data.ndim - 1
                )
                padded_data = np.pad(
                    attr_data, pad_width, mode="constant", constant_values=0
                )
            else:
                padded_data = attr_data

            # Create patches and move channel axis
            patches = padded_data[indices]
            patches = np.moveaxis(patches, 2, 1)
            patched_attrs[key] = patches

        # Calculate new timestamps
        new_sampling_rate = 1.0 / self.stride

        if self.timestamp_mode == "start":
            domain_start = start_time
            domain_end = domain_start + (num_patches - 1) / new_sampling_rate
        elif self.timestamp_mode == "middle":
            domain_start = start_time + self.patch_duration / 2
            domain_end = domain_start + (num_patches - 1) / new_sampling_rate

        new_domain = Interval(start=domain_start, end=domain_end)

        # Create appropriate output type
        if isinstance(ts, RegularTimeSeries):
            patched_ts = RegularTimeSeries.__new__(RegularTimeSeries)
            for key, value in patched_attrs.items():
                patched_ts.__dict__[key] = value
            patched_ts._sampling_rate = new_sampling_rate
            patched_ts._domain = new_domain
        else:  # IrregularTimeSeries
            new_timestamps = np.linspace(domain_start, domain_end, num_patches)
            patched_ts = IrregularTimeSeries.__new__(IrregularTimeSeries)
            patched_ts.__dict__["timestamps"] = new_timestamps
            for key, value in patched_attrs.items():
                patched_ts.__dict__[key] = value
            patched_ts._domain = new_domain

        return patched_ts

    def _is_regularly_spaced(self, ts: IrregularTimeSeries, tolerance=1e-6) -> bool:
        """Check if timestamps are regularly spaced.

        Args:
            ts: IrregularTimeSeries object.
            tolerance: Tolerance for checking regular spacing.

        Returns:
            True if timestamps are regularly spaced, False otherwise.
        """
        timestamps = ts.timestamps
        if len(timestamps) < 2:
            return True

        diffs = np.diff(timestamps)
        return np.allclose(diffs, diffs[0], atol=tolerance)
