import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)


class MaskingBase(ABC):
    """Base class for masking transforms.

    This class provides common functionality for adding binary masks to time series data.
    The mask is a boolean array of shape (time, channel) where True means keep and False means mask.

    Args:
        random_seed: Random seed for reproducibility (default: None).
    """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def __call__(self, data: Data) -> Data:
        """Apply masking transform to the data.

        Args:
            data: The Data object to mask.

        Returns:
            A new Data object with mask attributes added to time series.
        """
        # Create a shallow copy of the data object
        out = copy.copy(data)

        # Process all fields
        for key, value in data.__dict__.items():
            if key in ["_domain", "_absolute_start"]:
                # Skip private attributes related to domain
                continue
            elif isinstance(value, RegularTimeSeries):
                out.__dict__[key] = self._mask_time_series(value)
            elif isinstance(value, IrregularTimeSeries):
                # Check if regularly spaced
                if self._is_regularly_spaced(value):
                    out.__dict__[key] = self._mask_time_series(value)
                else:
                    # Skip irregularly spaced time series
                    out.__dict__[key] = copy.copy(value)
            elif isinstance(value, Interval):
                out.__dict__[key] = copy.copy(value)
            elif isinstance(value, ArrayDict):
                out.__dict__[key] = copy.copy(value)
            elif isinstance(value, Data) and value.domain is not None:
                # Recursively apply masking to nested Data objects
                out.__dict__[key] = self(value)
            else:
                # Preserve all other types (scalars, etc.)
                out.__dict__[key] = (
                    copy.copy(value) if hasattr(value, "__dict__") else value
                )

        return out

    def _mask_time_series(self, ts):
        """Add mask attribute to a time series.

        Args:
            ts: Time series object (RegularTimeSeries or IrregularTimeSeries).

        Returns:
            New time series with mask attribute added.
        """
        # Make a shallow copy
        new_ts = copy.copy(ts)

        # Get all array attributes from the time series (excluding timestamps)
        array_attrs = []
        for key in ts.keys():
            if key == "timestamps":
                continue
            attr = getattr(ts, key)
            if isinstance(attr, np.ndarray) and attr.ndim >= 2:
                array_attrs.append((key, attr))

        # If no suitable array attributes found, return unchanged
        if not array_attrs:
            return new_ts

        # Use the first array attribute to determine dimensions
        # All array attributes should have the same first two dimensions (time, channel)
        _, first_attr = array_attrs[0]
        time_dim = first_attr.shape[0]
        channel_dim = first_attr.shape[1] if len(first_attr.shape) > 1 else 1

        # Generate a single mask for all attributes
        mask = self._generate_mask(time_dim, channel_dim, ts)

        # Add mask as attribute
        new_ts.mask = mask

        return new_ts

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

    @abstractmethod
    def _generate_mask(self, time_dim: int, channel_dim: int, ts) -> np.ndarray:
        """Generate binary mask for the time series.

        Args:
            time_dim: Number of time points.
            channel_dim: Number of channels.
            ts: The time series object (needed for some masking methods).

        Returns:
            Binary mask array of shape (time_dim, channel_dim) where True means keep
            and False means mask.
        """
        pass


class TimeMasking(MaskingBase):
    """Time masking transform that masks random time windows across all channels.

    This transform masks contiguous windows in the time dimension. All channels are
    masked for the selected time windows.

    Args:
        mask_percentage: Percentage of time to mask (0-1).
        window_duration: Duration of each masked window in seconds.
        random_seed: Random seed for reproducibility (default: None).

    Example:
        >>> ts = RegularTimeSeries(
        ...     neural_activity=np.random.randn(1000, 64),
        ...     sampling_rate=100.0,
        ...     domain=Interval(0.0, 10.0),
        ... )
        >>> data = Data(recordings=ts, domain=Interval(0.0, 10.0))
        >>>
        >>> transform = TimeMasking(mask_percentage=0.2, window_duration=0.1, random_seed=42)
        >>> masked_data = transform(data)
    """

    def __init__(
        self,
        mask_percentage: float,
        window_duration: float,
        random_seed=None,
    ):
        super().__init__(random_seed=random_seed)

        if not 0 <= mask_percentage <= 1:
            raise ValueError(
                f"mask_percentage must be between 0 and 1, got {mask_percentage}"
            )
        if window_duration <= 0:
            raise ValueError(f"window_duration must be positive, got {window_duration}")

        self.mask_percentage = mask_percentage
        self.window_duration = window_duration

    def _generate_mask(self, time_dim: int, channel_dim: int, ts) -> np.ndarray:
        """Generate time masking pattern.

        Args:
            time_dim: Number of time points.
            channel_dim: Number of channels.
            ts: The time series object (needed to get sampling rate).

        Returns:
            Binary mask of shape (time_dim, channel_dim).
        """
        # Initialize mask with all True (keep everything)
        mask = np.ones((time_dim, channel_dim), dtype=bool)

        if time_dim == 0:
            return mask

        # Convert window duration to samples
        if isinstance(ts, RegularTimeSeries):
            window_samples = int(np.round(self.window_duration * ts.sampling_rate))
        else:
            # For IrregularTimeSeries, estimate sampling rate from timestamps
            if len(ts.timestamps) > 1:
                avg_dt = np.mean(np.diff(ts.timestamps))
                window_samples = int(np.round(self.window_duration / avg_dt))
            else:
                window_samples = 1

        # Ensure at least 1 sample
        window_samples = max(1, window_samples)

        # Calculate number of time points to mask
        num_samples_to_mask = int(np.round(time_dim * self.mask_percentage))

        if num_samples_to_mask == 0:
            return mask

        # Calculate number of windows needed
        num_windows = max(1, int(np.ceil(num_samples_to_mask / window_samples)))

        # Randomly sample start positions for windows
        for _ in range(num_windows):
            if time_dim <= window_samples:
                # If window is larger than time dimension, mask everything
                mask[:, :] = False
                break

            # Randomly select start position
            max_start = time_dim - window_samples
            if max_start > 0:
                start_idx = self._rng.integers(0, max_start + 1)
                end_idx = min(start_idx + window_samples, time_dim)

                # Mask this window (set to False)
                mask[start_idx:end_idx, :] = False

        return mask


class ChannelMasking(MaskingBase):
    """Channel masking transform that masks random channels across all time points.

    This transform masks entire channels (all time points for selected channels).

    Args:
        mask_percentage: Percentage of channels to mask (0-1).
        random_seed: Random seed for reproducibility (default: None).

    Example:
        >>> ts = RegularTimeSeries(
        ...     lfp=np.random.randn(500, 32),
        ...     sampling_rate=250.0,
        ...     domain=Interval(0.0, 2.0),
        ... )
        >>> data = Data(lfp_recording=ts, domain=Interval(0.0, 2.0))
        >>>
        >>> transform = ChannelMasking(mask_percentage=0.3, random_seed=42)
        >>> masked_data = transform(data)
    """

    def __init__(self, mask_percentage: float, random_seed=None):
        super().__init__(random_seed=random_seed)

        if not 0 <= mask_percentage <= 1:
            raise ValueError(
                f"mask_percentage must be between 0 and 1, got {mask_percentage}"
            )

        self.mask_percentage = mask_percentage

    def _generate_mask(self, time_dim: int, channel_dim: int, ts) -> np.ndarray:
        """Generate channel masking pattern.

        Args:
            time_dim: Number of time points.
            channel_dim: Number of channels.
            ts: The time series object (not used for channel masking).

        Returns:
            Binary mask of shape (time_dim, channel_dim).
        """
        # Initialize mask with all True (keep everything)
        mask = np.ones((time_dim, channel_dim), dtype=bool)

        if channel_dim == 0:
            return mask

        # Calculate number of channels to mask
        num_channels_to_mask = int(np.round(channel_dim * self.mask_percentage))

        if num_channels_to_mask == 0:
            return mask

        # Randomly select channels to mask
        channels_to_mask = self._rng.choice(
            channel_dim, size=num_channels_to_mask, replace=False
        )

        # Mask selected channels (all time points)
        mask[:, channels_to_mask] = False

        return mask


class BlockMasking(MaskingBase):
    """Block masking transform that masks random blocks in time and channel dimensions.

    This transform divides the data into a grid of blocks and randomly masks entire blocks.

    Args:
        time_block_size: Size of blocks in time dimension (in seconds).
        channel_block_size: Size of blocks in channel dimension.
        mask_percentage: Percentage of blocks to mask (0-1).
        random_seed: Random seed for reproducibility (default: None).

    Example:
        >>> ts = IrregularTimeSeries(
        ...     timestamps=np.arange(0, 10.0, 0.01),
        ...     firing_rates=np.random.randn(1000, 96),
        ...     domain=Interval(0.0, 10.0),
        ... )
        >>> data = Data(neural_data=ts, domain=Interval(0.0, 10.0))
        >>>
        >>> transform = BlockMasking(
        ...     time_block_size=0.5,
        ...     channel_block_size=8,
        ...     mask_percentage=0.3,
        ...     random_seed=42
        ... )
        >>> masked_data = transform(data)
    """

    def __init__(
        self,
        time_block_size: float,
        channel_block_size: int,
        mask_percentage: float,
        random_seed=None,
    ):
        super().__init__(random_seed=random_seed)

        if time_block_size <= 0:
            raise ValueError(f"time_block_size must be positive, got {time_block_size}")
        if channel_block_size <= 0:
            raise ValueError(
                f"channel_block_size must be positive, got {channel_block_size}"
            )
        if not 0 <= mask_percentage <= 1:
            raise ValueError(
                f"mask_percentage must be between 0 and 1, got {mask_percentage}"
            )

        self.time_block_size = time_block_size
        self.channel_block_size = channel_block_size
        self.mask_percentage = mask_percentage

    def _generate_mask(self, time_dim: int, channel_dim: int, ts) -> np.ndarray:
        """Generate block masking pattern.

        Args:
            time_dim: Number of time points.
            channel_dim: Number of channels.
            ts: The time series object (needed to get sampling rate).

        Returns:
            Binary mask of shape (time_dim, channel_dim).
        """
        # Initialize mask with all True (keep everything)
        mask = np.ones((time_dim, channel_dim), dtype=bool)

        if time_dim == 0 or channel_dim == 0:
            return mask

        # Convert time_block_size from seconds to samples
        if isinstance(ts, RegularTimeSeries):
            time_block_samples = int(np.round(self.time_block_size * ts.sampling_rate))
        else:
            # For IrregularTimeSeries, estimate sampling rate from timestamps
            if len(ts.timestamps) > 1:
                avg_dt = np.mean(np.diff(ts.timestamps))
                time_block_samples = int(np.round(self.time_block_size / avg_dt))
            else:
                time_block_samples = 1

        # Ensure at least 1 sample
        time_block_samples = max(1, time_block_samples)

        # Calculate number of blocks in each dimension
        num_time_blocks = int(np.ceil(time_dim / time_block_samples))
        num_channel_blocks = int(np.ceil(channel_dim / self.channel_block_size))
        total_blocks = num_time_blocks * num_channel_blocks

        if total_blocks == 0:
            return mask

        # Calculate number of blocks to mask
        num_blocks_to_mask = int(np.round(total_blocks * self.mask_percentage))

        if num_blocks_to_mask == 0:
            return mask

        # Create list of all block indices
        all_block_indices = [
            (t_block, c_block)
            for t_block in range(num_time_blocks)
            for c_block in range(num_channel_blocks)
        ]

        # Randomly select blocks to mask
        blocks_to_mask = self._rng.choice(
            len(all_block_indices),
            size=min(num_blocks_to_mask, len(all_block_indices)),
            replace=False,
        )

        # Mask selected blocks
        for block_idx in blocks_to_mask:
            t_block, c_block = all_block_indices[block_idx]

            # Calculate block boundaries
            t_start = t_block * time_block_samples
            t_end = min((t_block + 1) * time_block_samples, time_dim)
            c_start = c_block * self.channel_block_size
            c_end = min((c_block + 1) * self.channel_block_size, channel_dim)

            # Mask this block
            mask[t_start:t_end, c_start:c_end] = False

        return mask
