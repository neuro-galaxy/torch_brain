import copy

import numpy as np

from temporaldata import IrregularTimeSeries, RegularTimeSeries, Interval, Data


class Patching:
    r"""Patching transform that creates patches from temporal data along the time dimension.

    This transform takes a temporalData object and performs patching along the time dimension.
    The patch duration must be greater than or equal to the data duration. If the patch duration
    is longer than the data duration, zero-padding is applied. If the patch duration doesn't
    divide evenly into the number of timestamps/samples, zero-padding is also applied.

    Args:
        patch_duration (float): Duration of each patch in seconds. Must be >= data duration.
        stride (float, optional): Step size between patches. Defaults to patch_duration
            (non-overlapping). Currently unused since only one patch is created.
        timestamp_mode (str, optional): How to assign timestamps to patches. Options:
            - "first": Use the start time of the patch (default)
            - "middle": Use the middle time of the patch
            - "last": Use the end time of the patch
    """

    def __init__(self, patch_duration: float, stride: float = None, timestamp_mode: str = "first"):
        self.patch_duration = patch_duration
        self.stride = stride if stride is not None else patch_duration
        self.timestamp_mode = timestamp_mode

        if timestamp_mode not in ["first", "middle", "last"]:
            raise ValueError(
                f"timestamp_mode must be 'first', 'middle', or 'last', got '{timestamp_mode}'"
            )

    def __call__(self, data: Data) -> Data:
        """Apply patching transform to the data.

        Args:
            data: The temporalData object to patch.

        Returns:
            A new Data object with patched time-series fields.
        """
        if data.domain is None:
            raise ValueError("Data object must have a domain to apply patching.")

        data_duration = data.end - data.start

        # Check that patch duration is >= data duration
        if self.patch_duration < data_duration:
            raise ValueError(
                f"patch duration ({self.patch_duration}) is shorter than data duration ({data_duration})"
            )

        out = Data()

        # Process all fields
        for key, value in data.__dict__.items():
            if key in ["_domain", "_absolute_start", "splits"]:
                continue
            elif isinstance(value, IrregularTimeSeries):
                out.__dict__[key] = self._patch_irregular_time_series(
                    value, data.start, data_duration
                )
            elif isinstance(value, RegularTimeSeries):
                out.__dict__[key] = self._patch_regular_time_series(
                    value, data.start, data_duration
                )
            elif isinstance(value, Interval):
                val = copy.copy(value)
                out.__dict__[key] = val
            elif isinstance(value, Data) and value.domain is not None:
                out.__dict__[key] = self(value)
            else:
                out.__dict__[key] = copy.copy(value)

        out._domain = copy.copy(data._domain)

        out._absolute_start = data._absolute_start

        return out

    def _patch_irregular_time_series(
        self, ts: IrregularTimeSeries, data_start: float, data_duration: float
    ) -> IrregularTimeSeries:
        """Patch an IrregularTimeSeries object."""
        raise NotImplementedError("Patching of IrregularTimeSeries is not implemented yet.")

    def _patch_regular_time_series(
        self, ts: RegularTimeSeries, data_start: float, data_duration: float
    ) -> RegularTimeSeries:
        """Patch a RegularTimeSeries object."""
        # Extract the first two dimensions of the data
        time_dimension, channel_dimension  = ts.data.shape[:2]
        samples_per_patch = int(np.ceil(self.patch_duration * ts.sampling_rate))
        num_patches = int(np.ceil(time_dimension / samples_per_patch))

        # Pad the data with zeros to make it divisible by the number of patches
        zeros_to_add = num_patches * samples_per_patch - time_dimension
        padding = np.zeros((zeros_to_add, channel_dimension), dtype=ts.data.dtype)
        padded_data = np.concatenate([ts.data, padding], axis=0)
        data = padded_data.reshape(num_patches, channel_dimension, samples_per_patch)

        # Get the right timestamps depending on the timestamp mode
        timestamps = ts.timestamps

        # TODO: Need to keep going here as this sucks




