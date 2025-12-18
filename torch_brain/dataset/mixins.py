import numpy as np


class SpikingDatasetMixin:
    """
    Mixin class for datasets containing spiking data.

    Provides utilities for extracting unit information from neural datasets,
    such as retrieving the unique IDs of all recorded units.
    """

    def get_unit_ids(self) -> list[str]:
        """Return a sorted list of all unit IDs across all recordings in the dataset."""
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()
