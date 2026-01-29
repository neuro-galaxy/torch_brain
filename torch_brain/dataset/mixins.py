import numpy as np
import pandas as pd
from temporaldata import Data

from torch_brain.utils import np_string_prefix


class SpikingDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing spiking data.

    Provides:
        - ``get_unit_ids()`` for retreiving IDs of all included units.
        - If the class attribute ``spiking_dataset_mixin_uniquify_unit_ids`` is set to ``True``,
          unit IDs will be made unique across recordings by prefixing each unit ID with the
          corresponding ``session.id``. This helps avoid collisions when combining data from
          multiple sessions. (default: ``False``)
    """

    spiking_dataset_mixin_uniquify_unit_ids: bool = False

    def get_recording_hook(self, data: Data):
        if self.spiking_dataset_mixin_uniquify_unit_ids:
            data.units.id = np_string_prefix(
                f"{data.session.id}/",
                data.units.id.astype(str),
            )
        super().get_recording_hook(data)

    def get_unit_ids(self) -> list[str]:
        """Return a sorted list of all unit IDs across all recordings in the dataset."""
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()

    def compute_average_firing_rates(self) -> pd.DataFrame:
        """
        Compute and return the average firing rates for all units in the dataset.

        Returns:
            pd.DataFrame: DataFrame indexed by unit ID, containing a column 'firing_rate' (Hz)
                          with the average firing rate for each unit in the dataset.
        """
        unit_ids = []
        firing_rates = []
        for rid in self.recording_ids:
            data = self.get_recording(rid)

            total_time = (data.spikes.domain.end - data.spikes.domain.start).sum()
            idx, counts = np.unique(data.spikes.unit_index, return_counts=True)
            fr = np.zeros(len(data.units))
            fr[idx] = counts / total_time

            unit_ids.append(data.units.id)
            firing_rates.append(fr)

        unit_ids = np.concatenate(unit_ids)
        firing_rates = np.concatenate(firing_rates)

        df = pd.DataFrame({"firing_rate": firing_rates}, index=unit_ids)
        df.index.name = "unit_id"
        return df


class CalciumImagingDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing calcium imaging data.

    Provides:
        - ``get_roi_ids()`` for retrieving IDs of all included ROIs.
        - If the class attribute ``calcium_imaging_dataset_mixin_uniquify_roi_ids`` is set to ``True``,
          ROI IDs will be made unique across recordings by prefixing each ROI ID with the
          corresponding ``session.id``. This helps avoid collisions when combining data from
          multiple sessions. (default: ``False``)
    """

    calcium_imaging_dataset_mixin_uniquify_roi_ids: bool = False

    def get_recording_hook(self, data: Data):
        if self.calcium_imaging_dataset_mixin_uniquify_roi_ids:
            data.rois.id = np_string_prefix(
                f"{data.session.id}/",
                data.rois.id.astype(str),
            )
        super().get_recording_hook(data)

    def get_roi_ids(self) -> list[str]:
        """Return a sorted list of all ROI IDs across all recordings in the dataset."""
        ans = [self.get_recording(rid).rois.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()
