import numpy as np
import pandas as pd
from temporaldata import Data

from torch_brain.utils import np_string_prefix


class SpikingDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing spiking data.

    Provides:
        - ``get_unit_ids()`` for retrieving IDs of all included units.
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


class SEEGDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing sEEG data.

    Provides:
        - ``get_domain_intervals()`` for full-domain intervals
          (inherited from :class:`torch_brain.dataset.Dataset`).
        - ``get_channel_ids()`` for retrieving recording-disambiguated
          channel IDs in ``<channel_id>/<recording_id>`` format.
        - ``get_channel_arrays()`` for normalized channel metadata access
          (inherited from :class:`torch_brain.dataset.Dataset`).
    """

    # Channel-ID components used for hook-based uniquification.
    # Supported values are "subject_id" and "session_id".
    # Example: {"subject_id"} for subject-only prefixes.
    seeg_dataset_mixin_uniquify_channel_ids: set[str] | frozenset[str] = frozenset()
    _SEEG_CHANNEL_ID_COMPONENT_ORDER: tuple[str, str] = ("subject_id", "session_id")

    def get_recording_hook(self, data: Data):
        prefix = self._build_seeg_channel_id_prefix(data)
        if prefix:
            data.channels.id = np_string_prefix(prefix, data.channels.id.astype(str))
        super().get_recording_hook(data)

    def _normalize_channel_uniquify_components(self) -> tuple[str, ...]:
        config = self.seeg_dataset_mixin_uniquify_channel_ids
        valid_components = set(self._SEEG_CHANNEL_ID_COMPONENT_ORDER)

        if not isinstance(config, (set, frozenset)):
            raise TypeError(
                "'seeg_dataset_mixin_uniquify_channel_ids' must be a set or "
                f"frozenset; got {type(config).__name__}."
            )

        normalized = set(config)
        invalid = [item for item in normalized if item not in valid_components]
        if invalid:
            invalid_str = ", ".join(sorted(repr(item) for item in invalid))
            raise ValueError(
                "Invalid channel uniquify components: "
                f"{invalid_str}. Expected subset of "
                f"{sorted(valid_components)}."
            )
        return tuple(
            component
            for component in self._SEEG_CHANNEL_ID_COMPONENT_ORDER
            if component in normalized
        )

    def _build_seeg_channel_id_prefix(self, data: Data) -> str:
        components = self._normalize_channel_uniquify_components()
        if not components:
            return ""

        component_values = {
            "subject_id": getattr(getattr(data, "subject", None), "id", None),
            "session_id": getattr(getattr(data, "session", None), "id", None),
        }
        prefix_parts = [
            str(component_values[component])
            for component in components
            if component_values[component] is not None
        ]
        if not prefix_parts:
            return ""
        return "/".join(prefix_parts) + "/"

    def get_channel_ids(self, *, included_only: bool = False) -> list[str]:
        """Return sorted channel IDs across recordings.

        When channel-ID uniquification is enabled, this reuses ``get_recording(...)``
        so the returned IDs exactly match the hook-mutated recording view. Otherwise
        it falls back to ``<channel_id>/<recording_id>`` disambiguation.
        """
        all_ids = []
        for rid in self.recording_ids:
            channel_arrays = self.get_channel_arrays(rid, included_only=included_only)
            ids = np.asarray(channel_arrays["ids"]).astype(str)
            if self.seeg_dataset_mixin_uniquify_channel_ids:
                all_ids.append(ids)
                continue
            # Postfix recording ID to avoid cross-recording collisions.
            all_ids.append(np.char.add(np.char.add(ids, "/"), rid))
        if not all_ids:
            return []
        return np.sort(np.concatenate(all_ids).astype(str)).tolist()
