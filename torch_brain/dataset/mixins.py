import warnings

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


class MultiChannelDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing
    multi-channel recordings (e.g., sEEG).

    Provides:
        - ``get_channel_ids()`` for retrieving sorted channel IDs from
          recording views returned by ``get_recording(...)``.
        - Default channel-ID uniquification with ``subject.id`` only
          (``seeg_dataset_mixin_uniquify_channel_ids_with_subject=True``,
          ``seeg_dataset_mixin_uniquify_channel_ids_with_session=False``).
    """

    # Channel-ID uniquification toggles used by get_recording_hook.
    # Prefix order is always subject/session when enabled.
    seeg_dataset_mixin_uniquify_channel_ids_with_subject: bool = True
    seeg_dataset_mixin_uniquify_channel_ids_with_session: bool = False

    def get_recording_hook(self, data: Data):
        prefix = self._build_seeg_channel_id_prefix(data)
        if prefix:
            data.channels.id = np_string_prefix(prefix, data.channels.id.astype(str))
        super().get_recording_hook(data)

    def _normalize_channel_uniquify_components(self) -> tuple[str, ...]:
        with_subject = self.seeg_dataset_mixin_uniquify_channel_ids_with_subject
        with_session = self.seeg_dataset_mixin_uniquify_channel_ids_with_session
        if not isinstance(with_subject, bool):
            raise TypeError(
                "'seeg_dataset_mixin_uniquify_channel_ids_with_subject' must be "
                f"bool; got {type(with_subject).__name__}."
            )
        if not isinstance(with_session, bool):
            raise TypeError(
                "'seeg_dataset_mixin_uniquify_channel_ids_with_session' must be "
                f"bool; got {type(with_session).__name__}."
            )
        if with_session and not with_subject:
            warning_attr = "_seeg_dataset_mixin_warned_session_without_subject"
            if not getattr(self, warning_attr, False):
                warnings.warn(
                    "Channel-id uniquification with session only can create "
                    "cross-subject collisions when session.id is not globally unique.",
                    UserWarning,
                    stacklevel=2,
                )
                setattr(self, warning_attr, True)

        components = []
        if with_subject:
            components.append("subject_id")
        if with_session:
            components.append("session_id")
        return tuple(components)

    def _build_seeg_channel_id_prefix(self, data: Data) -> str:
        components = self._normalize_channel_uniquify_components()
        if not components:
            return ""

        component_values = {}
        if "subject_id" in components:
            component_values["subject_id"] = data.get_nested_attribute("subject.id")
        if "session_id" in components:
            component_values["session_id"] = data.get_nested_attribute("session.id")
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

        ``get_channel_ids`` aggregates ``rec.channels.id`` from ``get_recording(...)``.
        Any subject/session uniquification is applied there according to
        ``seeg_dataset_mixin_uniquify_channel_ids_with_subject`` and
        ``seeg_dataset_mixin_uniquify_channel_ids_with_session``.
        ``included_only=True`` filters by ``rec.channels.included`` before IDs are collected.
        """
        all_ids = []
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            ids = np.asarray(rec.channels.id).astype(str)
            if included_only:
                included_mask = np.asarray(rec.channels.included, dtype=bool)
                ids = ids[included_mask]
            all_ids.append(ids)
        if not all_ids:
            return []
        return np.sort(np.concatenate(all_ids).astype(str)).tolist()


# Backwards-compatibility alias.
SEEGDatasetMixin = MultiChannelDatasetMixin
