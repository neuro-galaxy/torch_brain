from dataclasses import dataclass
import numpy as np
import pandas as pd
from temporaldata import Data, Interval

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
        - ``get_sampling_rate()`` for retrieving signal sampling rate (Hz).
          Requires dataset classes to define
          ``seeg_dataset_mixin_sampling_rate_hz``.
        - ``get_domain_intervals()`` for retrieving full-domain intervals.
          Requires dataset classes to define
          ``seeg_dataset_mixin_domain_intervals``.
        - ``get_channel_view()`` for normalized channel metadata access.
          Requires dataset classes to define
          ``seeg_dataset_mixin_channel_views`` with full channel views.
        - ``get_channel_ids()`` for retrieving recording-disambiguated
          channel IDs in ``<channel_id>/<recording_id>`` format.
        - ``get_recording_info()`` for compact per-recording metadata.
          Requires dataset classes to define
          ``seeg_dataset_mixin_recording_infos``.
    """

    # Dataset classes should set this to expose a stable sampling-rate contract.
    seeg_dataset_mixin_sampling_rate_hz: float | None = None
    # Dataset classes should set this with all available recording-domain intervals.
    seeg_dataset_mixin_domain_intervals: dict[str, Interval] | None = None
    # Dataset classes should set this with full channel views for each recording.
    seeg_dataset_mixin_channel_views: dict[str, "SEEGDatasetMixin.ChannelView"] | None = None
    # Dataset classes should set this with compact recording metadata.
    seeg_dataset_mixin_recording_infos: dict[str, "SEEGDatasetMixin.RecordingInfo"] | None = None

    @dataclass(frozen=True)
    class ChannelView:
        """Channel metadata view for one recording."""

        ids: np.ndarray
        names: np.ndarray
        included_mask: np.ndarray
        # Channel coordinates ordered as (L, I, P), or ``None`` when unavailable.
        lip: np.ndarray | None

    @dataclass(frozen=True)
    class RecordingInfo:
        """Compact metadata summary for one recording."""

        recording_id: str
        subject_id: str | int | None
        session_id: str | int | None
        sampling_rate_hz: float
        domain: Interval
        n_channels: int
        n_included_channels: int

    def get_sampling_rate(self, recording_id: str | None = None) -> float:
        """Return recording sampling rate in Hz."""
        _ = recording_id  # keep signature compatible with per-recording datasets
        if self.seeg_dataset_mixin_sampling_rate_hz is None:
            raise NotImplementedError(
                "SEEG datasets must define 'seeg_dataset_mixin_sampling_rate_hz'."
            )
        return float(self.seeg_dataset_mixin_sampling_rate_hz)

    def get_domain_intervals(
        self, recording_ids: list[str] | None = None
    ) -> dict[str, Interval]:
        """Return full-domain intervals for the provided recordings."""
        # Domain intervals are dataset-specific and should be precomputed at init.
        if self.seeg_dataset_mixin_domain_intervals is None:
            raise NotImplementedError(
                "SEEG datasets must define 'seeg_dataset_mixin_domain_intervals'."
            )
        ids = self.recording_ids if recording_ids is None else recording_ids
        missing = [rid for rid in ids if rid not in self.seeg_dataset_mixin_domain_intervals]
        if missing:
            raise KeyError(f"Missing domain intervals for recording_ids: {missing}")
        return {rid: self.seeg_dataset_mixin_domain_intervals[rid] for rid in ids}

    def get_channel_view(
        self, recording_id: str, *, included_only: bool = False
    ) -> "SEEGDatasetMixin.ChannelView":
        """Return normalized channel metadata and optional LIP coordinates."""
        # Full channel views come from dataset-managed cache so schema conversion stays local.
        if self.seeg_dataset_mixin_channel_views is None:
            raise NotImplementedError(
                "SEEG datasets must define 'seeg_dataset_mixin_channel_views'."
            )

        if recording_id not in self.seeg_dataset_mixin_channel_views:
            raise KeyError(f"Missing channel view for recording_id '{recording_id}'.")

        full_view = self.seeg_dataset_mixin_channel_views[recording_id]
        if not included_only:
            return full_view

        # Included-only views are frequently requested by samplers/evaluators.
        # Cache the derived filtered view to avoid repeated mask application.
        included_cache = self._get_seeg_included_channel_view_cache()
        if recording_id in included_cache:
            # Keep the source full-view identity so we can invalidate stale derived
            # views if datasets replace full channel views after initialization.
            source_view_id, included_view = included_cache[recording_id]
            if source_view_id == id(full_view):
                return included_view

        included_view = self._filter_included_channels(full_view)
        included_cache[recording_id] = (id(full_view), included_view)
        return included_view

    def get_channel_ids(self, *, included_only: bool = False) -> list[str]:
        """Return sorted ``<channel_id>/<recording_id>`` IDs across recordings."""
        all_ids = []
        for rid in self.recording_ids:
            ids = self.get_channel_view(rid, included_only=included_only).ids
            ids = np.asarray(ids).astype(str)
            # Postfix recording ID to avoid cross-recording collisions.
            all_ids.append(np.char.add(np.char.add(ids, "/"), rid))
        if not all_ids:
            return []
        return np.sort(np.concatenate(all_ids).astype(str)).tolist()

    def get_recording_info(self, recording_id: str) -> "SEEGDatasetMixin.RecordingInfo":
        """Return compact metadata for one recording."""
        # RecordingInfo is also dataset-managed to avoid mixin-side schema assumptions.
        if self.seeg_dataset_mixin_recording_infos is None:
            raise NotImplementedError(
                "SEEG datasets must define 'seeg_dataset_mixin_recording_infos'."
            )
        if recording_id not in self.seeg_dataset_mixin_recording_infos:
            raise KeyError(f"Missing recording info for recording_id '{recording_id}'.")
        return self.seeg_dataset_mixin_recording_infos[recording_id]

    def _filter_included_channels(
        self, view: "SEEGDatasetMixin.ChannelView"
    ) -> "SEEGDatasetMixin.ChannelView":
        mask = view.included_mask
        lip = None if view.lip is None else view.lip[mask]
        return self.ChannelView(
            ids=view.ids[mask],
            names=view.names[mask],
            included_mask=np.ones(int(np.sum(mask)), dtype=bool),
            lip=lip,
        )

    def _get_seeg_included_channel_view_cache(
        self,
    ) -> dict[str, tuple[int, "SEEGDatasetMixin.ChannelView"]]:
        cache = getattr(self, "_seeg_included_channel_view_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_seeg_included_channel_view_cache", cache)
        return cache
