from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from torch_brain.data import Data
from torch_brain.utils import np_string_prefix
from torch_brain.utils.split import PROTOCOL_VERSION


def _identity_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def recording_fingerprint(path: Path) -> dict:
    """Capture the file revision used by immutable recording metadata snapshots."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class SpikingDatasetMixin:
    """
    Mixin class for :class:`torch_brain.datasets.Dataset` subclasses containing spiking data.

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
    Mixin class for :class:`torch_brain.datasets.Dataset` subclasses containing calcium imaging data.

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
    Mixin class for :class:`torch_brain.datasets.Dataset` subclasses containing
    multi-channel recordings (e.g., EEG, ECoG, EMG, sEEG, etc).

    Provides:
        - ``get_channel_ids()`` for retrieving sorted channel IDs from
          recording views returned by ``get_recording(...)``.
        - Configurable channel-ID uniquification by prepending metadata
          components before each channel id:
          ``multichannel_dataset_mixin_uniquify_channel_ids_with_session``
          prepends ``session.id`` (default ``False``) and
          ``multichannel_dataset_mixin_uniquify_channel_ids_with_subject``
          prepends ``subject.id`` (default ``True``). This ``subject.id``
          uniquification allows channels with the same name in the same
          subject to be treated as the same channel across sessions. If both
          are enabled,
          the prefix order is ``subject.id/session.id``.
    """

    # Channel-ID uniquification toggles used by get_recording_hook.
    # Prefix order is always subject/session when enabled.
    multichannel_dataset_mixin_uniquify_channel_ids_with_subject: bool = True
    multichannel_dataset_mixin_uniquify_channel_ids_with_session: bool = False

    def get_recording_hook(self, data: Data):
        prefix = self._build_multichannel_channel_id_prefix(data)
        if prefix:
            data.channels.id = np_string_prefix(
                prefix,
                data.channels.id.astype(str),
            )
        super().get_recording_hook(data)

    def _build_multichannel_channel_id_prefix(self, data: Data) -> str:
        prefix = ""
        if self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject:
            prefix += f"{data.subject.id}/"
        if self.multichannel_dataset_mixin_uniquify_channel_ids_with_session:
            prefix += f"{data.session.id}/"
        return prefix

    def get_channel_ids(self) -> list[str]:
        """Return sorted channel IDs across recordings.

        ``get_channel_ids`` aggregates ``rec.channels.id`` from ``get_recording(...)``.
        Any subject/session uniquification is applied there according to
        ``multichannel_dataset_mixin_uniquify_channel_ids_with_subject`` and
        ``multichannel_dataset_mixin_uniquify_channel_ids_with_session``.
        """
        ans = [self.get_recording(rid).channels.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()


class TemporalBlockDatasetMixin:
    """Read saved temporal blocks and selection identity without loading waveforms.

    Hosts provide ``_dataset_dir``, ``_use_split_selection``, ``recording_ids``,
    and ``get_split_key()``. Place this mixin before the Dataset base.
    Metadata is cached per instance and recordings must remain immutable.
    """

    def __init__(self, *args, **kwargs):
        self._temporal_selection_cache: dict = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def read_temporal_selection(
        path: Path, split_key: str, *, cache: dict | None = None
    ) -> dict | None:
        """Read block bounds and selection identity without loading waveforms.

        Historical files return None. Cached entries reject changed file revisions;
        signal values and legacy signal hashes do not enter the selection digest.
        """
        key = (path, split_key)
        if cache is not None and key in cache:
            captured, info = cache[key]
            try:
                current = recording_fingerprint(path)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Recording changed after capturing temporal metadata: {path}"
                ) from exc
            if current != captured:
                raise RuntimeError(
                    f"Recording changed after capturing temporal metadata: {path}"
                )
            return deepcopy(info)

        before = recording_fingerprint(path)
        info = None
        with h5py.File(path, "r") as handle:
            if "temporal_splits_json" in handle.attrs:
                metadata = json.loads(handle.attrs["temporal_splits_json"])
                if metadata["protocol"] != PROTOCOL_VERSION:
                    raise ValueError(f"Unsupported temporal protocol in {path}")
                _, _, regime, _, fold, split = split_key.split("$")
                if regime == "within_session":
                    bounds = metadata["blocks"][fold.removeprefix("fold")][split]
                    if (
                        len(bounds) != 2
                        or not np.isfinite(bounds).all()
                        or bounds[0] >= bounds[1]
                    ):
                        raise ValueError(f"Invalid temporal block in {path}: {bounds}")
                    group = handle["splits"][split_key]
                    starts, ends, labels = (
                        group[field][:] for field in ("start", "end", "label")
                    )
                    if (
                        starts.ndim != 1
                        or ends.shape != starts.shape
                        or labels.shape != starts.shape
                        or not np.isfinite(starts).all()
                        or not np.isfinite(ends).all()
                        or np.any(ends <= starts)
                    ):
                        raise ValueError(
                            f"Saved start/end/label arrays must match finite positive-duration windows: {path}, {split_key}"
                        )
                    extra = np.maximum(
                        0, (metadata["context_width_s"] - (ends - starts)) / 2
                    )
                    if np.any(starts - extra < bounds[0] - 1e-9) or np.any(
                        ends + extra > bounds[1] + 1e-9
                    ):
                        raise ValueError(
                            f"Saved windows cross temporal support boundaries: {path}, {split_key}"
                        )
                    selection = {
                        "metadata": metadata,
                        "split_key": split_key,
                        "start": starts.tolist(),
                        "end": ends.tolist(),
                        "label": labels.tolist(),
                        "channels": handle["channel_splits"][split_key][:].tolist(),
                    }
                    info = {
                        "protocol": metadata["protocol"],
                        "bounds": bounds,
                        "selection_digest": _identity_digest(selection),
                        "recording_path": str(path),
                        "recording_fingerprint": before,
                        "context_width_s": metadata["context_width_s"],
                        "gap_s": metadata["gap_s"],
                    }
        if recording_fingerprint(path) != before:
            raise RuntimeError(
                f"Recording changed while reading temporal metadata: {path}"
            )
        if cache is not None:
            # Readers return fresh mappings so caller mutations cannot alter the snapshot.
            cache[key] = (before.copy(), deepcopy(info))
        return info

    def get_temporal_block(self, recording_id: str) -> dict:
        """Return the prepared split's block for downstream split-local filtering."""
        if not self._use_split_selection or recording_id not in self.recording_ids:
            raise ValueError(
                "Temporal blocks require an active split-selected recording"
            )
        info = self.read_temporal_selection(
            self._dataset_dir / f"{recording_id}.h5",
            self.get_split_key(),
            cache=self._temporal_selection_cache,
        )
        if info is None:
            raise ValueError(
                "Native temporal splits are missing; reprepare this recording"
            )
        return info

    def describe_temporal_selection(self) -> dict:
        """Describe active native selections without loading waveform arrays.

        Explicit-recording and historical-only selections have no temporal
        summary. Mixing historical and native selected recordings is rejected.
        """
        if not self._use_split_selection:
            return {}
        split_key = self.get_split_key()
        selections = {
            rid: self.read_temporal_selection(
                self._dataset_dir / f"{rid}.h5",
                split_key,
                cache=self._temporal_selection_cache,
            )
            for rid in self.recording_ids
        }
        native = {rid: info for rid, info in selections.items() if info is not None}
        if not native:
            return {}
        if len(native) != len(selections):
            raise ValueError("Cannot mix historical and native temporal recordings")
        return {
            "temporal_protocol": PROTOCOL_VERSION,
            "temporal_selection_digests": {
                rid: info["selection_digest"] for rid, info in native.items()
            },
            "temporal_recording_fingerprints": {
                rid: info["recording_fingerprint"] for rid, info in native.items()
            },
        }
