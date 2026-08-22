from __future__ import annotations

import copy
import re
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import Literal, get_args

import h5py
import numpy as np

from torch_brain.data import Data, Interval

from ._utils import get_processed_dir
from .dataset import Dataset
from .mixins import MultiChannelDatasetMixin

SubsetTier = Literal["full"]
LabelMode = Literal["binary", "multiclass"]
Regime = Literal[
    "within-session",
    "hold-in-session",
    "hold-out-session",
    "hold-out-subject",
]
Split = Literal["train", "val", "test"]

VALID_SUBSET_TIERS = get_args(SubsetTier)
VALID_LABEL_MODES = get_args(LabelMode)
VALID_REGIMES = get_args(Regime)
VALID_SPLITS = get_args(Split)

# Supported BYD task labels available in processed H5 splits.
VALID_TASKS = (
    "delta_volume",
    "face_num",
    "frame_brightness",
    "global_flow",
    "gpt2_surprisal",
    "local_flow",
    "onset",
    "pitch",
    "speech",
    "volume",
    "word_gap",
    "word_head_pos",
    "word_index",
    "word_length",
    "word_part_speech",
)

H5_REGIME_BY_REGIME: dict[Regime, str] = {
    "within-session": "within_session",
    "hold-in-session": "within_session",
    "hold-out-session": "within_session",
    "hold-out-subject": "within_session",
}

LEGACY_H5_TASK_BY_TASK: dict[str, str] = {
    "onset": "sentence_onset",
    "word_head_pos": "word_head",
}

# Split interval and channel-mask keys share one selector key:
# <subset_tier>$<label_mode>$<eval_setting>$<task>$fold<k>$<split>

# Strict parser for canonical recording ids like "sub-CS44_ses-P44CSR1".
_RECORDING_ID_RE = re.compile(r"^sub-CS(\d+)_ses-P(\d+)CSR(\d+)$")
_BYD_FILE_SUFFIX_RE = re.compile(r"_behavior\+ecephys$", flags=re.IGNORECASE)


def _stack_coordinate_frame(
    *,
    channels,
    recording_id: str,
    dataset_name: str,
    frame_name: str,
    field_names: tuple[str, str, str],
    expected_length: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for field_name in field_names:
        try:
            field_values = np.asarray(
                getattr(channels, field_name), dtype=float
            ).reshape(-1)
        except AttributeError as exc:
            expected_fields = ", ".join(f"channels.{name}" for name in field_names)
            raise AttributeError(
                f"Missing required channel coordinate fields for {dataset_name} "
                f"recording '{recording_id}'. Expected {expected_fields}."
            ) from exc
        if len(field_values) != expected_length:
            raise ValueError(
                f"Channel coordinate field length mismatch for recording "
                f"'{recording_id}': {field_name} expected length {expected_length}, "
                f"actual length {len(field_values)}."
            )
        values.append(field_values)
    return np.stack(values, axis=1)


def _read_seeg_signal_metadata(
    h5_path: Path, *, recording_id: str
) -> dict[str, str | float]:
    with h5py.File(h5_path, "r") as handle:
        try:
            seeg_data = handle["seeg_data"]
        except KeyError as exc:
            raise ValueError(
                "Missing required brainsets 1.1.0 neural signal metadata for "
                f"recording '{recording_id}': /seeg_data group not found."
            ) from exc
        attrs = seeg_data.attrs
        if "unit" not in attrs or "scale_to_uV" not in attrs:
            raise ValueError(
                "Missing required brainsets 1.1.0 neural signal metadata for "
                f"recording '{recording_id}': /seeg_data attrs 'unit' and "
                "'scale_to_uV' are required."
            )
        return {
            "unit": str(attrs["unit"]),
            "scale_to_uV": float(attrs["scale_to_uV"]),
        }


def _to_recording_id(subject: int, session: int) -> str:
    # Normalize integer subject/session into the canonical H5 recording id.
    if (
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        or isinstance(session, bool)
        or not isinstance(session, int)
        or session < 0
    ):
        raise ValueError(
            "_to_recording_id received invalid subject/session values: "
            f"subject={subject!r}, session={session!r}. Expected subject and "
            "session to be non-negative integers."
        )
    return f"sub-CS{subject}_ses-P{subject}CSR{session}"


def _from_recording_id(recording_id: str) -> tuple[int, int]:
    # Parse canonical ids like "sub-CS44_ses-P44CSR1" back into integers.
    match = _RECORDING_ID_RE.match(recording_id)
    if match is None:
        raise ValueError(
            f"Invalid recording_id '{recording_id}'. Expected "
            "'sub-CS<subject>_ses-P<subject>CSR<session>'."
        )
    subject_from_cs = int(match.group(1))
    subject_from_p = int(match.group(2))
    if subject_from_cs != subject_from_p:
        raise ValueError(
            f"Invalid recording_id '{recording_id}': CS subject {subject_from_cs} "
            f"does not match P subject {subject_from_p}."
        )
    return subject_from_cs, int(match.group(3))


def _normalize_recording_stem(recording_stem: str) -> str:
    """Normalize one on-disk H5 stem to the canonical BYD recording id."""
    return _BYD_FILE_SUFFIX_RE.sub("", recording_stem)


class KelesBYD2024(MultiChannelDatasetMixin, Dataset):
    """Keles BYD 2024 iEEG dataset.

    Each instance operates in exactly one of two mutually-exclusive modes:
    - BYD benchmark mode (`recording_ids=None`): active recordings are resolved
      from BYD selectors.
    - Explicit-recording mode (`recording_ids` provided): active recordings come
      directly from that subset and split selectors must not be provided.

    Args:
        root: Root directory containing processed BYD artifacts.
        recording_ids: Optional explicit recording-id subset to expose from disk.
            If omitted, the dataset uses split selectors inferred from
            ``subset_tier/test_subject/test_session/split/label_mode/task/regime/fold``.
        transform: Optional sample transform.
        subset_tier: BYD v1 supports ``"full"`` only. Required in benchmark
            mode and must be omitted in explicit-recording mode.
        test_subject: Target test subject id. Required in benchmark mode and
            must be omitted in explicit-recording mode.
        test_session: Target test session id. Required in benchmark mode and
            must be omitted in explicit-recording mode.
        split: One of ``"train"``, ``"val"``, ``"test"``. Required in
            benchmark mode; must be omitted in explicit-recording mode.
        label_mode: One of ``"binary"``, ``"multiclass"``. Defaults to ``"binary"``
            in benchmark mode.
        task: BYD task name. Defaults to ``"speech"`` in benchmark mode.
            Supported values are:
            ``"delta_volume"``, ``"face_num"``, ``"frame_brightness"``,
            ``"global_flow"``, ``"gpt2_surprisal"``, ``"local_flow"``,
            ``"onset"``, ``"pitch"``, ``"speech"``, ``"volume"``,
            ``"word_gap"``, ``"word_head_pos"``, ``"word_index"``,
            ``"word_length"``, ``"word_part_speech"``.
        regime: One of ``"within-session"``, ``"hold-in-session"``,
            ``"hold-out-session"``, ``"hold-out-subject"``. Defaults to
            ``"within-session"`` in benchmark mode. BYD regime semantics:
            - ``"within-session"``: evaluate a single target recording only.
            - ``"hold-in-session"``: train on all eligible recordings; val/test
              use the target recording.
            - ``"hold-out-session"``: train excludes the target recording;
              val/test use the target recording.
            - ``"hold-out-subject"``: train excludes all recordings from the
              target subject; val/test use the target recording.
        fold: Fold index used only in benchmark mode. Defaults to ``0`` in
            benchmark mode and must be omitted in explicit-recording mode.
            Valid values for all regimes: ``0`` or ``1``.
        uniquify_channel_ids_with_subject: Whether to prefix channel IDs with
            ``subject.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``True``.
        uniquify_channel_ids_with_session: Whether to prefix channel IDs with
            ``session.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``False``.
        dirname: Subdirectory under ``root`` containing recording H5 files.
    """

    _ALLOWED_FOLDS_BY_REGIME: dict[Regime, tuple[int, ...]] = {
        "within-session": (0, 1),
        "hold-in-session": (0, 1),
        "hold-out-session": (0, 1),
        "hold-out-subject": (0, 1),
    }

    def __init__(
        self,
        root: str | None = None,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        *,
        subset_tier: SubsetTier | None = None,
        test_subject: int | None = None,
        test_session: int | None = None,
        split: Split | None = None,
        label_mode: LabelMode | None = None,
        task: str | None = None,
        regime: Regime | None = None,
        fold: int | None = None,
        uniquify_channel_ids_with_subject: bool = True,
        uniquify_channel_ids_with_session: bool = False,
        dirname: str = "keles_byd_2024",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        # Resolve and validate constructor inputs before touching dataset records.
        self._dataset_dir = Path(root) / dirname
        self._disk_recording_stem_by_recording_id: dict[str, str] | None = None

        # XOR recording-source behavior:
        # - no recording_ids => use split-resolved benchmark recordings
        # - recording_ids provided => use the explicit subset as active recordings
        use_split_selection = recording_ids is None
        self._use_split_selection = use_split_selection
        if use_split_selection:
            if subset_tier is None:
                subset_tier = "full"
            if label_mode is None:
                label_mode = "binary"
            if task is None:
                task = "speech"
            if regime is None:
                regime = "within-session"
            if fold is None:
                fold = 0

            self.subset_tier = subset_tier
            self.label_mode = label_mode
            self.task = task
            self.regime = regime
            self.fold = fold
            self.test_subject = test_subject
            self.test_session = test_session
            self.split = split

            self._validate_split_args()
            self.h5_regime = H5_REGIME_BY_REGIME[self.regime]
            active_recording_ids = self._split_recording_ids()
        else:
            unexpected_split_args = [
                name
                for name, value in (
                    ("subset_tier", subset_tier),
                    ("test_subject", test_subject),
                    ("test_session", test_session),
                    ("split", split),
                    ("label_mode", label_mode),
                    ("task", task),
                    ("regime", regime),
                    ("fold", fold),
                )
                if value is not None
            ]
            if unexpected_split_args:
                raise ValueError(
                    "When recording_ids is provided (explicit-recording mode), benchmark selector args "
                    "must be omitted. Unexpected args: "
                    f"{', '.join(unexpected_split_args)}."
                )
            active_recording_ids = self._resolve_requested_recording_ids(recording_ids)

        if not active_recording_ids:
            raise ValueError(
                "No active recording_ids resolved for KelesBYD2024 construction."
            )

        storage_recording_ids = [
            self._resolve_storage_recording_id(recording_id)
            for recording_id in active_recording_ids
        ]

        super().__init__(
            dataset_dir=self._dataset_dir,
            recording_ids=storage_recording_ids,
            transform=transform,
            namespace_attributes=["subject.id", "channels.id"],
            **kwargs,
        )
        # Public dataset identity stays canonical even when disk files include
        # BYD-specific suffixes like "_behavior+ecephys".
        self._storage_recording_id_by_recording_id = dict(
            zip(active_recording_ids, storage_recording_ids, strict=True)
        )
        self._recording_ids = sorted(active_recording_ids)
        self._sampling_rate_hz: float | None = None

        # Configure subject/session-based channel-id prefixing behavior.
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )

    def get_sampling_intervals(self) -> dict[str, Interval]:
        """Return split-specific sampling intervals for this dataset instance."""
        if not self._use_split_selection:
            raise RuntimeError(
                "get_sampling_intervals is only available in benchmark mode."
            )
        intervals: dict[str, Interval] = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            intervals[rid] = rec.splits
        return intervals

    def get_domain_intervals(self) -> dict[str, Interval]:
        """Return full-domain intervals for active recordings."""
        return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

    @property
    def sampling_rate(self) -> float:
        """Return recording sampling rate in Hz from seeg_data.sampling_rate."""
        if self._sampling_rate_hz is not None:
            return self._sampling_rate_hz

        recording_id = self.recording_ids[0]
        try:
            sampling_rate = float(
                self.get_recording(recording_id).seeg_data.sampling_rate
            )
        except AttributeError as exc:
            raise AttributeError(
                "KelesBYD2024 seeg_data must expose sampling_rate; "
                f"missing for recording '{recording_id}'."
            ) from exc
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(
                "Invalid seeg_data.sampling_rate value "
                f"{sampling_rate!r} for recording '{recording_id}'."
            )
        self._sampling_rate_hz = sampling_rate
        return sampling_rate

    def get_channel_metadata(self, recording_id: str) -> dict[str, object]:
        """Return normalized channel metadata arrays for one recording."""
        rec = self.get_recording(recording_id)
        channels = rec.channels

        ids = np.asarray(channels.id).astype(str)
        names = np.asarray(channels.name).astype(str)
        included_mask = np.asarray(channels.included, dtype=bool)

        if len(names) != len(ids):
            raise ValueError(
                f"Channel name length mismatch for recording '{recording_id}': "
                f"len(names)={len(names)} vs len(ids)={len(ids)}"
            )
        if len(included_mask) != len(ids):
            raise ValueError(
                f"Channel mask length mismatch for recording '{recording_id}': "
                f"len(mask)={len(included_mask)} vs len(ids)={len(ids)}"
            )

        coordinate_frames = {
            "byd_mni152_ras": _stack_coordinate_frame(
                channels=channels,
                recording_id=recording_id,
                dataset_name="KelesBYD2024",
                frame_name="byd_mni152_ras",
                field_names=(
                    "coord_byd_mni152_r",
                    "coord_byd_mni152_a",
                    "coord_byd_mni152_s",
                ),
                expected_length=len(ids),
            )
        }

        arrays: dict[str, np.ndarray | str] = {
            "ids": ids,
            "names": names,
            "included_mask": included_mask,
            "coordinate_frames": coordinate_frames,
            "indices": np.arange(len(ids), dtype=int),
        }
        optional_string_keys = (
            "label_dkt",
            "label_destrieux",
            "provided_location",
            "trajectory_role",
            "coordinate_assumption",
            "label_status",
            "qc_status",
        )
        for optional_key in optional_string_keys:
            if not hasattr(channels, optional_key):
                continue
            values = np.asarray(getattr(channels, optional_key)).astype(str).reshape(-1)
            if len(values) != len(ids):
                raise ValueError(
                    f"Channel {optional_key} length mismatch for recording "
                    f"'{recording_id}': len({optional_key})={len(values)} vs "
                    f"len(ids)={len(ids)}"
                )
            arrays[optional_key] = values
        return arrays

    def get_neural_signal_metadata(self, recording_id: str) -> dict[str, str | float]:
        storage_recording_id = self._storage_recording_id_by_recording_id.get(
            recording_id, recording_id
        )
        return _read_seeg_signal_metadata(
            self._dataset_dir / f"{storage_recording_id}.h5",
            recording_id=recording_id,
        )

    def get_recording(self, recording_id: str, _namespace: str = "") -> Data:
        """Load one recording by canonical BYD recording id."""
        storage_recording_id = self._storage_recording_id_by_recording_id[recording_id]
        if hasattr(self, "_data_objects"):
            data = copy.deepcopy(self._data_objects[storage_recording_id])
        else:
            fpath = self._filepaths[storage_recording_id]
            data = Data.from_hdf5(h5py.File(fpath))

        self.get_recording_hook(data)
        if _namespace:
            self.apply_namespace(data, _namespace + "/")
        return data

    def get_recording_hook(self, data: Data):
        """Apply split-specific channel inclusion mask when available."""
        if not self._use_split_selection:
            # Explicit-recording mode does not apply benchmark split routing.
            super().get_recording_hook(data)
            return

        recording_id = data.session.id
        channel_split_paths = self._channel_split_attr_paths()
        interval_paths = self._interval_attr_paths()

        # Benchmark mode requires both the channel mask and intervals.
        try:
            channel_mask, _ = self._resolve_first_nested_attribute(
                data, channel_split_paths
            )
            split_interval, _ = self._resolve_first_nested_attribute(
                data, interval_paths
            )
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                "Missing required benchmark selector attributes for KelesBYD2024 "
                f"recording '{recording_id}'. Expected channel mask at "
                f"{channel_split_paths}, "
                f"and split intervals at {interval_paths}."
            ) from exc
        data.channels.included = channel_mask
        data.splits = split_interval
        super().get_recording_hook(data)

    def describe_selection(self) -> dict[str, object]:
        """Return a compact debug summary of the resolved benchmark selection."""
        summary: dict[str, object] = {
            "uses_split_selection": self._use_split_selection,
            "active_recording_ids": list(self.recording_ids),
        }
        if not self._use_split_selection:
            return summary

        # Expose resolved split internals to make dataset/debug logs self-explanatory.
        summary.update(
            {
                "subset_tier": self.subset_tier,
                "label_mode": self.label_mode,
                "task": self.task,
                "regime": self.regime,
                "h5_regime": self.h5_regime,
                "fold": self.fold,
                "split": self.split,
                "test_subject": self.test_subject,
                "test_session": self.test_session,
                "test_recording_id": _to_recording_id(
                    self.test_subject, self.test_session
                ),
                "split_key": self._split_key(),
            }
        )
        return summary

    # Path/key builders.
    def _split_key(self) -> str:
        """Return the canonical key shared under `splits` and `channel_splits`."""
        return (
            f"{self.subset_tier}${self.label_mode}${self.h5_regime}${self.task}$"
            f"fold{self.fold}${self.split}"
        )

    def _interval_attr_path(self) -> str:
        # Primary split interval path under data.splits.
        return f"splits.{self._split_key()}"

    def _channel_split_attr_path(self) -> str:
        # Return the primary path for split-specific channel masks.
        return f"channel_splits.{self._split_key()}"

    def _legacy_split_key(self) -> str:
        legacy_task = LEGACY_H5_TASK_BY_TASK.get(self.task, self.task)
        return f"{self.label_mode}_{legacy_task}_fold{self.fold}_{self.split}"

    def _interval_attr_paths(self) -> list[str]:
        return [self._interval_attr_path(), self._legacy_split_key()]

    def _channel_split_attr_paths(self) -> list[str]:
        return [
            self._channel_split_attr_path(),
            f"channels.included_{self._legacy_split_key()}",
        ]

    def _resolve_first_nested_attribute(
        self, data: Data, paths: list[str]
    ) -> tuple[object, str]:
        for path in paths:
            try:
                return data.get_nested_attribute(path), path
            except (AttributeError, KeyError):
                continue
        raise KeyError(f"None of the candidate attribute paths exist: {paths}")

    def _validate_split_args(self) -> None:
        # Keep constructor strict so invalid benchmark configs fail immediately.
        if self.subset_tier not in VALID_SUBSET_TIERS:
            raise ValueError(
                f"Invalid subset_tier '{self.subset_tier}'. Must be one of {VALID_SUBSET_TIERS}."
            )
        if self.label_mode not in VALID_LABEL_MODES:
            raise ValueError(
                f"Invalid label_mode '{self.label_mode}'. Must be one of {VALID_LABEL_MODES}."
            )
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Invalid task '{self.task}'. Must be one of {VALID_TASKS}."
            )
        if self.regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{self.regime}'. Must be one of {VALID_REGIMES}."
            )
        if not isinstance(self.fold, Integral) or isinstance(self.fold, bool):
            raise TypeError(f"fold must be an int, got {type(self.fold).__name__}.")

        allowed_folds = self._ALLOWED_FOLDS_BY_REGIME[self.regime]
        if self.fold not in allowed_folds:
            allowed_values = " or ".join(str(value) for value in allowed_folds)
            raise ValueError(
                f"Fold for regime '{self.regime}' must be {allowed_values}, got {self.fold}."
            )
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{self.split}'. Must be one of {VALID_SPLITS}."
            )
        if not isinstance(self.test_subject, Integral) or isinstance(
            self.test_subject, bool
        ):
            raise TypeError(
                f"test_subject must be an int, got {type(self.test_subject).__name__}."
            )
        if not isinstance(self.test_session, Integral) or isinstance(
            self.test_session, bool
        ):
            raise TypeError(
                f"test_session must be an int, got {type(self.test_session).__name__}."
            )

    @classmethod
    def num_folds_for_regime(cls, regime: str) -> int:
        """Return the number of available folds for one regime."""
        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {VALID_REGIMES}."
            )
        return len(cls._ALLOWED_FOLDS_BY_REGIME[regime])

    def _resolve_requested_recording_ids(self, recording_ids: list[str]) -> list[str]:
        # Normalize explicit recording-id subsets to a stable, de-duplicated order.
        if not recording_ids:
            raise ValueError(
                "When using explicit-recording mode, recording_ids must contain at least one id."
            )
        ids = recording_ids
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(
                "When using explicit-recording mode, recording_ids must contain at least one id."
            )

        # Parse each id once so errors are raised consistently at construction.
        for rid in ids:
            _from_recording_id(rid)
        return ids

    def _full_subset_recording_ids_from_disk(self) -> list[str]:
        # Full tier should use all available recording ids in dataset_dir.
        return sorted(self._disk_stem_by_recording_id().keys())

    def _disk_stem_by_recording_id(self) -> dict[str, str]:
        # Build one canonical-id -> on-disk-stem map so split selection and file
        # loading agree even when BYD files carry an extra suffix.
        if self._disk_recording_stem_by_recording_id is not None:
            return self._disk_recording_stem_by_recording_id

        resolved: dict[str, str] = {}
        for path in sorted(self._dataset_dir.glob("*.h5")):
            recording_id = _normalize_recording_stem(path.stem)
            try:
                _from_recording_id(recording_id)
            except ValueError:
                # Ignore non-recording H5 artifacts in the same directory.
                continue
            existing = resolved.get(recording_id)
            if existing is not None and existing != path.stem:
                raise ValueError(
                    "Ambiguous BYD recording-id mapping: multiple H5 stems resolve to "
                    f"canonical id '{recording_id}' under dataset_dir '{self._dataset_dir}'."
                )
            resolved[recording_id] = path.stem
        self._disk_recording_stem_by_recording_id = resolved
        return resolved

    def _resolve_storage_recording_id(self, recording_id: str) -> str:
        stem = self._disk_stem_by_recording_id().get(recording_id)
        if stem is None:
            raise FileNotFoundError(
                f"Recording '{recording_id}' does not exist under dataset_dir "
                f"'{self._dataset_dir}'."
            )
        return stem

    def _eligible_recording_ids_for_subset_tier(self) -> list[str]:
        if self.subset_tier == "full":
            recording_ids = self._full_subset_recording_ids_from_disk()
        else:
            recording_ids = []
        if not recording_ids:
            raise ValueError(
                f"No eligible recording_ids found for subset_tier '{self.subset_tier}' "
                f"under dataset_dir '{self._dataset_dir}'."
            )
        return sorted(recording_ids)

    def _split_recording_ids(self) -> list[str]:
        """Resolve split-participating recording ids for constructor inputs."""
        test_recording_id = _to_recording_id(self.test_subject, self.test_session)
        eligible_recording_ids = self._eligible_recording_ids_for_subset_tier()
        if test_recording_id not in eligible_recording_ids:
            requested_pair = (self.test_subject, self.test_session)
            raise ValueError(
                f"Target pair {requested_pair} is not eligible for subset_tier "
                f"'{self.subset_tier}'."
            )

        if self.regime == "within-session":
            resolved_ids = [test_recording_id]
        elif self.regime == "hold-in-session":
            if self.split == "train":
                resolved_ids = list(eligible_recording_ids)
            else:
                resolved_ids = [test_recording_id]
        elif self.regime == "hold-out-session":
            if self.split == "train":
                resolved_ids = [
                    rid for rid in eligible_recording_ids if rid != test_recording_id
                ]
            else:
                resolved_ids = [test_recording_id]
        else:
            if self.split == "train":
                resolved_ids = [
                    rid
                    for rid in eligible_recording_ids
                    if _from_recording_id(rid)[0] != self.test_subject
                ]
            else:
                resolved_ids = [test_recording_id]

        resolved_ids = sorted(set(resolved_ids))
        if self.split == "train" and not resolved_ids:
            raise ValueError(
                "No training recording_ids resolved after applying regime/subset filters."
            )
        return resolved_ids
