from __future__ import annotations

import re
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import Literal, get_args

import h5py
import numpy as np

from torch_brain.data import Data, Interval

from ._pippi import (
    PIPPI_SUBSET_TIERS,
    pippi_subject_numbers_for_subset_tier,
)
from ._utils import get_processed_dir
from .dataset import Dataset
from .mixins import MultiChannelDatasetMixin

SubsetTier = Literal["full", "high-cov", "low-cov"]
LabelMode = Literal["binary", "multiclass"]
Regime = Literal[
    "within-session",
    "hold-in-session",
    "hold-out-session",
    "hold-out-subject",
]
Split = Literal["train", "val", "test"]

VALID_SUBSET_TIERS = PIPPI_SUBSET_TIERS
VALID_LABEL_MODES = get_args(LabelMode)
VALID_REGIMES = get_args(Regime)
VALID_SPLITS = get_args(Split)

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

_RECORDING_ID_RE = re.compile(
    r"^sub-(?P<subject>\d+)_ses-iemu_task-film_acq-(?P<acq>[A-Za-z0-9]+)_run-(?P<run>\d+)$"
)


def _stack_coordinate_frame(
    *,
    channels,
    recording_id: str,
    dataset_name: str,
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


def _to_recording_id(subject: int, acquisition: str, run: int) -> str:
    if (
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        or isinstance(run, bool)
        or not isinstance(run, int)
        or run < 1
        or not isinstance(acquisition, str)
        or not re.fullmatch(r"[A-Za-z0-9]+", acquisition)
    ):
        raise ValueError(
            "Invalid Pippi recording-id components: "
            f"subject={subject!r}, acquisition={acquisition!r}, run={run!r}."
        )
    return f"sub-{subject:02d}_ses-iemu_task-film_acq-{acquisition}_run-{run}"


def _from_recording_id(recording_id: str) -> tuple[int, str, int]:
    match = _RECORDING_ID_RE.fullmatch(recording_id)
    if match is None:
        raise ValueError(
            f"Invalid recording_id '{recording_id}'. Expected "
            "'sub-<subject>_ses-iemu_task-film_acq-<acq>_run-<run>'."
        )
    return int(match.group("subject")), match.group("acq"), int(match.group("run"))


class BerezutskayaPippi2022(MultiChannelDatasetMixin, Dataset):
    """Pippi movie SEEG dataset prepared from OpenNeuro ds003688.

    Each instance operates in exactly one of two mutually-exclusive modes:
    - Benchmark mode (`recording_ids=None`): active recordings are resolved
      from Pippi selectors and within-session split keys.
    - Explicit-recording mode (`recording_ids` provided): active recordings come
      directly from that subset and split selectors must be omitted.
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
        dirname: str = "berezutskaya_pippi_2022",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        self._dataset_dir = Path(root) / dirname

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
                    "When recording_ids is provided (explicit-recording mode), split-selection args "
                    "must be omitted. Unexpected args: "
                    f"{', '.join(unexpected_split_args)}."
                )
            active_recording_ids = self._resolve_requested_recording_ids(recording_ids)

        if not active_recording_ids:
            raise ValueError(
                "No active recording_ids resolved for BerezutskayaPippi2022 construction."
            )

        super().__init__(
            dataset_dir=self._dataset_dir,
            recording_ids=active_recording_ids,
            transform=transform,
            namespace_attributes=["subject.id", "channels.id"],
            **kwargs,
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )
        self._sampling_rate_by_recording_id: dict[str, float] = {}

    def get_sampling_intervals(self) -> dict[str, Interval]:
        if not self._use_split_selection:
            raise RuntimeError(
                "get_sampling_intervals is only available in benchmark mode."
            )
        return {rid: self.get_recording(rid).splits for rid in self.recording_ids}

    def get_domain_intervals(self) -> dict[str, Interval]:
        return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

    @property
    def sampling_rate(self) -> float:
        """Recording sampling rate in Hz."""
        raise RuntimeError(
            "BerezutskayaPippi2022 has mixed per-recording sampling rates. "
            "Use get_sampling_rate(recording_id) instead of dataset.sampling_rate."
        )

    def get_sampling_rate(self, recording_id: str) -> float:
        """Return one recording sampling rate in Hz."""
        cached_sampling_rate = self._sampling_rate_by_recording_id.get(recording_id)
        if cached_sampling_rate is not None:
            return cached_sampling_rate

        try:
            recording_sampling_rate = float(
                self.get_recording(recording_id).seeg_data.sampling_rate
            )
        except AttributeError as exc:
            raise AttributeError(
                "Missing required seeg_data.sampling_rate for BerezutskayaPippi2022 "
                f"recording '{recording_id}'."
            ) from exc

        if not np.isfinite(recording_sampling_rate) or recording_sampling_rate <= 0.0:
            raise ValueError(
                "Invalid seeg_data.sampling_rate for BerezutskayaPippi2022 "
                f"recording '{recording_id}': {recording_sampling_rate!r}"
            )

        self._sampling_rate_by_recording_id[recording_id] = recording_sampling_rate
        return recording_sampling_rate

    def get_channel_metadata(self, recording_id: str) -> dict[str, object]:
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

        arrays: dict[str, object] = {
            "ids": ids,
            "names": names,
            "included_mask": included_mask,
            "coordinate_frames": {
                "acpc": _stack_coordinate_frame(
                    channels=channels,
                    recording_id=recording_id,
                    dataset_name="BerezutskayaPippi2022",
                    field_names=("coord_acpc_x", "coord_acpc_y", "coord_acpc_z"),
                    expected_length=len(ids),
                )
            },
            "indices": np.arange(len(ids), dtype=int),
        }
        optional_string_keys = (
            "group",
            "hemisphere",
            "label_dkt",
            "label_destrieux",
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
        return _read_seeg_signal_metadata(
            self._dataset_dir / f"{recording_id}.h5",
            recording_id=recording_id,
        )

    def get_recording_hook(self, data: Data):
        # Explicit-recording mode does not apply benchmark split routing.
        if not self._use_split_selection:
            super().get_recording_hook(data)
            return

        recording_id = data.session.id
        channel_split_path = self._channel_split_attr_path()
        interval_path = self._interval_attr_path()
        try:
            channel_mask = data.get_nested_attribute(channel_split_path)
            split_interval = data.get_nested_attribute(interval_path)
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                "Missing required split-selection attributes for BerezutskayaPippi2022 "
                f"recording '{recording_id}'. Expected channel mask at "
                f"'{channel_split_path}', and split intervals at '{interval_path}'."
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
                "split_key": self._split_key(),
            }
        )
        return summary

    def _split_key(self) -> str:
        return (
            f"{self.subset_tier}${self.label_mode}${self.h5_regime}${self.task}$"
            f"fold{self.fold}${self.split}"
        )

    def _interval_attr_path(self) -> str:
        return f"splits.{self._split_key()}"

    def _channel_split_attr_path(self) -> str:
        return f"channel_splits.{self._split_key()}"

    def _validate_split_args(self) -> None:
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
        if int(self.test_session) < 1:
            raise ValueError(f"test_session must be >= 1, got {self.test_session}.")

    @classmethod
    def num_folds_for_regime(cls, regime: str) -> int:
        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {VALID_REGIMES}."
            )
        return len(cls._ALLOWED_FOLDS_BY_REGIME[regime])

    def _resolve_requested_recording_ids(self, recording_ids: list[str]) -> list[str]:
        if not recording_ids:
            raise ValueError(
                "When using explicit-recording mode, recording_ids must contain at least one id."
            )
        ids = sorted(set(recording_ids))
        for rid in ids:
            _from_recording_id(rid)
        return ids

    def _full_subset_recording_ids_from_disk(self) -> list[str]:
        resolved_ids: set[str] = set()
        for path in sorted(self._dataset_dir.glob("*.h5")):
            recording_id = path.stem
            try:
                _from_recording_id(recording_id)
            except ValueError:
                continue
            resolved_ids.add(recording_id)
        return sorted(resolved_ids)

    def _eligible_recording_ids_for_subset_tier(self) -> list[str]:
        recording_ids = self._full_subset_recording_ids_from_disk()
        subject_numbers = pippi_subject_numbers_for_subset_tier(self.subset_tier)
        if subject_numbers is not None:
            # Coverage-based tiers are defined at the subject level, so every
            # run for an eligible subject participates in that subset.
            recording_ids = [
                recording_id
                for recording_id in recording_ids
                if _from_recording_id(recording_id)[0] in subject_numbers
            ]
        if not recording_ids:
            raise ValueError(
                f"No eligible recording_ids found for subset_tier '{self.subset_tier}' "
                f"under dataset_dir '{self._dataset_dir}'."
            )
        return recording_ids

    def _resolve_test_recording_id(self, eligible_recording_ids: list[str]) -> str:
        matches = []
        for recording_id in eligible_recording_ids:
            subject_number, _acquisition, run = _from_recording_id(recording_id)
            if subject_number == int(self.test_subject) and run == int(
                self.test_session
            ):
                matches.append(recording_id)
        if not matches:
            raise ValueError(
                f"No eligible Pippi recording found for "
                f"(test_subject={self.test_subject}, test_session={self.test_session})."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous Pippi selection for "
                f"(test_subject={self.test_subject}, test_session={self.test_session}): {matches}"
            )
        return matches[0]

    def _split_recording_ids(self) -> list[str]:
        eligible_recording_ids = self._eligible_recording_ids_for_subset_tier()
        test_recording_id = self._resolve_test_recording_id(eligible_recording_ids)
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
                    if _from_recording_id(rid)[0] != int(self.test_subject)
                ]
            else:
                resolved_ids = [test_recording_id]

        resolved_ids = sorted(set(resolved_ids))
        if self.split == "train" and not resolved_ids:
            raise ValueError(
                "No training recording_ids resolved after applying regime/subset filters."
            )
        return resolved_ids
