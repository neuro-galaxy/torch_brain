from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from torch_brain.data import Data, Interval
from torch_brain.datasets.dataset import Dataset
from torch_brain.datasets.mixins import MultiChannelDatasetMixin
from torch_brain.utils.split import _get_integer_hash_from_string

OpenNeuroSplitType = Literal["intrasession", "intersubject", "intersession"]

_VALID_SPLIT_TYPES = get_args(OpenNeuroSplitType)


class OpenNeuroDataset(MultiChannelDatasetMixin, Dataset):
    """
    Base class for OpenNeuro datasets.

    This class provides an interface for loading, representing, and manipulating
    OpenNeuro datasets using the MultiChannelDatasetMixin and the Dataset interface.
    It supports various splitting strategies for machine learning workflows, notably
    'intrasession', 'intersubject', and 'intersession' splits.

    Args:
        root: Root directory containing processed OpenNeuro dataset artifacts.
        dataset_dir: Relative dataset directory within the root path.
        split_type: The split strategy to use, must be one of
            'intrasession', 'intersubject', or 'intersession'.
        recording_ids: List of recording IDs to include,
            or None to use all available recordings.
        transform: Optional sample transform.
        uniquify_channel_ids_with_subject: Whether to prefix channel IDs with
            ``subject.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``False``.
        uniquify_channel_ids_with_session: Whether to prefix channel IDs with
            ``session.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``True``.
        task_paradigm: The task paradigm of the dataset. Depends on the dataset.
            Defaults to None.
        split_ratios: Tuple of three floats (train, val, test) whose sum must be 1.0.
            Specifies the proportion of the dataset to use for the train, validation,
            and test splits, respectively. All ratios must be in [0, 1] and their sum must be 1.0.
            If the sum does not equal 1.0, a ValueError is raised.
        seed: The seed for the random number generator. Used for computing splits in
        intersubject and intersession mode. Defaults to 42.
    """

    def __init__(
        self,
        root: str,
        dataset_dir: str,
        split_type: OpenNeuroSplitType,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        uniquify_channel_ids_with_subject: bool = False,
        uniquify_channel_ids_with_session: bool = True,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        **kwargs,
    ):
        if split_type not in _VALID_SPLIT_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. Must be one of {_VALID_SPLIT_TYPES}."
            )
        self.split_type = split_type

        super().__init__(
            dataset_dir=Path(root) / dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        # Configure subject/session-based channel-id prefixing behavior.
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )

        self.split_ratios = self._validate_split_ratios(split_ratios)

        self.seed = seed

    def _validate_split_ratios(
        self, split_ratios: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Validate the split ratios.

        Args:
            split_ratios: Tuple of three floats (train, val, test), whose sum must be 1.0.

        Returns:
            Tuple[float, float, float]: The validated split ratios.

        Raises:
            ValueError: If any ratio is negative, or sum does not equal 1.0.
        """
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError("`split_ratios` cannot contain negative values")
        if not np.isclose(sum(split_ratios), 1.0, atol=1e-8):
            raise ValueError("The sum of `split_ratios` must be exactly 1.0")
        return split_ratios

    def get_sampling_intervals(
        self,
        split: Literal["train", "val", "test"] | None = None,
    ) -> dict[str, Interval]:
        """
        Retrieve the sampling intervals for each recording according to the specified split.

        If `split` is None, returns the full interval domain for every recording for unrestricted sampling.
        If a split ("train", "val", or "test") is provided, returns only the intervals (within each recording)
        eligible for sampling under the current split type and task paradigm.

        The selection of intervals is determined according to:
        - The current `self.split_type` (intrasession, intersubject, or intersession).
        - Whether a `self.task_paradigm` is specified, which influences the interval extraction.

        Args:
            split: One of "train", "val", or "test" to select intervals corresponding to that split,
                or None to retrieve the entire domain for all recordings.

        Returns:
            Dictionary mapping recording IDs to their valid Interval objects for sampling in the given split
            (or full Interval domain if split is None).

        Raises:
            ValueError: If the requested `split` or the dataset's `split_type` is not recognized/supported.
            KeyError: If a required split or assignment attribute is missing in a recording.

        Notes:
            - Intervals are defined based on recording domains and split logic.

        """
        if split is None:
            return super().get_sampling_intervals()

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split {split!r}. Must be one of 'train', 'val', 'test'."
            )

        intervals = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            intervals[rid] = self.get_default_sampling_intervals(rec, split)
        return intervals

    def get_default_sampling_intervals(
        self,
        recording: Data,
        split: Literal["train", "val", "test"],
    ) -> Interval:
        """
        Get the default sampling intervals for a given split. These intervals are behavior agnostic, meaning they
        do not take into account any task or behavioral (event/label) annotations when creating the train, val,
        and test splits—interval assignment is performed solely based on session or subject, not on in-task structure.

        Notes:
        - For split_type == "intrasession", intervals are split causally into train, val, and test based on split_ratios.
        - For split_type == "intersubject" or "intersession", only the assigned recordings are included for each split
        (using k-fold assignment); all others return an empty interval.
        """
        if self.split_type == "intrasession":
            starts = np.asarray(recording.domain.start)
            ends = np.asarray(recording.domain.end)
            durations = ends - starts

            train_ends = starts + durations * self.split_ratios[0]
            val_ends = train_ends + durations * self.split_ratios[1]
            test_ends = val_ends + durations * self.split_ratios[2]

            if split == "train":
                return Interval(start=starts, end=train_ends)
            elif split == "val":
                return Interval(start=train_ends, end=val_ends)
            elif split == "test":
                return Interval(start=val_ends, end=test_ends)
            raise ValueError(
                f"Invalid split {split!r}. Must be one of 'train', 'val', 'test'."
            )

        elif self.split_type == "intersubject" or self.split_type == "intersession":
            if self.split_type == "intersubject":
                string_id = recording.subject.id
            elif self.split_type == "intersession":
                string_id = f"{recording.subject.id}_{recording.session.id}"

            base_str = f"{string_id}_{self.seed}"
            hash_int = _get_integer_hash_from_string(base_str)
            normalized_hash = (hash_int % 10000) / 10000.0

            if normalized_hash < self.split_ratios[0]:
                assignment = "train"
            elif normalized_hash < (self.split_ratios[0] + self.split_ratios[1]):
                assignment = "val"
            else:
                assignment = "test"

            if assignment == split:
                return recording.domain
            else:
                return Interval(start=np.array([]), end=np.array([]))

        raise ValueError(
            f"Invalid split_type '{self.split_type}'. Must be one of {_VALID_SPLIT_TYPES}."
        )
