from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

from torch_brain.data import Data
from torch_brain.datasets.dataset import Dataset
from torch_brain.utils import np_string_prefix

from ._utils import get_processed_dir

FoldType = Literal["intrasession", "intersubject", "intersession"]
VALID_FOLD_TYPES = get_args(FoldType)


class KempSleepEDF2013(Dataset):
    """Sleep-EDF Database Expanded containing 197 whole-night polysomnographic sleep recordings.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare kemp_sleep_edf_2013

    Args:
        root: Root directory for the dataset. Defaults to ``processed_dir`` from brainsets config.
        recording_ids: List of recording IDs to load.
        transform: Data transformation to apply.
        uniquify_channel_ids: Whether to prefix channel IDs with session ID to ensure uniqueness. Defaults to True.
        fold_number: The cross-validation fold index (0 to 2 for a 3-fold split). Defaults to 0.
        fold_type: The splitting strategy. Must be one of:
            - \"intrasession\": Epoch-level stratified split within each session.
            - \"intersubject\": Subject-level split (subjects are assigned to train/valid/test).
            - \"intersession\": Session-level split (subject-session pairs are assigned to train/valid/test).
            Defaults to \"intrasession\".
        dirname: Subdirectory for the dataset. Defaults to "kemp_sleep_edf_2013".
    """

    def __init__(
        self,
        root: str | None = None,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        uniquify_channel_ids: bool = True,
        fold_number: int = 0,
        fold_type: FoldType = "intrasession",
        dirname: str = "kemp_sleep_edf_2013",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.uniquify_channel_ids = uniquify_channel_ids

        if fold_number is None or not (0 <= fold_number < 3):
            raise ValueError(
                f"Fold number must be an integer between 0 and 2, got {fold_number}"
            )

        self.fold_number = fold_number
        self.fold_type = fold_type

        if fold_type not in VALID_FOLD_TYPES:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. Must be one of {VALID_FOLD_TYPES}."
            )

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ):

        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intrasession":
            key = f"splits.fold_{self.fold_number}.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }
        elif self.fold_type in ("intersubject", "intersession"):
            key = f"splits.{self.fold_type}_fold_{self.fold_number}_assignment"
            fallback_key = f"splits.fold_{self.fold_number}_assignment"
            result = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                try:
                    assignment = str(rec.get_nested_attribute(key))
                except (AttributeError, KeyError):
                    assignment = str(rec.get_nested_attribute(fallback_key))
                if assignment == split:
                    result[rid] = rec.domain
            return result

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique channel ids across sessions
        # so we prefix the channel ids with the session id to ensure uniqueness
        if self.uniquify_channel_ids:
            data.channels.id = np_string_prefix(f"{data.session.id}/", data.channels.id)

        super().get_recording_hook(data)
