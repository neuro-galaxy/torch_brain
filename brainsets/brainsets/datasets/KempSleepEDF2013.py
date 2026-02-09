from typing import Callable, Optional, Literal
from pathlib import Path
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from torch_brain.dataset import Dataset


class KempSleepEDF2013(Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        split_type: Optional[Literal["fold_0", "fold_1", "fold_2"]] = "fold_0",
        dirname: str = "kemp_sleep_edf_2013",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.uniquify_channel_ids = uniquify_channel_ids
        self.split_type = split_type

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):

        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if self.split_type is None:
            raise ValueError("Only split=None supported when split_type is None.")

        if self.split_type not in ["fold_0", "fold_1", "fold_2"]:
            raise ValueError(
                f"Invalid split_type '{self.split_type}'."
                " Must be one of ['fold_0', 'fold_1', 'fold_2'] or None."
            )

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of 'train', 'valid', 'test', or None."
            )

        key = f"{self.split_type}.{split}"
        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique channel ids across sessions
        # so we prefix the channel ids with the session id to ensure uniqueness
        if self.uniquify_channel_ids:
            data.channels.id = np_string_prefix(f"{data.session.id}/", data.channels.id)

        super().get_recording_hook(data)
