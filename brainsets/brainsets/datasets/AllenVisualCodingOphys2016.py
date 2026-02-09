from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, CalciumImagingDatasetMixin


class AllenVisualCodingOphys2016(CalciumImagingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: Optional[Literal["poyo_plus"]] = "poyo_plus",
        dirname: str = "allen_visual_coding_ophys_2016",
        **kwargs,
    ):

        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "rois.id"],
            **kwargs,
        )

        self.split_type = split_type

        # ROI IDs are unique across sessions in this dataset
        self.calcium_imaging_dataset_mixin_uniquify_roi_ids = False

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):

        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if self.split_type is None:
            raise ValueError("Only split=None supported when split_type is None.")

        elif self.split_type == "poyo_plus":
            key = f"{split}_domain"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }

        else:
            raise ValueError(f"Invalid split_type '{self.split_type}'.")
