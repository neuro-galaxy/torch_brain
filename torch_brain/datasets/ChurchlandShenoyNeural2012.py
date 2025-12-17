from typing import Optional, Literal
from pathlib import Path
from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from .dataset import Dataset, SpikingDatasetMixin


class ChurchlandShenoyNeural2012(SpikingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        dirname: str = "churchland_shenoy_neural_2012",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        split_type: Optional[Literal["cursor_velocity"]] = "cursor_velocity",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

        self.split_type = split_type

    def get_sampling_intervals(self, split: Literal["train", "valid", "test"]):
        domain_key = "domain" if split is None else f"{split}_domain"
        ans = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            ans[rid] = getattr(data, domain_key)

            if self.split_type == "cursor_velocity":
                ans[rid] = ans[rid] & data.cursor.domain & data.spikes.domain

        return ans

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique unit ids across sessions
        # so we prefix the unit ids with the session id to ensure uniqueness
        data.units.id = np_string_prefix(
            f"{data.session.id}/",
            data.units.id.astype(str),
        )

        super().get_recording_hook(data)
