from typing import Optional, Literal
from pathlib import Path
from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from .dataset import Dataset, SpikingDatasetMixin


class PerichMillerPopulation2018(SpikingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        dirname: str = "perich_miller_population_2018",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

    def get_sampling_intervals(self, split: Literal["train", "valid", "test"]):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique unit ids across sessions
        # so we prefix the unit ids with the session id to ensure uniqueness
        data.units.id = np_string_prefix(
            f"{data.session.id}/",
            data.units.id.astype(str),
        )

        super().get_recording_hook(data)
