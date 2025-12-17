from typing import Optional, Literal
from pathlib import Path
from torch_brain.transforms import TransformType
from torch_brain.utils import numpy_string_prefix
from temporaldata import Data

from dataset import Dataset, SpikingDatasetMixin, MultiDataset


class PerichMillerPopulation2018(SpikingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        dirname: str = "perich_miller_population_2018",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

        self.split = split

    def get_sampling_intervals(self):
        domain_key = "domain" if self.split is None else f"{self.split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique unit ids across sessions
        # so we prefix the unit ids with the session id to ensure uniqueness
        data.units.id = numpy_string_prefix(
            f"{data.session.id}/",
            data.units.id.astype(str),
        )

        super().get_recording_hook(data)


if __name__ == "__main__":
    ds = PerichMillerPopulation2018(root="../brainsets/data/processed", split="train")
    print(ds.recording_ids)
    print(ds.get_unit_ids())
    # print(ds.get_subject_ids())

    mds = MultiDataset({"pm": ds})
    print(mds.recording_ids)
    print(mds.datasets["pm"].get_unit_ids())
    # print(mds.get_sampling_intervals())

    # print(ds.get_sampling_intervals())
