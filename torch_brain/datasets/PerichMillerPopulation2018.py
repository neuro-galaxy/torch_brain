from typing import Optional, Literal
from pathlib import Path
from .dataset import TemporalDataset, SpikingDatasetMixin
from torch_brain.transforms import TransformType


class PerichMillerPopulation2018(SpikingDatasetMixin, TemporalDataset):
    def __init__(
        self,
        root: str,
        name: str = "perich_miller_population_2018",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / name,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

        self.split = split

    def get_sampling_intervals(self):
        if self.split is None:
            domain_key = "domain"
        else:
            domain_key = f"{self.split}_domain"

        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }


if __name__ == "__main__":
    ds = PerichMillerPopulation2018(root="../brainsets/data/processed", split="train")
    print(ds.get_unit_ids())
    print(ds.get_sampling_intervals())
