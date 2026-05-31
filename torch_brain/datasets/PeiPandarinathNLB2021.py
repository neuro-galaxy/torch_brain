from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.datasets import Dataset, SpikingDatasetMixin

from ._utils import get_processed_dir


class PeiPandarinathNLB2021(SpikingDatasetMixin, Dataset):
    """
    Curated spiking neural activity datasets from the Neural Latents Benchmark
    2021 (NLB'21).

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare pei_pandarinath_nlb_2021

    """

    def __init__(
        self,
        root: Optional[str] = None,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "pei_pandarinath_nlb_2021",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "units.id"],
            **kwargs,
        )

        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
