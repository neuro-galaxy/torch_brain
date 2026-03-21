from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class PerichMillerPopulation2018(SpikingDatasetMixin, Dataset):
    """
    Motor cortex (M1 and PMd) spiking activity and reaching kinematics from four macaques
    performing center-out and random target reaching tasks. The monkeys were trained to move a cursor from a central target to one of eight peripheral targets arranged in a circle.


    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare perich_miller_population_2018``.

    **Tasks:** Center-Out and Random Target

    **Brain Regions:** M1, PMd

    **Dataset Statistics**

    - **Subjects:** 4
    - **Total Sessions:** 111 (84 Center-Out, 27 Random Target)
    - **Total Units:** 10,410
    - **Events:** ~11.1M spikes, ~15.5M behavioral timestamps

    **References**

    Perich, M. G., Miller, L. E., Azabou, M., & Dyer, E. L.
    *Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys.*
    `Neuron <https://doi.org/10.1016/j.neuron.2018.09.030>`_.
    Dataset: `Dandiset 000688 <https://doi.org/10.48324/dandi.000688/0.250122.1735>`_.

    Args:
        root (str): Root directory for the dataset.
        recording_ids (list[str], optional): List of recording IDs to load.
        transform (Callable, optional): Data transformation to apply.
        dirname (str, optional): Subdirectory for the dataset. Defaults to "perich_miller_population_2018".
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "perich_miller_population_2018",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
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
