from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin

from ._utils import get_processed_dir


class ChurchlandShenoyNeural2012(SpikingDatasetMixin, Dataset):
    """
    Motor cortex (M1 and PMd) spiking activity and reaching kinematics from 2 monkeys
    performing center-out reaching tasks with right hand.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare churchland_shenoy_neural_2012``.

    **Tasks:** Center-Out

    **Brain Regions:** M1, PMd

    **Dataset Statistics**

    - **Subjects:** 2
    - **Total Sessions:** 10
    - **Total Units:** 1,911
    - **Events:** ~739M spikes, ~85M behavioral timestamps

    **Links**

    - Paper: `Churchland et al. (2012) – Nature <https://www.nature.com/articles/nature11129>`_
    - Dataset: `Dandiset 000070 <https://dandiarchive.org/dandiset/000070>`_

    **Reference**

    Churchland, M., Cunningham, J. P., Kaufman, M. T., Foster, J. D.,
    Nuyujukian, P., Ryu, S. I., & Shenoy, K. V.
    *Neural population dynamics during reaching.*
    `DANDI Archive Dataset <https://doi.org/10.48324/dandi.000070/0.251218.1714>`_,
    Version 0.251218.1714.

    Args:
        root (str, optional): Root directory for the dataset. Defaults to ``processed_dir`` from brainsets config.
        recording_ids (list[str], optional): List of recording IDs to load.
        transform (Callable, optional): Data transformation to apply.
        split_type (str, optional): Which split type to use. Defaults to "cursor_velocity".
        dirname (str, optional): Subdirectory for the dataset. Defaults to "churchland_shenoy_neural_2012".

    """

    def __init__(
        self,
        root: Optional[str] = None,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: Optional[Literal["cursor_velocity"]] = "cursor_velocity",
        dirname: str = "churchland_shenoy_neural_2012",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

        self.spiking_dataset_mixin_uniquify_unit_ids = True
        self.split_type = split_type

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        ans = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            ans[rid] = getattr(data, domain_key)

            if self.split_type == "cursor_velocity":
                ans[rid] = ans[rid] & data.cursor.domain & data.spikes.domain

        return ans
