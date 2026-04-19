from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin

from ._utils import get_processed_dir


class FlintSlutzkyAccurate2012(SpikingDatasetMixin, Dataset):
    """
    Motor cortex (M1) spiking activity and reaching kinematics from 1 monkey
    performing center-out reaching tasks.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare flint_slutzky_accurate_2012``.

    **Tasks:** Center-Out

    **Brain Regions:** M1

    **Dataset Statistics**

    - **Subjects:** 1
    - **Total Sessions:** 5
    - **Total Units:** 957
    - **Events:** ~7.9M spikes, ~319k behavioral timestamps

    **Links**

    - Paper: `Flint et al. (2012) – Journal of Neural Engineering <https://doi.org/10.1088/1741-2560/9/4/046006>`_
    - Dataset: `CRCNS Flint 2012 dataset <https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012>`_

    **Reference**

    Flint, R. D., Lindberg, E. W., Jordan, L. R., Miller, L. E., & Slutzky, M. W. (2012).
    *Accurate decoding of reaching movements from field potentials in the absence of spikes.*
    `Journal of Neural Engineering <https://doi.org/10.1088/1741-2560/9/4/046006>`_, 9(4), 046006.

    Args:
        root (str, optional): Root directory for the dataset. Defaults to ``processed_dir`` from brainsets config.
        recording_ids (list[str], optional): List of recording IDs to load.
        transform (Callable, optional): Data transformation to apply.
        split_type (str, optional): Which split type to use. Defaults to "hand_velocity".
        dirname (str, optional): Subdirectory for the dataset. Defaults to "flint_slutzky_accurate_2012".

    """

    def __init__(
        self,
        root: Optional[str] = None,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: Optional[Literal["hand_velocity"]] = "hand_velocity",
        dirname: str = "flint_slutzky_accurate_2012",
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

            if self.split_type == "hand_velocity":
                ans[rid] = ans[rid] & data.hand.domain & data.spikes.domain

        return ans
