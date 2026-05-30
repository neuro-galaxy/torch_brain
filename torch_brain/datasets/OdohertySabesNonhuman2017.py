from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ._utils import get_processed_dir
from .dataset import Dataset
from .mixins import SpikingDatasetMixin


class OdohertySabesNonhuman2017(SpikingDatasetMixin, Dataset):
    """
    Motor cortex (M1 and S1) spiking activity and reaching kinematics from 2 monkeys
    performing random target reaching tasks with right hand.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare odoherty_sabes_nonhuman_2017

    **Tasks:** Random Target

    **Brain Regions:** M1, S1

    **Dataset Statistics**

    - **Subjects:** 2
    - **Total Sessions:** 47
    - **Total Units:** 16,566
    - **Events:** ~105.2M spikes, ~12.4M behavioral timestamps

    **Links**

    - Paper: `O'Doherty and Sabes (2018) - Journal of Neural Engineering <https://pubmed.ncbi.nlm.nih.gov/29192609/>`_
    - Dataset: `Zenodo Record 3854034 <https://zenodo.org/records/3854034>`_

    **Reference**

    O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., & Sabes, P. N. (2020).
    *Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology.*
    `Zenodo Dataset <https://doi.org/10.5281/zenodo.788569>`_.

    Args:
        root: Root directory for the dataset. Defaults to ``processed_dir`` from brainsets config.
        recording_ids: List of recording IDs to load.
        transform: Data transformation to apply.
        split_type: Which split type to use. Defaults to "cursor_velocity".
        dirname: Subdirectory for the dataset. Defaults to "odoherty_sabes_nonhuman_2017".

    """

    def __init__(
        self,
        root: str | None = None,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        split_type: Literal["cursor_velocity"] | None = "cursor_velocity",
        dirname: str = "odoherty_sabes_nonhuman_2017",
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
        split: Literal["train", "valid", "test"] | None = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        ans = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            ans[rid] = getattr(data, domain_key)

            if self.split_type == "cursor_velocity":
                ans[rid] = ans[rid] & data.cursor.domain & data.spikes.domain

        return ans
