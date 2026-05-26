from typing import Callable, Optional

from brainsets.datasets.OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType


class KlinzingSleepDS005555(OpenNeuroDataset):
    """
    Klinzing Sleep iEEG Dataset (OpenNeuro DS005555).

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare klinzing_sleep_ds005555

    Each dataset instance uses a split strategy (`split_type`) and can optionally be
    restricted to specific recordings via recording_ids.

    Args:
        root (str): Root directory containing processed Klinzing Sleep artifacts.
        split_type (OpenNeuroSplitType): Dataset split strategy, e.g. train/valid/test as designated by the workflow.
        recording_ids (list[str], optional): List of explicit recording IDs to load. If omitted, the dataset uses split-based recording selection.
        transform (Callable, optional): Optional transform to apply to each sample.
        **kwargs: Additional keyword arguments forwarded to OpenNeuroDataset.

    **References**

    Klinzing, J. G., et al. (Year). "Sleep iEEG Dataset." Repository: https://openneuro.org/datasets/ds005555
    """

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        dataset_dir = "klinzing_sleep_ds005555"
        super().__init__(
            root=root,
            dataset_dir=dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            split_type=split_type,
            **kwargs,
        )
