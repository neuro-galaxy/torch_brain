from collections.abc import Callable

from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType


class ShiraziHBNR1DS005505(OpenNeuroDataset):
    """
    Shirazi HBN Resting State 1 (HBN-R1) iEEG Dataset (OpenNeuro DS005505).

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare shirazi_hbn_r1_ds005505

    Each dataset instance uses a split strategy (`split_type`) and can optionally be
    restricted to specific recordings via recording_ids.

    Args:
        root: Root directory containing processed HBN-R1 artifacts.
        split_type: The split type describing train/valid/test regime.
        recording_ids: List of explicit recording IDs to load. If omitted, the dataset uses split-based recording selection.
        transform: Optional transform to apply to samples.
        **kwargs: Additional keyword arguments forwarded to the base OpenNeuroDataset.

    **References**

    Shirazi, S. Y., et al. (Year). "Dataset title." Repository: https://openneuro.org/datasets/ds005505
    """

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        dataset_dir = "shirazi_hbnr1_ds005505"
        super().__init__(
            root=root,
            dataset_dir=dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            split_type=split_type,
            **kwargs,
        )
