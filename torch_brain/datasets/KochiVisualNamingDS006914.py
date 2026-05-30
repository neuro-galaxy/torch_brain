from collections.abc import Callable

from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType


class KochiVisualNamingDS006914(OpenNeuroDataset):
    """
    Kochi Visual Naming iEEG Dataset (OpenNeuro DS006914).

    .. admonition:: Preprocessing

        To download and prepare this dataset, run

        .. code:: shell

            brainsets prepare kochi_visualnaming_ds006914

    Each dataset instance uses a split strategy (`split_type`) and can optionally be
    restricted to specific recordings via recording_ids.

    Args:
        root: Root directory containing processed Visual Naming artifacts.
        split_type: The split type describing train/valid/test regime.
        recording_ids: List of explicit recording IDs to load. If omitted, the dataset uses split-based recording selection.
        transform: Optional transform to apply to samples.
        **kwargs: Additional keyword arguments forwarded to the base OpenNeuroDataset.

    **References**

    Kochi, J., et al. (Year). "Visual Naming iEEG Dataset." Repository: https://openneuro.org/datasets/ds006914
    """

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        dataset_dir = "kochi_visualnaming_ds006914"
        super().__init__(
            root=root,
            dataset_dir=dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            split_type=split_type,
            **kwargs,
        )
