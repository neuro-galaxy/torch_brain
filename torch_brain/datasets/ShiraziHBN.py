from collections.abc import Callable

from .nested import NestedDataset
from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType

RELEASES = tuple(range(1, 12))


class ShiraziHBN(NestedDataset):
    """
    Healthy Brain Network (HBN) EEG Dataset combining all 11 data releases.

    .. admonition:: Preprocessing

        To download and prepare all 11 releases of this dataset, run

        .. code:: shell

            brainsets prepare shirazi_hbn

        To download and prepare a specific release (1-11), run

        .. code:: shell

            brainsets prepare shirazi_hbn --release <release_id>

    Processed recordings are stored per release in ``shirazi_hbn_r1`` through
    ``shirazi_hbn_r11`` under the root directory. This dataset composes those
    release directories into a single interface.

    Each dataset instance uses a split strategy (`split_type`) and can optionally be
    restricted to specific recordings via recording_ids.
    Exposed ``recording_ids`` take the form ``"r<release_id>/<recording_id>"``
    (for example, ``"r1/sub-NDARAC904DMU_task-DespicableMe"``).

    Args:
        root: Root directory containing processed HBN artifacts.
        split_type: The split type describing train/valid/test regime.
        releases: List of release IDs (1-11) to include. If omitted, all releases
            are loaded.
        recording_ids: List of explicit recording IDs to load. IDs may be either
            bare recording names (matched across all selected releases) or
            release-prefixed names of the form ``"r<release_id>/<recording_id>"``.
            If omitted, the dataset uses split-based recording selection.
        transform: Optional transform to apply to samples.
        **kwargs: Additional keyword arguments forwarded to each release
            :class:`OpenNeuroDataset`.

    **References**

    Shirazi, S. Y., et al. (2025). "Healthy Brain Network (HBN) EEG - Release 1."

    Shirazi, S. Y., et al. (2024). "HBN-EEG: The FAIR implementation of the Healthy
    Brain Network (HBN) electroencephalography dataset". bioRxiv.

    Alexander, L. M., et al. (2017). "An open resource for transdiagnostic research
    in pediatric mental health and learning disorders." Scientific data.

    Repository: https://openneuro.org/datasets/ds005505
    """

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        releases: list[int] | None = None,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        if releases is None:
            releases = list(RELEASES)
        else:
            invalid = [release for release in releases if release not in RELEASES]
            if invalid:
                raise ValueError(
                    f"Invalid release IDs: {invalid}. Releases must be in {RELEASES}."
                )

        prefixed_recording_ids: dict[int, list[str]] = {
            release_id: [] for release_id in releases
        }
        shared_recording_ids: list[str] = []
        if recording_ids is not None:
            for recording_id in recording_ids:
                if "/" in recording_id:
                    release_name, bare_recording_id = recording_id.split("/", 1)
                    if not release_name.startswith("r"):
                        raise ValueError(
                            f"Invalid recording_id '{recording_id}'. Expected format "
                            "'r<release_id>/<recording_id>'."
                        )
                    release_id = int(release_name[1:])
                    if release_id not in prefixed_recording_ids:
                        raise ValueError(
                            f"Recording '{recording_id}' references release "
                            f"{release_id}, which is not included in releases={releases}."
                        )
                    prefixed_recording_ids[release_id].append(bare_recording_id)
                else:
                    shared_recording_ids.append(recording_id)

        datasets = {}
        for release_id in releases:
            release_name = f"r{release_id}"
            release_recording_ids = (
                prefixed_recording_ids[release_id] + shared_recording_ids
            )
            datasets[release_name] = OpenNeuroDataset(
                root=root,
                dataset_dir=f"shirazi_hbn_{release_name}",
                split_type=split_type,
                recording_ids=release_recording_ids or None,
                **kwargs,
            )

        self.split_type = split_type
        self.releases = releases
        super().__init__(datasets=datasets, transform=transform)
