import re
from collections.abc import Callable
from pathlib import Path

from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType

RELEASES = tuple(range(1, 12))

# Processed files are named "HBN_R{release_id}_{recording_id}.h5" by the pipeline.
_RELEASE_PREFIX_RE = re.compile(r"^HBN_R(\d+)_")


class ShiraziHBN(OpenNeuroDataset):
    """
    Healthy Brain Network (HBN) EEG Dataset combining all 11 data releases.

    .. admonition:: Preprocessing

        To download and prepare all 11 releases of this dataset, run

        .. code:: shell

            brainsets prepare shirazi_hbn

        To download and prepare a specific release (1-11), run

        .. code:: shell

            brainsets prepare shirazi_hbn --release <release_id>

    All releases are processed as a single brainset: every processed recording is
    stored in one directory (``shirazi_hbn``) and named with a release prefix,
    ``HBN_R<release_id>_<recording_id>.h5``.

    Args:
        root: Root directory containing processed HBN artifacts.
        split_type: The split type describing train/valid/test regime.
        releases: List of release IDs (1-11) to include. If omitted, all releases
            are loaded.
        recording_ids: List of recording IDs (file stems) to load. If omitted, all
            recordings of the selected releases are loaded.
        transform: Optional transform to apply to samples.
        dirname: Name of the processed dataset directory under ``root``.
            Defaults to ``"shirazi_hbn"``.
        **kwargs: Additional keyword arguments forwarded to
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
        dirname: str = "shirazi_hbn",
        **kwargs,
    ):
        if releases is not None:
            invalid = [release for release in releases if release not in RELEASES]
            if invalid:
                raise ValueError(
                    f"Invalid release IDs: {invalid}. Releases must be in {RELEASES}."
                )
        self.releases = list(releases) if releases is not None else list(RELEASES)

        # When a subset of releases is requested (and no explicit recordings are
        # given), select the matching files by their "HBN_R{release_id}_" prefix.
        # Otherwise, defer to OpenNeuroDataset (loads all *.h5 / the given ids).
        if recording_ids is None and releases is not None:
            allowed = set(self.releases)
            dataset_dir = Path(root) / dirname
            recording_ids = sorted(
                path.stem
                for path in dataset_dir.glob("*.h5")
                if (match := _RELEASE_PREFIX_RE.match(path.stem))
                and int(match.group(1)) in allowed
            )
            if not recording_ids:
                raise ValueError(
                    f"No recordings found for releases {self.releases} in {dataset_dir}."
                )

        super().__init__(
            root=root,
            dataset_dir=dirname,
            split_type=split_type,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )
