"""Base pipeline classes for OpenNeuro datasets."""

__all__ = [
    "OpenNeuroPipeline",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}

import logging
import sys
import warnings
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

import h5py
import pandas as pd

try:
    from mne_bids import read_raw_bids

    MNE_BIDS_AVAILABLE = True
except ImportError:
    read_raw_bids = None
    MNE_BIDS_AVAILABLE = False

from torch_brain.data import (
    BrainsetDescription,
    Data,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
    serialize_fn_map,
)
from torch_brain.utils.bids import (
    build_bids_path,
    check_eeg_recording_files_exist,
    check_ieeg_recording_files_exist,
    fetch_eeg_recordings,
    fetch_ieeg_recordings,
    get_subject_info,
)
from torch_brain.utils.mne import (
    extract_channels,
    extract_measurement_date,
    extract_signal,
)
from torch_brain.utils.openneuro import (
    construct_s3_url_from_path,
    download_dataset_description,
    download_participants_tsv,
    download_recording,
    fetch_all_filenames,
    fetch_latest_snapshot_tag,
    fetch_participants_tsv,
    fetch_species,
)

from .pipeline import BrainsetPipeline

base_openneuro_parser = ArgumentParser()
base_openneuro_parser.add_argument("--redownload", action="store_true")
base_openneuro_parser.add_argument("--reprocess", action="store_true")
base_openneuro_parser.add_argument(
    "--on-version-mismatch",
    choices=["abort", "continue", "prompt"],
    default="prompt",
    help=(
        "Behavior when origin_version differs from latest OpenNeuro version: "
        "'abort' raises an error, 'continue' proceeds with warning, "
        "'prompt' asks for confirmation in interactive sessions."
    ),
)

OpenNeuroDataModality = Literal["eeg", "ieeg"]


def _require_mne_bids(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


class OpenNeuroPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro dataset pipelines.

    This class provides foundational tools and conventions for preprocessing and handling
    `OpenNeuro <https://openneuro.org/>`_ datasets within the Brainsets framework.
    It is designed to be subclassed for specific datasets and supports both EEG and iEEG modalities.

    **Attributes (to be defined by subclasses):**
        - :attr:`dataset_id`: Identifier for the OpenNeuro dataset (e.g., "ds005555").
        - :attr:`brainset_id`: Unique local identifier for the brainset.
        - :attr:`origin_version`: Version string corresponding to the raw source dataset.
        - :attr:`derived_version`: Version or tag indicating the processing version of the derived data.
        - :attr:`description`: Optional textual description of the dataset.
        - :attr:`modality`: Data modality for this pipeline. Must be overridden by subclasses.

    **Customization points:**
        This class supports and encourages dataset-specific customizations via:
            - :attr:`CHANNEL_NAME_REMAPPING`: Map original to standardized channel names.
            - :attr:`TYPE_CHANNELS_REMAPPING`: Map channel types to specific channel names.
            - :attr:`IGNORE_CHANNELS`: List channels to exclude from processing.

        These can be set as class attributes or managed dynamically by overriding the following methods:
            - :meth:`get_channel_name_remapping()`
            - :meth:`get_type_channels_remapping()`

        The :meth:`process_common` method implements the standard steps and routines shared
        by all OpenNeuro datasets. This provides a consistent entry point for all dataset
        processing. Subclasses may extend or override the :meth:`process` method to
        implement dataset-specific processing logic.

    **Documentation can be found in the official brainsets docs:**
    See [Creating an OpenNeuro Pipeline](https://brainsets.readthedocs.io/en/latest/concepts/openneuro_pipeline.html) for the complete guide on building OpenNeuro pipelines.

    """

    parser = base_openneuro_parser
    """Argument parser for common OpenNeuro pipeline flags."""

    modality: OpenNeuroDataModality
    """Data modality for this pipeline. Must be overridden by subclasses."""

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555", "ds006914")."""

    brainset_id: str
    """Unique identifier for the brainset."""

    origin_version: str
    """Version of the original data. Must be specified by the author of each pipeline."""

    derived_version: str
    """Version of the processed data. Must be specified by the author of each pipeline."""

    description: str | None = None
    """Optional description of the dataset."""

    CHANNEL_NAME_REMAPPING: dict[str, str] | None = None
    """Optional dict mapping original channel name to new standardized name.

    For more complex configurations (e.g., per-recording mappings), override
    get_channel_name_remapping() instead.
    """

    TYPE_CHANNELS_REMAPPING: dict[str, list[str]] | None = None
    """Optional dict mapping channel types to lists of channel names.

    For more complex configurations (e.g., per-recording mappings), override
    get_type_channels_remapping() instead.
    """

    IGNORE_CHANNELS: list[str] | None = None
    """Optional list of channel names to ignore.

    Channel names should be specified as they appear in the original namespace of
    the raw object (i.e., prior to any remapping or type changes).
    """

    @staticmethod
    def validate_dataset_id(dataset_id: str) -> None:
        """Validate OpenNeuro dataset identifier format.

        OpenNeuro dataset IDs follow the format 'ds' followed by exactly 6 digits,
        where the numeric portion ranges from 000001 to 009999.

        Args:
            dataset_id: The dataset identifier in strict format:
                - Must be lowercase 'ds' followed by exactly 6 digits.
                - Numeric portion must be between 000001 and 009999.

        Raises:
            ValueError: If the dataset ID format is invalid, does not match strict format,
                or the numeric part is outside the valid range.
        """
        if (
            not isinstance(dataset_id, str)
            or len(dataset_id) != 8
            or not dataset_id.startswith("ds")
            or not dataset_id[2:].isdigit()
        ):
            raise ValueError(
                f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by exactly 6 digits."
            )

        numeric_part = int(dataset_id[2:])
        if numeric_part < 1 or numeric_part > 9999:
            raise ValueError(
                f"Dataset ID '{dataset_id}' has invalid numeric portion. Must be between 000001 and 009999."
            )

    @classmethod
    def _validate_dataset_version(
        cls,
        latest_snapshot_tag: str,
        on_mismatch: Literal["abort", "continue", "prompt"] = "prompt",
    ) -> None:
        """Validate origin version against the latest OpenNeuro snapshot tag.

        Args:
            latest_snapshot_tag: The latest snapshot tag available on OpenNeuro for this dataset.
            on_mismatch: Policy when ``origin_version`` differs from latest
                (``"abort"``, ``"continue"``, or ``"prompt"``). If a mismatch is detected, the
                ``on_mismatch`` parameter determines the behavior (default: ``"prompt"``):
                    - ``"abort"``: Raises an error and exits the pipeline.
                    - ``"continue"``: Logs a warning and proceeds with the latest version.
                    - ``"prompt"``: Prompts the user for confirmation and proceeds if confirmed.

        Raises:
            SystemExit: If mismatch policy aborts execution or user declines prompt.
        """

        def user_confirms(
            prompt: str,
        ) -> bool:
            """Return True if the user confirms continuation, False otherwise."""
            answer = input(prompt).strip().lower()
            return answer in {"y", "yes"}

        if latest_snapshot_tag != cls.origin_version:
            if on_mismatch == "continue":
                logging.warning(
                    f"⚠️ Dataset version '{cls.origin_version}' was used to create the brainset pipeline for dataset '{cls.dataset_id}', "
                    f"but the latest available version on OpenNeuro is '{latest_snapshot_tag}'. "
                    "Downloading data or running the pipeline now will use the latest version, "
                    "which may differ from the original version used, potentially causing errors or inconsistencies. "
                    "Check the CHANGES file of the dataset for details about the differences between versions."
                )
            elif on_mismatch == "abort":
                raise SystemExit(
                    "🛑 Aborting pipeline due to dataset version mismatch."
                )
            elif on_mismatch == "prompt":
                prompt_message = (
                    f"⚠️ Dataset '{cls.dataset_id}' pipeline version is '{cls.origin_version}', "
                    f"but latest on OpenNeuro is '{latest_snapshot_tag}'. "
                    "👉 Continue with latest version? [y/N]: "
                )
                if not user_confirms(prompt_message):
                    raise SystemExit(
                        "🛑 Aborted by user due to dataset version mismatch."
                    )

    @staticmethod
    def _validate_on_mismatch_policy(on_version_mismatch: str) -> None:
        """Validate that on_version_mismatch policy is compatible with execution mode.

        In non-interactive sessions, the 'prompt' policy is invalid because it requires
        user input. This validation runs early to provide a clear error message.

        Args:
            on_version_mismatch: Policy value ('abort', 'continue', or 'prompt').

        Raises:
            ValueError: If on_version_mismatch='prompt' in non-interactive mode.
        """
        if on_version_mismatch == "prompt" and not sys.stdin.isatty():
            raise ValueError(
                "Cannot use --on-version-mismatch='prompt' in non-interactive mode. "
                "The program is running without a TTY and cannot prompt for user input. "
                "Set --on-version-mismatch to either 'continue' (warn and proceed) or 'abort' (fail on mismatch)."
            )

    @staticmethod
    def _normalize_species(species: str | None) -> str | None:
        """Normalize species names to ``"HOMO_SAPIENS"`` or None.

        Args:
            species: The input species name (string or None).

        Returns:
            ``"HOMO_SAPIENS"`` for recognized human aliases, otherwise None.
        """
        if not isinstance(species, str):
            return None

        normalized_species = species.strip().lower()
        homo_sapiens_aliases = {
            "homo",
            "homo sapiens",
            "human",
            "humans",
            "h. sapiens",
        }
        if normalized_species in homo_sapiens_aliases:
            return "HOMO_SAPIENS"
        return None

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Namespace | None) -> pd.DataFrame:
        """Generate a manifest DataFrame by discovering recordings from OpenNeuro.

        This implementation queries OpenNeuro S3 and parses BIDS-compliant
        filenames to discover recordings for the pipeline modality.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line

        Returns:
            DataFrame with columns:
                - subject_id: Subject identifier (e.g., 'sub-01')
                - recording_id: Recording identifier (index)
                - s3_url: S3 URL for downloading
        """
        # Determine the 'on_version_mismatch' policy from args if available, else default to 'prompt'
        on_version_mismatch = args.on_version_mismatch
        cls._validate_on_mismatch_policy(on_version_mismatch)

        # Validate that dataset ID has the correct format
        cls.validate_dataset_id(cls.dataset_id)

        # Fetch the latest snapshot tag available on OpenNeuro for the dataset
        latest_snapshot_tag = fetch_latest_snapshot_tag(cls.dataset_id)
        cls._validate_dataset_version(
            latest_snapshot_tag, on_mismatch=on_version_mismatch
        )

        # Fetch the species of the participants in the dataset
        species = fetch_species(cls.dataset_id)
        species = cls._normalize_species(species)

        # Fetch the participants.tsv file from the dataset
        participants_data = fetch_participants_tsv(cls.dataset_id)

        # Fetch all filenames in the dataset from OpenNeuro S3
        all_files = fetch_all_filenames(cls.dataset_id)

        # Depending on modality, extract a list of recordings
        if cls.modality == "eeg":
            recordings = fetch_eeg_recordings(all_files)
        elif cls.modality == "ieeg":
            recordings = fetch_ieeg_recordings(all_files)
        else:
            raise ValueError(f"Unknown modality: {cls.modality}")

        manifest_list = []
        for rec in recordings:
            subject_id = rec["subject_id"]
            recording_id = rec["recording_id"]
            fpath = rec["fpath"]

            # Construct the S3 URL for the recording
            s3_url = construct_s3_url_from_path(
                cls.dataset_id,
                fpath,
                recording_id,
            )

            # Fetch the subject information from the participants.tsv file
            subject_info = get_subject_info(subject_id, participants_data)

            manifest_list.append(
                {
                    "subject_id": subject_id,
                    "recording_id": recording_id,
                    "s3_url": s3_url,
                    "latest_snapshot_tag": latest_snapshot_tag,
                    "age": subject_info.get("age"),
                    "sex": subject_info.get("sex"),
                    "species": species,
                }
            )

        if not manifest_list:
            raise ValueError(
                f"No {cls.modality.upper()} recordings found in dataset {cls.dataset_id}"
            )

        # Create a DataFrame for the manifest and set 'recording_id' as its index
        manifest = pd.DataFrame(manifest_list)
        return manifest.set_index("recording_id")

    def download(self, manifest_item) -> pd.Series:
        """Download data for a single recording from OpenNeuro S3.

        Args:
            manifest_item: A single row of the manifest

        Returns:
            Series containing ``subject_id``, ``recording_id``, ``s3_url``, ``latest_snapshot_tag``, ``age``, ``sex``, and ``species``.
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        subject_id = manifest_item.subject_id
        recording_id = manifest_item.Index
        s3_url = manifest_item.s3_url
        root_dir = self.raw_dir
        redownload = getattr(self.args, "redownload", False)

        # dataset_description.json is required for mne-bids to recognize a valid BIDS dataset
        download_dataset_description(
            self.dataset_id,
            root_dir,
            redownload=redownload,
        )

        # participants.tsv is optional; persist it so processing can read subject
        # metadata from disk without re-fetching from S3.
        try:
            download_participants_tsv(
                self.dataset_id,
                root_dir,
                redownload=redownload,
            )
        except RuntimeError as e:
            warnings.warn(
                f"Could not download participants.tsv for {self.dataset_id}: {e}. "
                "Skipping subject information.",
                stacklevel=2,
            )

        if not redownload:
            if self.modality == "eeg":
                if check_eeg_recording_files_exist(root_dir, recording_id):
                    self.update_status("Already Downloaded")
                    return manifest_item
            elif self.modality == "ieeg":
                if check_ieeg_recording_files_exist(root_dir, recording_id):
                    self.update_status("Already Downloaded")
                    return manifest_item

        try:
            download_recording(s3_url, root_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {self.dataset_id}: {str(e)}"
            ) from e

        return manifest_item

    def process_common(self, download_output: pd.Series) -> tuple[Data, Path] | None:
        """Process data files and create a Data object.

        This method handles common OpenNeuro processing tasks:
        1. Loads BIDS-structured data files using MNE-BIDS
        2. Extracts metadata (subject, session, device, brainset descriptions)
        3. Extracts signal and channel information
        5. Creates a Data object

        Args:
            download_output: Series returned by download()

        Returns:
            Tuple of ``(data, store_path)``, or ``None`` if processing is skipped.
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output.Index
        subject_id = download_output.subject_id
        species = download_output.species
        age = download_output.age
        sex = download_output.sex

        store_path = self.processed_dir / f"{recording_id}.h5"
        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        _require_mne_bids("_process_common")
        self.update_status(f"Loading {self.modality.upper()} file")
        bids_path = build_bids_path(self.raw_dir, recording_id, self.modality)
        raw = read_raw_bids(
            bids_path,
            on_ch_mismatch="reorder",
            verbose="CRITICAL",
        )

        self.update_status("Extracting Metadata")
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        dataset_description = (
            self.description
            if self.description
            else f"OpenNeuro dataset {self.dataset_id}"
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version=download_output.latest_snapshot_tag,
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        subject_description = SubjectDescription(
            id=subject_id,
            species=species,
            age=age,
            sex=sex,
        )

        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=recording_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(
            raw,
            ignore_channels=self.IGNORE_CHANNELS,
        )

        self.update_status("Building Channels")
        channels = extract_channels(
            raw,
            channel_names_mapping=self.get_channel_name_remapping(recording_id),
            type_channels_mapping=self.get_type_channels_remapping(recording_id),
            ignore_channels=self.IGNORE_CHANNELS,
        )

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "channels": channels,
            "domain": signal.domain,
        }
        data_kwargs[self.modality] = signal

        data = Data(**data_kwargs)

        return data, store_path

    def process(self, download_output: pd.Series) -> None:
        """Process and save the dataset.

        Default implementation calls :meth:`_process_common` and persists the
        result. Subclasses can override to add dataset-specific processing.

        Args:
            download_output: Series returned by download()
        """
        result = self.process_common(download_output)

        if result is None:
            return

        data, store_path = result

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    def get_channel_name_remapping(
        self,
        recording_id: str | None = None,
    ) -> dict[str, str] | None:
        """Return channel name remapping for a given recording.

        Override this method to provide per-recording channel name remappings.
        The default implementation returns the class-level CHANNEL_NAME_REMAPPING attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Mapping from original channel names to standardized names, or
            ``None``.
        """
        return self.CHANNEL_NAME_REMAPPING

    def get_type_channels_remapping(
        self,
        recording_id: str | None = None,
    ) -> dict[str, list[str]] | None:
        """Return channel type remapping for a given recording.

        Override this method to provide per-recording channel type remappings.
        The default implementation returns the class-level TYPE_CHANNELS_REMAPPING attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Mapping from channel type to channel name list, or ``None``.
        """
        return self.TYPE_CHANNELS_REMAPPING
