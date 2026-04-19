"""Brain Imaging Data Structure (BIDS) utilities.

This module provides utility functions to parse BIDS-compliant filenames, discover BIDS recordings in a dataset, and check for the existence of BIDS-conformant data files.

For more information about BIDS, see the BIDS specification: https://bids-specification.readthedocs.io/en/stable/
"""

from collections import defaultdict
from typing import Optional, Literal
from pathlib import Path
import warnings
import re
import json
import pandas as pd

try:
    from mne_bids import (
        get_bids_path_from_fname,
        get_entities_from_fname,
        get_entity_vals,
        BIDSPath,
    )

    MNE_BIDS_AVAILABLE = True
except ImportError:
    get_bids_path_from_fname = None
    get_entities_from_fname = None
    get_entity_vals = None
    BIDSPath = None
    MNE_BIDS_AVAILABLE = False

# BIDS EEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording. edf+ files permitted.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording. bdf+ files permitted.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
EEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf"}

# BIDS iEEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording.
# - NWB (.nwb): Neurodata Without Borders format for standardized neurophysiology storage.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html
IEEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf", ".nwb"}

# BIDS entity short names (BIDS v1.10.1):
# - subject: 'sub'
# - session: 'ses'
# - task: 'task'
# - acquisition: 'acq'
# - run: 'run'
# - description: 'desc'
# Reference: https://bids-specification.readthedocs.io/en/stable/appendices/entities.html
# Note: The short names are used to group recordings by entity.
BIDS_ENTITY_SHORT_NAMES = {
    "subject": "sub",
    "sub": "sub",
    "session": "ses",
    "ses": "ses",
    "task": "task",
    "acquisition": "acq",
    "acq": "acq",
    "run": "run",
    "description": "desc",
    "desc": "desc",
}

_SUPPORTED_MODALITIES = ["eeg", "ieeg"]
_ModalityLiteral = Literal[_SUPPORTED_MODALITIES]


def _check_mne_bids_available(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


def fetch_eeg_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
) -> list[dict]:
    """Discover all EEG recordings inside a BIDS dataset or list of files.

    Args:
        source: Either the BIDS root directory (as a str, Path, or BIDSPath), or a list of files (each item being a str, Path, or BIDSPath).

    Returns:
        List of dicts with key/value pairs for BIDS entities extracted from the fetched recording files.
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acquisition_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Relative path to EEG file

        For more information on BIDS entities, see:
        https://bids-specification.readthedocs.io/en/stable/appendices/entities.html


    Notes:
        If `source` is a BIDSPath pointing to a subfolder inside the BIDS root directory,
        this function will return all EEG recording files within that specific subfolder.
        This does not work if the subfolder is a string or Path object.
    """
    _check_mne_bids_available("fetch_eeg_recordings")
    return _fetch_recordings(source, EEG_EXTENSIONS, "eeg")


def fetch_ieeg_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
) -> list[dict]:
    """Discover all iEEG recordings inside a BIDS dataset or list of files.

    Args:
        source: Either the BIDS root directory (as a str, Path, or BIDSPath), or a list of files (each item being a str, Path, or BIDSPath).

    Returns:
        List of dicts with key/value pairs for BIDS entities extracted from the fetched recording files.
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acquisition_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Relative path to iEEG file

        For more information on BIDS entities, see:
        https://bids-specification.readthedocs.io/en/stable/appendices/entities.html


    Notes:
        If `source` is a BIDSPath pointing to a subfolder inside the BIDS root directory,
        this function will return all iEEG recording files within that specific subfolder.
        This does not work if the subfolder is a string or Path object.
    """
    _check_mne_bids_available("fetch_ieeg_recordings")
    return _fetch_recordings(source, IEEG_EXTENSIONS, "ieeg")


def group_recordings_by_entity(
    recordings: list[dict],
    fixed_entities: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Group BIDS-compliant recordings by specified fixed entities.

    BIDS entities (e.g., subject, session, task) are standardized labels in BIDS filenames.
    For more information on BIDS entities, see:
    https://bids-specification.readthedocs.io/en/stable/appendices/entities.html

    Group keys are constructed using only the entities listed in
    `fixed_entities`; all other entities are implicitly allowed to vary within
    a group.

    By default (`fixed_entities=None`), groups are created by all entities except 'run'.

    Entities can be provided in long form (e.g., `subject`, `session`) or short
    BIDS form (e.g., `sub`, `ses`).

    Args:
        recordings: List of recording dictionaries that include a `recording_id`
            key.
        fixed_entities: Entities that must remain fixed within each group.
            If None, all entities except `run` are kept in the grouping key.

    Returns:
        Dictionary mapping a grouping key to the list of recordings in that group.

    Raises:
        ValueError: If an entity name is unsupported.
    """
    _check_mne_bids_available("group_recordings_by_entity")

    def _normalize_entity_list(entities: list[str], arg_name: str) -> list[str]:
        normalized = []
        for entity in entities:
            short_name = BIDS_ENTITY_SHORT_NAMES.get(entity.lower())
            if short_name is None:
                raise ValueError(
                    f"Unsupported BIDS entity '{entity}' in '{arg_name}'. "
                    f"Expected one of: {sorted(BIDS_ENTITY_SHORT_NAMES)}"
                )
            normalized.append(short_name)
        return normalized

    if fixed_entities is None:
        warnings.warn(
            "No fixed_entities were specified for grouping recordings. "
            "By default, recordings will be grouped by all BIDS entities present in the filename, except for the 'run' entity "
            "(so only the run may vary within each group). "
            "To control grouping more precisely, specify 'fixed_entities' as a list of BIDS entities (e.g., ['subject', 'session', 'task'])."
        )

    normalized_fixed = (
        set(_normalize_entity_list(fixed_entities, "fixed_entities"))
        if fixed_entities is not None
        else None
    )

    entity_groups: dict[str, list[dict]] = defaultdict(list)
    token_pattern = re.compile(r"^(?P<entity>[a-z]+)-[^_]+$")

    for recording in recordings:
        recording_id = recording["recording_id"]
        components = recording_id.split("_")
        key_components = []

        for component in components:
            match = token_pattern.match(component)
            if match is None:
                continue

            entity_short_name = match.group("entity")

            if normalized_fixed is None:
                if entity_short_name == "run":
                    continue
                key_components.append(component)
                continue

            if entity_short_name in normalized_fixed:
                key_components.append(component)

        entity_key = "_".join(key_components)
        entity_groups[entity_key].append(recording)

    return dict(entity_groups)


def check_eeg_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
) -> bool:
    """Check if EEG data files corresponding to a BIDS recording_id exist in the BIDS root directory.

    Note: The BIDS root directory is the top-level folder of a BIDS dataset.
    All data and metadata within the dataset are organized relative to this root directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')

    Returns:
        True if at least one EEG data file is found, False otherwise.
    """
    _check_mne_bids_available("check_eeg_recording_files_exist")
    return _check_recording_files_exist(bids_root, recording_id, EEG_EXTENSIONS)


def check_ieeg_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
) -> bool:
    """Check if iEEG data files corresponding to a BIDS recording_id exist in the BIDS root directory.

    Note: The BIDS root directory is the top-level folder of a BIDS dataset.
    All data and metadata within the dataset are organized relative to this root directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')

    Returns:
        True if at least one iEEG data file is found, False otherwise.
    """
    _check_mne_bids_available("check_ieeg_recording_files_exist")
    return _check_recording_files_exist(bids_root, recording_id, IEEG_EXTENSIONS)


def build_bids_path(
    bids_root: str | Path,
    recording_id: str,
    modality: _ModalityLiteral,
) -> BIDSPath:
    """Build a mne_bids.BIDSPath for a given recording_id, modality, and BIDS root directory.

    Note: The BIDS root directory is the top-level folder of a BIDS dataset.
    All data and metadata within the dataset are organized relative to this root directory.

    BIDSPath is a helper class from mne-bids for representing BIDS file paths and entities.
    For more information on BIDSPath, see:
    https://mne.tools/mne-bids/stable/generated/mne_bids.BIDSPath.html

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
        modality: Modality (supported values: 'eeg', 'ieeg')

    Returns:
        BIDSPath configured for reading via mne_bids.read_raw_bids.

    Raises:
        ValueError: If any unsupported BIDS entities are present in recording_id.
    """
    _check_mne_bids_available("build_bids_path")
    _validate_modality(modality)

    if not _is_bids_root(bids_root):
        raise ValueError(
            f"bids_root ('{bids_root}') must point to a valid BIDS root directory."
        )

    try:
        entities = get_entities_from_fname(recording_id, on_error="raise")
    except KeyError as err:
        raise ValueError(
            f"Unsupported BIDS entity '{err.args[0]}' in recording_id: {recording_id}. "
            f"Expected one of: {sorted(set(BIDS_ENTITY_SHORT_NAMES.values()))}"
        ) from err

    return BIDSPath(
        root=bids_root,
        subject=entities.get("subject"),
        session=entities.get("session"),
        task=entities.get("task"),
        acquisition=entities.get("acquisition"),
        run=entities.get("run"),
        description=entities.get("description"),
        datatype=modality,
        suffix=modality,
    )


def load_json_sidecar(bids_path: BIDSPath) -> dict:
    """Load the JSON sidecar file for a given BIDS file.

    The JSON sidecar file contains metadata and additional annotations about a BIDS recording,
    such as acquisition parameters, device settings, channel information, and other contextual
    information that supplements the raw data file. Sidecar files are an essential component
    of BIDS datasets and are typically found alongside the primary data files with the same
    base name but a `.json` extension.

    For more information on JSON sidecar files, see:
    EEG: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html#sidecar-json-_eegjson
    iEEG: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html#sidecar-json-_ieegjson

    Args:
        bids_path: A BIDSPath object representing the BIDS file for which to load the JSON sidecar.
    Returns:
        Dictionary containing the JSON sidecar data.

    Raises:
        FileNotFoundError: If no JSON sidecar file is found for the BIDS path.
    """
    _check_mne_bids_available("load_json_sidecar")
    if not isinstance(bids_path, BIDSPath):
        raise TypeError(
            f"bids_path must be a BIDSPath object, got {type(bids_path).__name__}."
        )

    try:
        sidecar_path = bids_path.find_matching_sidecar(
            extension=".json", on_error="raise"
        )
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except RuntimeError as err:
        raise FileNotFoundError(f"No JSON sidecar file found for {bids_path}.") from err


def load_participants_tsv(bids_root: Path | str) -> Optional[pd.DataFrame]:
    """Load participants.tsv data from a BIDS root directory.

    The participants.tsv file is a tab-delimited file containing information about all subjects
    in the dataset, such as participant_id, age, sex, and other metadata. This file is part
    of the BIDS standard, and is typically found at the root of the BIDS dataset.

    For more information on participants.tsv, see:
    https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/data-summary-files.html#participants-file

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')

    Returns:
        pd.DataFrame: Participant information indexed by participant_id,
            or None if participants.tsv is missing the 'participant_id' column.

    Raises:
        FileNotFoundError: If participants.tsv file is not found in the BIDS root directory.
    """
    if not (Path(bids_root) / "participants.tsv").exists():
        raise FileNotFoundError(f"participants.tsv file not found in {bids_root}.")

    df = pd.read_csv(
        Path(bids_root) / "participants.tsv",
        sep="\t",
        na_values=["n/a", "N/A"],
        keep_default_na=True,
    )

    if "participant_id" not in df.columns:
        warnings.warn(
            f"No participant_id column found in participants.tsv file in BIDS root directory {bids_root}. "
            "Returning None."
        )
        return None

    df = df.set_index("participant_id")
    return df


def get_subject_info(
    subject_id: str,
    participants_data: pd.DataFrame | None,
) -> dict[str, float | str | None]:
    """Retrieve demographic information (age, sex) for a given subject from a participants DataFrame.

    Args:
        subject_id: BIDS subject identifier (e.g., 'sub-01').
        participants_data: DataFrame of participants.tsv data. If None, returns None for both age and sex.

    Returns:
        Dictionary with keys 'age' and 'sex', each mapping to the value or None if not found.
    """
    if participants_data is None:
        warnings.warn(
            "The participants.tsv file was not provided. No subject information can be retrieved. "
            "Returning None for age and sex. Please provide a valid participants.tsv file."
        )
        return {"age": None, "sex": None}

    if subject_id not in participants_data.index:
        warnings.warn(
            f"Subject {subject_id} not found in participants.tsv file. "
            "Setting age and sex to None."
        )
        return {"age": None, "sex": None}

    row = participants_data.loc[subject_id]

    age = row.get("age", None)
    if pd.isna(age):
        warnings.warn(
            f"Age for subject {subject_id} is NaN in participants.tsv file. "
            "Setting age to None."
        )
        age = None

    sex = row.get("sex", None)
    if pd.isna(sex):
        warnings.warn(
            f"Sex for subject {subject_id} is NaN in participants.tsv file. "
            "Setting sex to None."
        )
        sex = None

    return {"age": age.astype(float) if age is not None else None, "sex": sex}


def _fetch_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
    extensions: set[str],
    modality: _ModalityLiteral,
) -> list[dict]:
    """
    Internal helper for discovering BIDS recordings that match provided file extensions and modality.

    Args:
        source: Either the BIDS root directory (as a str, Path, or BIDSPath),
            or a list of files (each item being a str, Path, or BIDSPath).
            If `source` is a BIDSPath pointing to a subfolder inside the BIDS root directory, this function will
            search for all files within the subfolder corresponding to the given modality and extensions.
            All files matching the extensions for that modality will be returned.
        extensions: Set of allowed file extensions (e.g., EEG_EXTENSIONS).
        modality: Modality to filter by (supported values: 'eeg', 'ieeg').

    Returns:
        List of dicts, each containing key/value pairs for BIDS entities extracted from the fetched recording files.
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acquisition_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Path to the recording file

        For more information on BIDS entities, see:
        https://bids-specification.readthedocs.io/en/stable/appendices/entities.html

    Raises:
        TypeError: If `source` is None or is not a valid type (BIDSPath, Path, str, or list thereof).
        ValueError: If `source` is a directory that is not a valid BIDS root and isn't a BIDSPath, or if modality is not supported.
    """
    _validate_modality(modality)

    # Determine the files to analyze
    if source is None:
        raise TypeError(
            "'source' must be a BIDSPath, Path, or string, or a list of those types. None was provided."
        )

    if isinstance(source, (str, Path)):
        # If the source is a string or Path, check if it is a valid BIDS root directory
        if _is_bids_root(source):
            source = BIDSPath(root=source, datatype=modality).match()
        else:
            # If the source is not a valid BIDS root directory, raise an error
            raise ValueError(
                f"The 'source' parameter points to '{source}', "
                "which is a directory that does not appear to be a valid BIDS root folder. "
                "If 'source' does not point to a BIDS root directory, it must be provided as a BIDSPath object."
            )

    if isinstance(source, BIDSPath):
        source = source.update(datatype=modality).match()

    if len(source) == 0:
        return []

    recordings = []
    seen_recording_ids = set()

    for filepath in source:
        ext = Path(filepath).suffix.lower()
        if ext not in extensions:
            continue

        if not isinstance(filepath, BIDSPath):
            filepath = get_bids_path_from_fname(filepath, check=False)

        if filepath.datatype != modality:
            continue

        components = []
        entities = filepath.entities
        if entities["subject"]:
            components.append(f"sub-{entities['subject']}")
        if entities["session"]:
            components.append(f"ses-{entities['session']}")
        if entities["task"]:
            components.append(f"task-{entities['task']}")
        if entities["acquisition"]:
            components.append(f"acq-{entities['acquisition']}")
        if entities["run"]:
            components.append(f"run-{entities['run']}")
        if entities["description"]:
            components.append(f"desc-{entities['description']}")
        recording_id = "_".join(components)

        if recording_id in seen_recording_ids:
            continue
        seen_recording_ids.add(recording_id)

        recordings.append(
            {
                "recording_id": recording_id,
                "subject_id": (
                    f"sub-{entities['subject']}" if entities["subject"] else None
                ),
                "session_id": (
                    f"ses-{entities['session']}" if entities["session"] else None
                ),
                "task_id": entities["task"] if entities["task"] else None,
                "acquisition_id": (
                    entities["acquisition"] if entities["acquisition"] else None
                ),
                "run_id": entities["run"] if entities["run"] else None,
                "description_id": (
                    entities["description"] if entities["description"] else None
                ),
                "fpath": filepath,
            }
        )

    return recordings


def _check_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
    extensions: set[str],
) -> bool:
    """Check if any data file for a BIDS recording exists in the BIDS root directory.

    Note: The BIDS root directory is the top-level folder of a BIDS dataset.
    All data and metadata within the dataset are organized relative to this root directory.

    This searches for any file belonging to the recording in the proper BIDS directory structure,
    matching any of the supported file extensions. It supports all BIDS-compliant formats (e.g., .edf, .vhdr, .set, .bdf, .eeg, .nwb)
    plus .fif for MNE-processed files.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
        extensions: Set of allowed file extensions (e.g., EEG_EXTENSIONS or IEEG_EXTENSIONS)

    Returns:
        True if at least one data file is found, False otherwise.
    """
    entities = get_entities_from_fname(recording_id, on_error="raise")
    subject_dir = Path(bids_root) / f"sub-{entities['subject']}"

    if not subject_dir.exists():
        return False

    for file in subject_dir.rglob(f"{recording_id}_*"):
        if file.suffix.lower() in extensions:
            return True

    return False


def _is_bids_root(path: str | Path) -> bool:
    """
    Determine whether a given filesystem path corresponds to a valid BIDS root directory.

    A valid BIDS root directory must:
      - Be a directory (not a file).
      - Contain at least one subject folder (e.g., a sub-XX directory in BIDS naming convention).
        This is checked using get_entity_vals(path, 'subject'), which searches for subject-level entities.

    Args:
        path (str or Path): The path to check.

    Returns:
        bool: True if the path is a valid BIDS root (i.e., it is a directory and has at least one BIDS subject entity).
              False otherwise.

    Example:
        >>> _is_bids_root("/path/to/bids/dataset")
        True
    """
    _check_mne_bids_available("_is_bids_root")

    if not isinstance(path, Path):
        path = Path(path)

    # Validate that the path is a folder
    is_folder = path.is_dir()

    # Check for mandatory root file
    is_root = (path / "dataset_description.json").exists()

    # Check for BIDS entities (subjects) to ensure it's a BIDS structure
    # If we are in a subfolder, get_entity_vals will still work because it
    # looks at the directory names in the path.
    try:
        subjects = get_entity_vals(path, "subject")
        has_bids_structure = len(subjects) > 0
    except Exception:
        has_bids_structure = False

    if is_folder and is_root and has_bids_structure:
        return True
    return False


def _validate_modality(modality: _ModalityLiteral) -> None:
    """Validate that the provided modality is supported both by this module and by the BIDS specification.

    Verifies that the modality string is one of the supported values defined in
    _SUPPORTED_MODALITIES. This function is used to ensure type safety
    at runtime for modality parameters passed to functions.

    Args:
        modality: Modality string to validate. Must be one of the supported values in _SUPPORTED_MODALITIES.

    Raises:
        ValueError: If modality is not one of the supported values in _SUPPORTED_MODALITIES.

    Examples:
        >>> _validate_modality("eeg")  # No error
        >>> _validate_modality("ieeg")  # No error
        >>> _validate_modality("EEG")  # Raises ValueError
        >>> _validate_modality("unsupported_modality")  # Raises ValueError
    """
    if modality not in _SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported modality '{modality}'. Expected one of: {_SUPPORTED_MODALITIES}."
        )
