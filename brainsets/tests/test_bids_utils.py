from pathlib import Path
from io import StringIO
import shutil

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    import mne_bids
    from mne_bids import BIDSPath

    MNE_BIDS_AVAILABLE = True
except ImportError:
    BIDSPath = None
    MNE_BIDS_AVAILABLE = False

try:
    from brainsets.utils.bids_utils import (
        EEG_EXTENSIONS,
        IEEG_EXTENSIONS,
        fetch_eeg_recordings,
        fetch_ieeg_recordings,
        check_eeg_recording_files_exist,
        build_bids_path,
        load_json_sidecar,
        get_subject_info,
        group_recordings_by_entity,
        load_participants_tsv,
        _fetch_recordings,
        _validate_modality,
    )
except ImportError:
    EEG_EXTENSIONS = None
    IEEG_EXTENSIONS = None
    fetch_eeg_recordings = None
    fetch_ieeg_recordings = None
    check_eeg_recording_files_exist = None
    build_bids_path = None
    load_json_sidecar = None
    get_subject_info = None
    group_recordings_by_entity = None
    load_participants_tsv = None
    _fetch_recordings = None
    _validate_modality = None


# ============================================================================
# BIDS-Specific Fixtures (for test_bids_utils.py only)
# ============================================================================
@pytest.fixture
def bids_root(tmp_path):
    """Create a temporary BIDS root directory.

    This is a generic fixture for tests that need a temporary BIDS-compliant
    directory structure. Use this as a base for other fixtures that need to
    populate the directory with test data.

    Returns:
        Path: Path to the temporary BIDS root directory.
    """
    bids_root = tmp_path / "ds000001"
    (bids_root / "sub-01" / "ses-01" / "eeg").mkdir(parents=True, exist_ok=True)
    (bids_root / "sub-02" / "ses-01" / "eeg").mkdir(parents=True, exist_ok=True)
    (
        bids_root
        / "sub-01"
        / "ses-01"
        / "eeg"
        / "sub-01_ses-01_task-rest_run-01_eeg.edf"
    ).write_bytes(b"FAKE1")
    (
        bids_root
        / "sub-02"
        / "ses-01"
        / "eeg"
        / "sub-02_ses-01_task-rest_run-01_eeg.bdf"
    ).write_bytes(b"FAKE2")
    (
        bids_root
        / "sub-01"
        / "ses-01"
        / "eeg"
        / "sub-01_ses-01_task-rest_run-02_eeg.edf"
    ).write_bytes(b"FAKE3")
    (
        bids_root
        / "sub-02"
        / "ses-01"
        / "eeg"
        / "sub-02_ses-01_task-rest_run-02_eeg.bdf"
    ).write_bytes(b"FAKE4")
    (
        bids_root
        / "sub-01"
        / "ses-01"
        / "eeg"
        / "sub-01_ses-01_task-rest_run-03_eeg.edf"
    ).write_bytes(b"FAKE5")
    (
        bids_root
        / "sub-02"
        / "ses-01"
        / "eeg"
        / "sub-02_ses-01_task-rest_run-03_eeg.bdf"
    ).write_bytes(b"FAKE6")
    (
        bids_root
        / "sub-01"
        / "ses-01"
        / "eeg"
        / "sub-01_ses-01_task-rest_run-04_events.tsv"
    ).write_text("dummy")
    (bids_root / "dataset_description.json").write_text("{}")

    (bids_root / "dataset_description.json").write_text("{}")
    return bids_root


@pytest.fixture
def bids_root_with_participants(bids_root):
    """Create a BIDS root directory with a participants.tsv file.

    Includes three sample participants with varied demographic data:
    - sub-01: age=34, sex=F
    - sub-02: age=n/a (missing), sex=N/A (missing)
    - sub-03: age=28, sex=M

    Args:
        bids_root: Temporary BIDS root directory fixture.

    Returns:
        Path: Path to the BIDS root directory with participants.tsv.
    """
    participants_tsv = bids_root / "participants.tsv"
    participants_tsv.write_text(
        "participant_id\tage\tsex\n"
        "sub-01\t34\tF\n"
        "sub-02\tn/a\tN/A\n"
        "sub-03\t28\tM\n"
    )
    return bids_root


@pytest.fixture
def invalid_bids_root(tmp_path):
    """Create an invalid BIDS root directory.

    Returns:
        Path: Path to the invalid BIDS root directory.
    """
    bids_root = tmp_path / "ds000001"
    return bids_root


def _make_participants_df(tsv_content: str) -> pd.DataFrame:
    """Helper to create a participants DataFrame from TSV content.

    Args:
        tsv_content: TSV formatted string with participant data.

    Returns:
        DataFrame indexed by participant_id.
    """
    return pd.read_csv(
        StringIO(tsv_content),
        sep="\t",
        na_values=["n/a", "N/A"],
        keep_default_na=True,
    ).set_index("participant_id")


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestValidateModality:
    """Test the _validate_modality internal validation function."""

    def test_accepts_valid_eeg_modality(self):
        """Test that 'eeg' modality is accepted without raising an error."""
        _validate_modality("eeg")

    def test_accepts_valid_ieeg_modality(self):
        """Test that 'ieeg' modality is accepted without raising an error."""
        _validate_modality("ieeg")

    def test_raises_value_error_for_unsupported_modality(self):
        """Test that unsupported modalities raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            _validate_modality("meg")

    def test_raises_value_error_for_uppercase_modality(self):
        """Test that uppercase modalities raise ValueError (case-sensitive)."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            _validate_modality("EEG")

    def test_raises_value_error_for_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            _validate_modality("")


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestFetchRecordings:
    """Test the _fetch_recordings internal function and its wrappers."""

    @pytest.fixture
    def eeg_source_paths(self):
        """Create a list of EEG source file paths."""
        return [
            Path("sub-01/eeg/sub-01_task-Sleep_eeg.edf"),
            "sub-01/eeg/sub-01_task-Sleep_channels.tsv",
            Path("sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vhdr"),
            "sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vmrk",
            "participants.tsv",
        ]

    @pytest.fixture
    def mixed_extension_paths(self):
        """Create a mixed source with multiple modalities and file extensions."""
        return [
            # EEG recordings (valid + duplicate + unsupported sidecars)
            Path(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-01_eeg.edf"
            ),
            Path(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-01_eeg.vhdr"
            ),  # duplicate ID
            Path(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-01_eeg.vmrk"
            ),
            Path(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-02_eeg.bdf"
            ),
            "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-03_eeg.set",
            Path(
                "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_acq-headband_run-03_eeg.json"
            ),
            # iEEG recordings (valid + unsupported sidecars)
            Path("sub-02/ieeg/sub-02_task-visual_run-01_ieeg.nwb"),
            Path("sub-02/ieeg/sub-02_task-visual_run-02_ieeg.edf"),
            Path("sub-02/ieeg/sub-02_task-visual_run-03_ieeg.vhdr"),
            Path("sub-02/ieeg/sub-02_task-visual_run-03_ieeg.eeg"),
            Path("sub-02/ieeg/sub-02_task-visual_run-03_ieeg.vmrk"),
            # Other dataset files / other modality
            Path("sub-03/meg/sub-03_task-rest_meg.fif"),
            Path("participants.tsv"),
            Path("dataset_description.json"),
        ]

    # ----------- Basic input validation tests -----------
    def test_raises_type_error_when_source_is_none(self):
        """Test that _fetch_recordings raises TypeError when source is None."""
        with pytest.raises(
            TypeError,
            match="'source' must be a BIDSPath, Path, or string, or a list of those types. None was provided.",
        ):
            _fetch_recordings(None, EEG_EXTENSIONS, "eeg")

    def test_accepts_list_of_strings_and_paths_sources(self, eeg_source_paths):
        """Test that _fetch_recordings handles both string and Path source types."""
        recordings = _fetch_recordings(
            eeg_source_paths,
            EEG_EXTENSIONS,
            "eeg",
        )

        assert len(recordings) == 2

        rec1 = next(r for r in recordings if r["subject_id"] == "sub-01")
        assert rec1["recording_id"] == "sub-01_task-Sleep"
        assert rec1["task_id"] == "Sleep"
        assert rec1["session_id"] is None

        rec2 = next(r for r in recordings if r["subject_id"] == "sub-02")
        assert rec2["recording_id"] == "sub-02_ses-01_task-Rest_acq-headband_run-01"
        assert rec2["task_id"] == "Rest"
        assert rec2["session_id"] == "ses-01"
        assert rec2["acquisition_id"] == "headband"
        assert rec2["run_id"] == "01"

    def test_accepts_bids_root_as_path_or_string(self, bids_root):
        """Test that a BIDS root passed as Path or str is accepted and scanned."""
        try:
            recordings_from_path = _fetch_recordings(
                source=bids_root,
                extensions=EEG_EXTENSIONS,
                modality="eeg",
            )
            recordings_from_str = _fetch_recordings(
                source=str(bids_root),
                extensions=EEG_EXTENSIONS,
                modality="eeg",
            )

            ids_from_path = sorted(r["recording_id"] for r in recordings_from_path)
            ids_from_str = sorted(r["recording_id"] for r in recordings_from_str)

            assert len(recordings_from_path) == 6
            assert len(recordings_from_str) == 6

            assert ids_from_path == [
                "sub-01_ses-01_task-rest_run-01",
                "sub-01_ses-01_task-rest_run-02",
                "sub-01_ses-01_task-rest_run-03",
                "sub-02_ses-01_task-rest_run-01",
                "sub-02_ses-01_task-rest_run-02",
                "sub-02_ses-01_task-rest_run-03",
            ]
            assert ids_from_str == [
                "sub-01_ses-01_task-rest_run-01",
                "sub-01_ses-01_task-rest_run-02",
                "sub-01_ses-01_task-rest_run-03",
                "sub-02_ses-01_task-rest_run-01",
                "sub-02_ses-01_task-rest_run-02",
                "sub-02_ses-01_task-rest_run-03",
            ]
        finally:
            # Clean up after test: remove created bids_root and all its contents
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_accepts_bids_subfolder_as_bids_path(self, bids_root):
        """Test that files are returned when source is a BIDSPath pointing to a subfolder inside BIDS root."""
        try:
            # Point source to the EEG folder inside BIDS, not the BIDS root itself
            source = BIDSPath(root=bids_root, subject="01", session="01")
            recordings = fetch_eeg_recordings(source=source)

            assert isinstance(recordings, list)
            assert len(recordings) == 3
            assert [r["recording_id"] for r in recordings] == [
                "sub-01_ses-01_task-rest_run-01",
                "sub-01_ses-01_task-rest_run-02",
                "sub-01_ses-01_task-rest_run-03",
            ]
            assert str(recordings[0]["fpath"].basename).endswith("_eeg.edf")
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_raises_value_error_for_bids_subfolder_as_path_or_string(self, bids_root):
        """Test that non-BIDS-root Path/str sources are rejected with ValueError."""
        subject_folder = bids_root / "sub-01"
        try:
            with pytest.raises(
                ValueError, match="does not appear to be a valid BIDS root"
            ):
                _fetch_recordings(
                    source=subject_folder,
                    extensions=EEG_EXTENSIONS,
                    modality="eeg",
                )

            with pytest.raises(
                ValueError, match="does not appear to be a valid BIDS root"
            ):
                _fetch_recordings(
                    source=str(subject_folder),
                    extensions=EEG_EXTENSIONS,
                    modality="eeg",
                )
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    # ----------- Basic behavior tests -----------
    def test_filters_by_extension_modality_and_deduplicates(
        self, mixed_extension_paths
    ):
        """Test extension filtering, modality filtering, and deduplication of recording IDs.

        The function should:
        - Filter out unsupported extensions (.json, .tsv)
        - Filter out mismatched modalities (ieeg when eeg is requested)
        - Deduplicate by recording_id (keep only first occurrence)
        """
        recordings = _fetch_recordings(
            mixed_extension_paths,
            EEG_EXTENSIONS,
            "eeg",
        )

        assert len(recordings) == 3
        assert [r["recording_id"] for r in recordings] == [
            "sub-01_ses-01_task-rest_acq-headband_run-01",
            "sub-01_ses-01_task-rest_acq-headband_run-02",
            "sub-01_ses-01_task-rest_acq-headband_run-03",
        ]
        assert all(r["subject_id"] == "sub-01" for r in recordings)
        assert all(r["session_id"] == "ses-01" for r in recordings)
        assert all(r["task_id"] == "rest" for r in recordings)
        assert recordings[0]["acquisition_id"] == "headband"
        assert recordings[1]["acquisition_id"] == "headband"
        assert recordings[2]["acquisition_id"] == "headband"
        assert recordings[0]["run_id"] == "01"
        assert recordings[1]["run_id"] == "02"
        assert recordings[2]["run_id"] == "03"
        assert "sub-01_ses-01_task-rest_acq-headband_run-01_eeg.edf" in str(
            recordings[0]["fpath"]
        )
        assert "sub-01_ses-01_task-rest_acq-headband_run-02_eeg.bdf" in str(
            recordings[1]["fpath"]
        )
        assert "sub-01_ses-01_task-rest_acq-headband_run-03_eeg.set" in str(
            recordings[2]["fpath"]
        )

    def test_fetch_eeg_recordings_wrapper_returns_eeg_files(
        self, mixed_extension_paths
    ):
        """Test fetch_eeg_recordings wrapper function.

        Verifies that the wrapper correctly filters EEG extensions and populates
        all entity fields in the returned recording dictionaries.
        """
        recordings = fetch_eeg_recordings(source=mixed_extension_paths)

        assert len(recordings) == 3

        assert (
            recordings[0]["recording_id"]
            == "sub-01_ses-01_task-rest_acq-headband_run-01"
        )
        assert recordings[0]["subject_id"] == "sub-01"
        assert recordings[0]["session_id"] == "ses-01"
        assert recordings[0]["task_id"] == "rest"
        assert recordings[0]["acquisition_id"] == "headband"
        assert recordings[0]["run_id"] == "01"
        assert recordings[0]["description_id"] is None
        assert "sub-01_ses-01_task-rest_acq-headband_run-01_eeg.edf" in str(
            recordings[0]["fpath"]
        )

        assert (
            recordings[1]["recording_id"]
            == "sub-01_ses-01_task-rest_acq-headband_run-02"
        )
        assert recordings[1]["subject_id"] == "sub-01"
        assert recordings[1]["session_id"] == "ses-01"
        assert recordings[1]["task_id"] == "rest"
        assert recordings[1]["acquisition_id"] == "headband"
        assert recordings[1]["run_id"] == "02"
        assert recordings[1]["description_id"] is None
        assert "sub-01_ses-01_task-rest_acq-headband_run-02_eeg.bdf" in str(
            recordings[1]["fpath"]
        )

        assert (
            recordings[2]["recording_id"]
            == "sub-01_ses-01_task-rest_acq-headband_run-03"
        )
        assert recordings[2]["subject_id"] == "sub-01"
        assert recordings[2]["session_id"] == "ses-01"
        assert recordings[2]["task_id"] == "rest"
        assert recordings[2]["acquisition_id"] == "headband"
        assert recordings[2]["run_id"] == "03"
        assert recordings[2]["description_id"] is None
        assert "sub-01_ses-01_task-rest_acq-headband_run-03_eeg.set" in str(
            recordings[2]["fpath"]
        )

    def test_fetch_ieeg_recordings_wrapper_return_ieeg_files(
        self, mixed_extension_paths
    ):
        """Test fetch_ieeg_recordings wrapper function.

        Verifies that the wrapper correctly filters iEEG extensions including .nwb format.
        """
        recordings = fetch_ieeg_recordings(source=mixed_extension_paths)

        assert len(recordings) == 3
        assert recordings[0]["recording_id"] == "sub-02_task-visual_run-01"
        assert "sub-02_task-visual_run-01_ieeg.nwb" in str(recordings[0]["fpath"])
        assert recordings[1]["recording_id"] == "sub-02_task-visual_run-02"
        assert "sub-02_task-visual_run-02_ieeg.edf" in str(recordings[1]["fpath"])
        assert recordings[2]["recording_id"] == "sub-02_task-visual_run-03"
        assert "sub-02_task-visual_run-03_ieeg.vhdr" in str(recordings[2]["fpath"])

    def test_returns_empty_list_for_no_matching_files(self):
        """Test that empty source list returns empty recordings."""
        source = [
            Path("participants.tsv"),
            Path("dataset_description.json"),
        ]

        recordings = fetch_eeg_recordings(source=source)
        assert len(recordings) == 0


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestGroupRecordingsByEntity:
    """Test grouping of BIDS recordings by specified entities."""

    @pytest.fixture
    def recording_with_description(self):
        """Create a recording with description entity."""
        return {
            "recording_id": "sub-01_task-rest_desc-clean",
            "subject_id": "sub-01",
            "session_id": None,
            "task_id": "rest",
            "acquisition_id": None,
            "run_id": None,
            "description_id": "clean",
            "fpath": Path("sub-01/eeg/sub-01_task-rest_desc-clean_eeg.edf"),
        }

    @pytest.fixture
    def recordings_with_multiple_runs(self):
        """Create recordings with multiple runs of the same task."""
        return [
            {
                "recording_id": "sub-01_task-rest_run-01_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "01",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-01_desc-clean_eeg.edf"),
            },
            {
                "recording_id": "sub-01_task-rest_run-02_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "02",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-02_desc-clean_eeg.edf"),
            },
            {
                "recording_id": "sub-01_task-rest_run-03_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "03",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-03_desc-clean_eeg.edf"),
            },
        ]

    @pytest.fixture
    def recordings_with_non_numeric_runs(self):
        """Create recordings with non-numeric run identifiers."""
        return [
            {
                "recording_id": "sub-01_task-rest_run-A_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "A",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-A_desc-clean_eeg.edf"),
            },
            {
                "recording_id": "sub-01_task-rest_run-B_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "B",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-B_desc-clean_eeg.edf"),
            },
        ]

    @pytest.fixture
    def recordings_with_different_descriptions(self):
        """Create recordings with different description suffixes (should not group)."""
        return [
            {
                "recording_id": "sub-01_task-rest_run-01_desc-clean",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "01",
                "description_id": "clean",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-01_desc-clean_eeg.edf"),
            },
            {
                "recording_id": "sub-01_task-rest_run-02_desc-raw",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": "02",
                "description_id": "raw",
                "fpath": Path("sub-01/eeg/sub-01_task-rest_run-02_desc-raw_eeg.edf"),
            },
        ]

    @pytest.fixture
    def recordings_with_multiple_sessions(self):
        """Create recordings with different sessions (for grouping by specific entities)."""
        return [
            {
                "recording_id": "sub-01_ses-01_task-rest_desc-clean",
                "subject_id": "sub-01",
                "session_id": "ses-01",
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": None,
                "description_id": "clean",
                "fpath": Path(
                    "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_desc-clean_eeg.edf"
                ),
            },
            {
                "recording_id": "sub-01_ses-02_task-rest_desc-clean",
                "subject_id": "sub-01",
                "session_id": "ses-02",
                "task_id": "rest",
                "acquisition_id": None,
                "run_id": None,
                "description_id": "clean",
                "fpath": Path(
                    "sub-01/ses-02/eeg/sub-01_ses-02_task-rest_desc-clean_eeg.edf"
                ),
            },
        ]

    def test_groups_multiple_runs_with_same_entities(
        self, recordings_with_multiple_runs
    ):
        """Test that recordings with different run IDs are grouped together.

        By default, the 'run' entity is excluded from grouping keys, so multiple runs
        of the same task/session/subject should be grouped under one key.
        """
        grouped = group_recordings_by_entity(recordings_with_multiple_runs)

        assert list(grouped.keys()) == ["sub-01_task-rest_desc-clean"]
        assert len(grouped["sub-01_task-rest_desc-clean"]) == 3
        assert [
            rec["recording_id"] for rec in grouped["sub-01_task-rest_desc-clean"]
        ] == [
            "sub-01_task-rest_run-01_desc-clean",
            "sub-01_task-rest_run-02_desc-clean",
            "sub-01_task-rest_run-03_desc-clean",
        ]

    def test_keeps_non_run_recording_as_single_group(self, recording_with_description):
        """Test that a recording without a run entity is kept as its own group."""
        grouped = group_recordings_by_entity([recording_with_description])

        assert grouped == {"sub-01_task-rest_desc-clean": [recording_with_description]}

    def test_does_not_group_recordings_with_different_suffixes(
        self, recordings_with_different_descriptions
    ):
        """Test that recordings with different description suffixes are not grouped.

        Recordings are only grouped if all non-variable entities match.
        """
        grouped = group_recordings_by_entity(recordings_with_different_descriptions)

        assert len(grouped) == 2
        assert "sub-01_task-rest_desc-clean" in grouped
        assert "sub-01_task-rest_desc-raw" in grouped

    def test_groups_non_numeric_run_identifiers(self, recordings_with_non_numeric_runs):
        """Test that non-numeric run IDs (e.g., 'A', 'B') are also grouped correctly."""
        grouped = group_recordings_by_entity(recordings_with_non_numeric_runs)

        assert list(grouped.keys()) == ["sub-01_task-rest_desc-clean"]
        assert [
            rec["recording_id"] for rec in grouped["sub-01_task-rest_desc-clean"]
        ] == [
            "sub-01_task-rest_run-A_desc-clean",
            "sub-01_task-rest_run-B_desc-clean",
        ]

    def test_groups_by_custom_fixed_entities_long_names(
        self, recordings_with_multiple_sessions
    ):
        """Test grouping with custom fixed entities using long entity names.

        When fixed_entities=['subject', 'task', 'description'], sessions
        should be allowed to vary within a group.
        """
        grouped = group_recordings_by_entity(
            recordings_with_multiple_sessions,
            fixed_entities=["subject", "task", "description"],
        )

        assert list(grouped.keys()) == ["sub-01_task-rest_desc-clean"]
        assert len(grouped["sub-01_task-rest_desc-clean"]) == 2

    def test_groups_by_custom_fixed_entities_short_names(
        self, recordings_with_multiple_sessions
    ):
        """Test grouping with custom fixed entities using short BIDS entity names.

        Short names (sub, task, desc) should work the same as long names.
        """
        grouped = group_recordings_by_entity(
            recordings_with_multiple_sessions,
            fixed_entities=["sub", "task", "desc"],
        )

        assert list(grouped.keys()) == ["sub-01_task-rest_desc-clean"]

    def test_raises_value_error_for_unsupported_entity(
        self, recording_with_description
    ):
        """Test that an unsupported entity name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported BIDS entity"):
            group_recordings_by_entity(
                [recording_with_description],
                fixed_entities=["not-an-entity"],
            )


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestCheckRecordingFilesExist:
    """Test checking for existence of BIDS recording files."""

    @pytest.fixture
    def bids_dir_with_invalid_recording_file(self, bids_root):
        """Create a BIDS directory with an invalid recording file."""
        invalid_recording_file = (
            bids_root
            / "sub-01"
            / "ses-01"
            / "eeg"
            / "sub-01_ses-01_task-rest_run-04_events.tsv"
        )
        invalid_recording_file.write_text("dummy")
        return bids_root

    def test_returns_false_when_recording_file_missing(
        self, bids_dir_with_invalid_recording_file
    ):
        """Test that False is returned when the subject directory doesn't exist."""
        try:
            assert (
                check_eeg_recording_files_exist(
                    bids_dir_with_invalid_recording_file,
                    "sub-01_ses-01_task-rest_run-04",
                )
                is False
            )
        finally:
            shutil.rmtree(bids_dir_with_invalid_recording_file, ignore_errors=True)

    def test_returns_true_when_matching_recording_file_exists(self, bids_root):
        """Test that True is returned when a supported EEG file exists.

        The check should be case-insensitive for file extensions.
        """
        try:
            assert (
                check_eeg_recording_files_exist(
                    bids_root, "sub-01_ses-01_task-rest_run-01"
                )
                is True
            )
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestBuildBidsPath:
    """Test building mne_bids.BIDSPath objects from recording IDs."""

    def test_builds_complete_bids_path_with_all_entities(self, bids_root):
        """Test that build_bids_path correctly parses all BIDS entities.

        Verifies that all entities (subject, session, task, acquisition, run, description)
        are correctly extracted and set on the BIDSPath object.
        """
        try:
            bids_path = build_bids_path(
                bids_root=bids_root,
                recording_id="sub-01_ses-02_task-rest_acq-ecog_run-03_desc-preproc",
                modality="ieeg",
            )

            assert bids_path.root == bids_root
            assert bids_path.subject == "01"
            assert bids_path.session == "02"
            assert bids_path.task == "rest"
            assert bids_path.acquisition == "ecog"
            assert bids_path.run == "03"
            assert bids_path.description == "preproc"
            assert bids_path.datatype == "ieeg"
            assert bids_path.suffix == "ieeg"
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_builds_minimal_bids_path_with_required_entities(self, bids_root):
        """Test that build_bids_path works with only required entities."""
        try:
            bids_path = build_bids_path(
                bids_root=bids_root,
                recording_id="sub-01_task-rest",
                modality="eeg",
            )

            assert bids_path.subject == "01"
            assert bids_path.task == "rest"
            assert bids_path.session is None
            assert bids_path.acquisition is None
            assert bids_path.run is None
            assert bids_path.description is None
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_handles_missing_entities_gracefully(self, bids_root):
        """Test that build_bids_path extracts available entities even when some are missing.

        The function uses get_entities_from_fname which allows missing entities
        and will return None for them.
        """
        try:
            bids_path = build_bids_path(
                bids_root,
                "sub-01_ses-02_acq-ecog_run-03_desc-preproc",
                "eeg",
            )
            assert bids_path.subject == "01"
            assert bids_path.task is None
            assert bids_path.acquisition == "ecog"
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_builds_path_with_minimal_entities(self, bids_root):
        """Test that build_bids_path works with only subject entity.

        The function requires only subject to construct a valid BIDSPath.
        """
        try:
            bids_path = build_bids_path(bids_root, "sub-01", "eeg")
            assert bids_path.subject == "01"
            assert bids_path.task is None
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_raises_value_error_for_unsupported_modality(self, bids_root):
        """Test that unsupported modalities are rejected."""
        try:
            with pytest.raises(ValueError, match="Unsupported modality"):
                build_bids_path(bids_root, "sub-01_task-rest", "meg")
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_raises_value_error_for_unsupported_entity(self, bids_root):
        """Test that ValueError is raised if recording_id contains unsupported entity names."""
        # entity 'foo' is not a supported BIDS entity
        try:
            recording_id = "sub-01_task-rest_foo-bar_eeg.edf"
            with pytest.raises(ValueError, match="Unsupported BIDS entity"):
                build_bids_path(bids_root, recording_id, "eeg")
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_raises_value_error_for_invalid_bids_root(self, invalid_bids_root):
        """Test that ValueError is raised if bids_root does not correspond to a valid BIDS root directory."""
        # invalid_bids_root is an empty dir and does not have a dataset_description.json
        try:
            recording_id = "sub-01_task-rest"
            with pytest.raises(
                ValueError, match="must point to a valid BIDS root directory"
            ):
                build_bids_path(
                    bids_root=invalid_bids_root,
                    recording_id=recording_id,
                    modality="eeg",
                )
        finally:
            shutil.rmtree(invalid_bids_root, ignore_errors=True)


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestLoadJsonSidecar:
    """Test loading JSON sidecar files for BIDS data."""

    @pytest.fixture
    def bids_dir_with_json_sidecar(self, bids_root):
        """Create a BIDS directory with a JSON sidecar file.

        Args:
            bids_root: Temporary BIDS root directory.

        Returns:
            Path: Path to the created BIDS root directory.
        """
        sidecar_path = bids_root / "sub-01" / "ieeg" / "sub-01_task-rest_ieeg.json"
        sidecar_path.parent.mkdir(parents=True)
        sidecar_path.write_text('{"OriginalRecordingTimestamp": "2024-01-01T10:00:00"}')
        return bids_root

    def test_accepts_bidspath_input(self, bids_dir_with_json_sidecar):
        """Test that load_json_sidecar correctly accepts BIDSPath input."""
        try:
            bids_path = mne_bids.BIDSPath(
                root=bids_dir_with_json_sidecar,
                subject="01",
                task="rest",
                datatype="ieeg",
                suffix="ieeg",
            )
            sidecar = load_json_sidecar(bids_path)
            assert sidecar["OriginalRecordingTimestamp"] == "2024-01-01T10:00:00"
        finally:
            shutil.rmtree(bids_dir_with_json_sidecar, ignore_errors=True)

    def test_raises_type_error_for_non_bidspath_input(self):
        """Test that TypeError is raised if bids_path is not a BIDSPath object."""
        not_a_bidspath = "this/is/not/a/BIDSPath"
        with pytest.raises(TypeError, match="bids_path must be a BIDSPath object"):
            load_json_sidecar(not_a_bidspath)

    def test_loads_json_sidecar_content(self, bids_dir_with_json_sidecar):
        """Test that JSON sidecar content is correctly loaded and parsed.

        The function should return a dictionary with the sidecar JSON content.
        """
        try:
            bids_path = mne_bids.BIDSPath(
                root=bids_dir_with_json_sidecar,
                subject="01",
                task="rest",
                datatype="ieeg",
                suffix="ieeg",
            )
            sidecar = load_json_sidecar(bids_path)
            assert sidecar["OriginalRecordingTimestamp"] == "2024-01-01T10:00:00"
        finally:
            shutil.rmtree(bids_dir_with_json_sidecar, ignore_errors=True)

    def test_raises_file_not_found_when_sidecar_missing(self, bids_root):
        """Test that FileNotFoundError is raised when JSON sidecar doesn't exist."""
        try:
            bids_path = mne_bids.BIDSPath(
                root=bids_root,
                subject="01",
                task="rest",
                datatype="ieeg",
                suffix="ieeg",
            )
            with pytest.raises(
                FileNotFoundError, match="No JSON sidecar file found for"
            ):
                load_json_sidecar(bids_path)
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestLoadParticipantsTsv:
    """Test loading participants.tsv files from BIDS datasets."""

    def test_raises_file_not_found_when_missing(self, bids_root):
        """Test that FileNotFoundError is raised when participants.tsv doesn't exist."""
        try:
            with pytest.raises(
                FileNotFoundError, match="participants.tsv file not found in"
            ):
                load_participants_tsv(bids_root)
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_returns_none_without_participant_id_column(self, bids_root):
        """Test that None is returned when participant_id column is missing.

        This allows graceful handling of malformed participants.tsv files.
        """
        try:
            participants_tsv = bids_root / "participants.tsv"
            participants_tsv.write_text("subject_id\tage\tsex\nsub-01\t34\tF\n")

            participants_data = load_participants_tsv(bids_root)
            assert participants_data is None
        finally:
            shutil.rmtree(bids_root, ignore_errors=True)

    def test_loads_and_indexes_by_participant_id(self, bids_root_with_participants):
        """Test that participants.tsv is correctly loaded and indexed by participant_id.

        The DataFrame should be indexed by participant_id with other columns preserved.
        """
        try:
            participants_data = load_participants_tsv(bids_root_with_participants)

            assert participants_data is not None
            assert participants_data.index.name == "participant_id"
            assert list(participants_data.index) == ["sub-01", "sub-02", "sub-03"]
        finally:
            shutil.rmtree(bids_root_with_participants, ignore_errors=True)

    def test_preserves_column_data_types(self, bids_root_with_participants):
        """Test that data types are correctly preserved when loading.

        Numeric columns should remain numeric, and string columns should remain strings.
        """
        try:
            participants_data = load_participants_tsv(bids_root_with_participants)

            assert participants_data.loc["sub-01", "age"] == 34
            assert participants_data.loc["sub-01", "sex"] == "F"
        finally:
            shutil.rmtree(bids_root_with_participants, ignore_errors=True)

    def test_handles_na_values_correctly(self, bids_root_with_participants):
        """Test that NA values (n/a, N/A) are converted to NaN/None.

        This ensures consistent handling of missing data across the dataset.
        """
        try:
            participants_data = load_participants_tsv(bids_root_with_participants)

            assert pd.isna(participants_data.loc["sub-02", "age"])
            assert pd.isna(participants_data.loc["sub-02", "sex"])
        finally:
            shutil.rmtree(bids_root_with_participants, ignore_errors=True)


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestGetSubjectInfo:
    """Test retrieval of subject demographic information."""

    @pytest.fixture
    def participants_df_with_age_sex(self):
        """Create a participants DataFrame with age and sex information."""
        return _make_participants_df(
            "participant_id\tage\tsex\n" "sub-01\t34\tF\n" "sub-02\t28\tM\n"
        )

    @pytest.fixture
    def participants_df_with_na_values(self):
        """Create a participants DataFrame with NA/N/A values."""
        return _make_participants_df(
            "participant_id\tage\tsex\n" "sub-01\tn/a\tN/A\n" "sub-02\t28\tM\n"
        )

    @pytest.fixture
    def participants_df_missing_age_sex(self):
        """Create a participants DataFrame without age and sex columns."""
        return _make_participants_df("participant_id\thandedness\n" "sub-01\tright\n")

    def test_returns_age_and_sex_when_available(self, participants_df_with_age_sex):
        """Test that age and sex are correctly returned when available in participants.tsv."""
        subject_info = get_subject_info(
            "sub-01", participants_data=participants_df_with_age_sex
        )
        assert subject_info == {"age": 34, "sex": "F"}

    def test_returns_none_for_missing_subject(self, participants_df_with_age_sex):
        """Test that None values are returned for subjects not found in participants.tsv."""
        subject_info = get_subject_info(
            "sub-99", participants_data=participants_df_with_age_sex
        )
        assert subject_info == {"age": None, "sex": None}

    def test_normalizes_na_values_to_none(self, participants_df_with_na_values):
        """Test that NA/N/A values in participants.tsv are normalized to None."""
        subject_info = get_subject_info(
            "sub-01", participants_data=participants_df_with_na_values
        )
        assert subject_info == {"age": None, "sex": None}

    def test_returns_none_when_required_columns_missing(
        self, participants_df_missing_age_sex
    ):
        """Test that None is returned for age and sex when those columns don't exist."""
        subject_info = get_subject_info(
            "sub-01", participants_data=participants_df_missing_age_sex
        )
        assert subject_info == {"age": None, "sex": None}

    def test_returns_none_values_when_no_participants_data_provided(self):
        """Test that None values are returned when participants_data is not provided.

        This allows the function to degrade gracefully without participants.tsv.
        """
        subject_info = get_subject_info("sub-01", participants_data=None)
        assert subject_info == {"age": None, "sex": None}

    def test_handles_partial_data_in_participants_tsv(
        self, participants_df_with_age_sex
    ):
        """Test that get_subject_info correctly returns available data.

        When a subject exists with age and sex, both should be returned as expected values.
        """
        subject_info = get_subject_info(
            "sub-01", participants_data=participants_df_with_age_sex
        )
        assert subject_info["age"] == 34
        assert subject_info["sex"] == "F"


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
def test_ieeg_extensions_include_nwb():
    """Test that IEEG_EXTENSIONS includes the .nwb format.

    NWB (Neurodata Without Borders) is an important format for iEEG recordings.
    """
    assert ".nwb" in IEEG_EXTENSIONS


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
def test_eeg_extensions_do_not_include_nwb():
    """Test that EEG_EXTENSIONS does not include .nwb format.

    NWB is specific to iEEG recordings, not standard EEG.
    """
    assert ".nwb" not in EEG_EXTENSIONS


class TestCheckMneBidsAvailable:
    """Test that functions raise ImportError when MNE_BIDS is not available."""

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_fetch_eeg_recordings_raises_import_error(self):
        """Test that fetch_eeg_recordings raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import fetch_eeg_recordings

        source = MagicMock()
        with pytest.raises(ImportError, match="fetch_eeg_recordings requires mne-bids"):
            fetch_eeg_recordings(source)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_fetch_ieeg_recordings_raises_import_error(self):
        """Test that fetch_ieeg_recordings raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import fetch_ieeg_recordings

        source = MagicMock()
        with pytest.raises(
            ImportError, match="fetch_ieeg_recordings requires mne-bids"
        ):
            fetch_ieeg_recordings(source)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_group_recordings_by_entity_raises_import_error(self):
        """Test that group_recordings_by_entity raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import group_recordings_by_entity

        recordings = MagicMock()
        with pytest.raises(
            ImportError, match="group_recordings_by_entity requires mne-bids"
        ):
            group_recordings_by_entity(recordings)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_check_eeg_recording_files_exist_raises_import_error(self):
        """Test that check_eeg_recording_files_exist raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import check_eeg_recording_files_exist

        bids_root = MagicMock()
        recording_id = MagicMock()
        with pytest.raises(
            ImportError, match="check_eeg_recording_files_exist requires mne-bids"
        ):
            check_eeg_recording_files_exist(bids_root, recording_id)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_check_ieeg_recording_files_exist_raises_import_error(self):
        """Test that check_ieeg_recording_files_exist raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import check_ieeg_recording_files_exist

        bids_root = MagicMock()
        recording_id = MagicMock()
        with pytest.raises(
            ImportError, match="check_ieeg_recording_files_exist requires mne-bids"
        ):
            check_ieeg_recording_files_exist(bids_root, recording_id)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_load_json_sidecar_raises_import_error(self):
        """Test that load_json_sidecar raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import load_json_sidecar

        bids_path = MagicMock()
        with pytest.raises(ImportError, match="load_json_sidecar requires mne-bids"):
            load_json_sidecar(bids_path)

    @patch("brainsets.utils.bids_utils.MNE_BIDS_AVAILABLE", False)
    def test_build_bids_path_raises_import_error(self):
        """Test that build_bids_path raises ImportError when MNE_BIDS is unavailable."""
        from brainsets.utils.bids_utils import build_bids_path

        bids_root = MagicMock()
        recording_id = MagicMock()
        modality = MagicMock()
        with pytest.raises(ImportError, match="build_bids_path requires mne-bids"):
            build_bids_path(bids_root, recording_id, modality)
