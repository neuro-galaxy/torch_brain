"""Unit tests for OpenNeuro Pipeline classes."""

import pytest
import numpy as np
import pandas as pd
import mne

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, PropertyMock
from argparse import Namespace

from temporaldata import Data, Interval

from brainsets.utils.openneuro.pipeline import (
    OpenNeuroPipeline,
)

from brainsets.taxonomy import Species

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_args_no_reprocessing():
    """Mock args with redownload and reprocess set to False."""
    return Namespace(redownload=False, reprocess=False)


@pytest.fixture
def mock_args_with_reprocessing():
    """Mock args with redownload and reprocess set to True."""
    return Namespace(redownload=True, reprocess=True)


@pytest.fixture
def manifest_row():
    """Mock manifest row with typical structure."""
    row = MagicMock()
    row.Index = "rec-001"
    row.subject_id = "sub-01"
    row.s3_url = "s3://openneuro.org/ds005085/sub-01/eeg/rec-001"
    row.species = "homo sapiens"
    row.age = 25
    row.sex = "M"
    row.latest_snapshot_tag = "1.0.0"
    return row


@pytest.fixture
def participants_df():
    """Mock participants DataFrame."""
    return pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-02"],
            "age": [25, 30],
            "sex": ["M", "F"],
        }
    ).set_index("participant_id")


@pytest.fixture
def mock_raw():
    """Mock MNE raw object."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.info = {"sfreq": 250.0, "meas_date": None, "bads": ["EEG_BAD"]}
    raw.get_data.return_value = np.random.randn(10, 1000)
    raw.ch_names = [
        "EEG_1",
        "EEG_2",
        "EEG_3",
        "EEG_IGNORE",
        "EEG_BAD",
        "EOG_L",
        "EOG_R",
        "EMG",
        "STIM",
    ] + ["unused"] * 3
    raw.get_channel_types.return_value = ["eeg"] * len(raw.ch_names)
    raw.times = np.linspace(0, 4, 1000)
    raw.get_montage.return_value = None
    return raw


@pytest.fixture
def eeg_pipeline_class():
    """Concrete EEG pipeline class for testing."""

    class TestEEGPipeline(OpenNeuroPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_eeg_brainset"
        origin_version = "1.0.0"
        derived_version = "1.0.0"
        modality = "eeg"

    return TestEEGPipeline


@pytest.fixture
def ieeg_pipeline_class():
    """Concrete iEEG pipeline class for testing."""

    class TestIEEGPipeline(OpenNeuroPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_ieeg_brainset"
        origin_version = "1.0.0"
        derived_version = "1.0.0"
        modality = "ieeg"

    return TestIEEGPipeline


@pytest.fixture
def eeg_pipeline_instance(eeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated EEG pipeline."""
    instance = eeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


@pytest.fixture
def ieeg_pipeline_instance(ieeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated iEEG pipeline."""
    instance = ieeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


# ============================================================================
# Tests for validate_dataset_id
# ============================================================================


class TestValidateDatasetId:
    """Tests for OpenNeuroPipeline.validate_dataset_id helper method."""

    def test_valid_strict_format_passes(self):
        """Valid strict format (ds + 6 digits) does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds005085")

    def test_another_valid_strict_format_passes(self):
        """Another valid strict format with different digits does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds000001")

    def test_max_valid_numeric_value_passes(self):
        """Maximum valid numeric value (009999) does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds009999")

    def test_uppercase_prefix_raises_error(self):
        """Uppercase 'DS' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("DS005085")

    def test_whitespace_around_input_raises_error(self):
        """Whitespace around input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("  ds005085  ")

    def test_missing_prefix_raises_error(self):
        """Missing 'ds' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("005085")

    def test_non_numeric_suffix_raises_error(self):
        """Non-numeric suffix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds00a00")

    def test_too_few_digits_raises_error(self):
        """Too few digits after 'ds' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds5085")

    def test_too_many_digits_raises_error(self):
        """Too many digits after 'ds' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds0050850")

    def test_numeric_part_exceeds_range_raises_error(self):
        """Numeric part exceeding 9999 raises ValueError."""
        with pytest.raises(ValueError, match="invalid numeric portion"):
            OpenNeuroPipeline.validate_dataset_id("ds010000")

    def test_numeric_part_below_minimum_raises_error(self):
        """Numeric part of 000000 raises ValueError."""
        with pytest.raises(ValueError, match="invalid numeric portion"):
            OpenNeuroPipeline.validate_dataset_id("ds000000")

    def test_invalid_format_no_prefix_raises_error(self):
        """Invalid format without 'ds' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("invalid")

    def test_empty_string_raises_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("   ")


class TestValidateDatasetVersion:
    """Tests for OpenNeuroPipeline._validate_dataset_version helper method."""

    class _VersionTestPipeline(OpenNeuroPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_eeg_brainset"
        origin_version = "1.0.0"
        derived_version = "1.0.0"
        modality = "eeg"

    def test_returns_when_versions_match(self):
        """Returns cleanly when latest tag matches origin version."""
        result = self._VersionTestPipeline._validate_dataset_version("1.0.0")
        assert result is None

    def test_warns_when_versions_differ_with_continue(self, caplog):
        """Logs warning when version differs and policy is 'continue'."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="continue"
        )

        assert result is None
        assert "version '1.0.0' was used to create the brainset pipeline" in caplog.text
        assert "but the latest available version on OpenNeuro is '2.0.0'" in caplog.text

    def test_on_mismatch_abort_raises_error(self):
        """on_mismatch='abort' exits cleanly when versions differ."""
        with pytest.raises(
            SystemExit, match="Aborting pipeline due to dataset version mismatch"
        ):
            self._VersionTestPipeline._validate_dataset_version(
                "2.0.0", on_mismatch="abort"
            )

    def test_on_mismatch_continue_warns_and_returns(self, caplog):
        """on_mismatch='continue' logs warning and returns latest version."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="continue"
        )

        assert result is None
        assert "version '1.0.0' was used to create the brainset pipeline" in caplog.text

    @patch("builtins.input", return_value="y")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_interactive_accept_continues(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' with interactive TTY and 'y' continues."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="prompt"
        )

        assert result is None
        mock_input.assert_called_once()
        assert "Continue with latest version?" in mock_input.call_args[0][0]

    @patch("builtins.input", return_value="n")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_interactive_reject_aborts(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' with interactive TTY and 'n' aborts."""
        with pytest.raises(
            SystemExit, match="Aborted by user due to dataset version mismatch"
        ):
            self._VersionTestPipeline._validate_dataset_version(
                "2.0.0", on_mismatch="prompt"
            )

        mock_input.assert_called_once()

    @patch("builtins.input")
    @patch("sys.stdin.isatty")
    def test_on_mismatch_prompt_no_prompt_when_versions_match(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' does not prompt or check TTY when versions match."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "1.0.0", on_mismatch="prompt"
        )

        assert result is None
        mock_isatty.assert_not_called()
        mock_input.assert_not_called()

    @patch("builtins.input", return_value="yes")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_accepts_yes_variant(self, mock_isatty, mock_input):
        """on_mismatch='prompt' accepts 'yes' as valid confirmation."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="prompt"
        )

        assert result is None


class TestValidateOnMismatchPolicy:
    """Tests for OpenNeuroPipeline._validate_on_mismatch_policy helper method."""

    @patch("sys.stdin.isatty", return_value=True)
    def test_prompt_in_interactive_mode_succeeds(self, mock_isatty):
        """_validate_on_mismatch_policy succeeds with 'prompt' in interactive mode."""
        result = OpenNeuroPipeline._validate_on_mismatch_policy("prompt")
        assert result is None
        mock_isatty.assert_called_once()

    @patch("sys.stdin.isatty", return_value=False)
    def test_prompt_in_non_interactive_mode_raises_valueerror(self, mock_isatty):
        """_validate_on_mismatch_policy raises ValueError with 'prompt' in non-interactive mode."""
        with pytest.raises(
            ValueError, match="Cannot use --on-version-mismatch='prompt'"
        ):
            OpenNeuroPipeline._validate_on_mismatch_policy("prompt")

        with pytest.raises(ValueError, match="non-interactive mode"):
            OpenNeuroPipeline._validate_on_mismatch_policy("prompt")

        with pytest.raises(ValueError, match="continue"):
            OpenNeuroPipeline._validate_on_mismatch_policy("prompt")

    @patch("sys.stdin.isatty", return_value=False)
    def test_continue_in_non_interactive_mode_succeeds(self, mock_isatty):
        """_validate_on_mismatch_policy succeeds with 'continue' in non-interactive mode."""
        result = OpenNeuroPipeline._validate_on_mismatch_policy("continue")
        assert result is None
        mock_isatty.assert_not_called()

    @patch("sys.stdin.isatty", return_value=False)
    def test_abort_in_non_interactive_mode_succeeds(self, mock_isatty):
        """_validate_on_mismatch_policy succeeds with 'abort' in non-interactive mode."""
        result = OpenNeuroPipeline._validate_on_mismatch_policy("abort")
        assert result is None
        mock_isatty.assert_not_called()

    @patch("sys.stdin.isatty", return_value=True)
    def test_continue_in_interactive_mode_succeeds(self, mock_isatty):
        """_validate_on_mismatch_policy succeeds with 'continue' in interactive mode."""
        result = OpenNeuroPipeline._validate_on_mismatch_policy("continue")
        assert result is None
        mock_isatty.assert_not_called()

    @patch("sys.stdin.isatty", return_value=True)
    def test_abort_in_interactive_mode_succeeds(self, mock_isatty):
        """_validate_on_mismatch_policy succeeds with 'abort' in interactive mode."""
        result = OpenNeuroPipeline._validate_on_mismatch_policy("abort")
        assert result is None
        mock_isatty.assert_not_called()


class TestGetManifestPolicyValidation:
    """Tests for get_manifest policy validation."""

    @patch("sys.stdin.isatty", return_value=False)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    def test_get_manifest_raises_valueerror_on_prompt_non_interactive(
        self, mock_fetch_files, mock_isatty, eeg_pipeline_class, temp_dir
    ):
        """get_manifest raises ValueError when on_version_mismatch='prompt' in non-interactive mode."""
        from argparse import Namespace

        args = Namespace(
            on_version_mismatch="prompt", redownload=False, reprocess=False
        )
        mock_fetch_files.return_value = []

        with pytest.raises(
            ValueError, match="Cannot use --on-version-mismatch='prompt'"
        ):
            eeg_pipeline_class.get_manifest(temp_dir, args)

    @patch("sys.stdin.isatty", return_value=False)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_latest_snapshot_tag")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_manifest_continues_on_continue_non_interactive(
        self,
        mock_part,
        mock_species,
        mock_latest_tag,
        mock_ver,
        mock_id,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_isatty,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest succeeds when on_version_mismatch='continue' in non-interactive mode."""
        from argparse import Namespace

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        mock_fetch_files.return_value = []
        mock_fetch_eeg.return_value = []
        mock_latest_tag.return_value = "1.0.0"
        mock_species.return_value = "homo sapiens"
        mock_part.return_value = None

        try:
            eeg_pipeline_class.get_manifest(temp_dir, args)
        except ValueError:
            pass

    @patch("sys.stdin.isatty", return_value=False)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_latest_snapshot_tag")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_manifest_continues_on_abort_non_interactive(
        self,
        mock_part,
        mock_species,
        mock_latest_tag,
        mock_ver,
        mock_id,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_isatty,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest succeeds when on_version_mismatch='abort' in non-interactive mode."""
        from argparse import Namespace

        args = Namespace(on_version_mismatch="abort", redownload=False, reprocess=False)
        mock_fetch_files.return_value = []
        mock_fetch_eeg.return_value = []
        mock_latest_tag.return_value = "1.0.0"
        mock_species.return_value = "homo sapiens"
        mock_part.return_value = None

        try:
            eeg_pipeline_class.get_manifest(temp_dir, args)
        except ValueError:
            pass


class TestNormalizeSpecies:
    """Tests for OpenNeuroPipeline._normalize_species helper method."""

    @pytest.mark.parametrize(
        "species",
        [
            "homo",
            "homo sapiens",
            "human",
            "humans",
            "H. sapiens",
            "  HUMAN  ",
            "h. sapiens",
        ],
    )
    def test_returns_homo_sapiens_for_supported_aliases(self, species):
        """Supported aliases normalize to canonical homo sapiens."""
        assert OpenNeuroPipeline._normalize_species(species) == "homo sapiens"

    @pytest.mark.parametrize("species", ["mus musculus", "canis lupus", "", None, 42])
    def test_returns_unknown_for_non_human_or_invalid_values(self, species):
        """Non-human or invalid values normalize to unknown."""
        assert OpenNeuroPipeline._normalize_species(species) == "unknown"


class TestGetManifest:
    """Tests for get_manifest class method."""

    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_latest_snapshot_tag")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    @patch("brainsets.utils.openneuro.pipeline.construct_s3_url_from_path")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_eeg_success(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_s3,
        mock_subject_info,
        mock_participants_tsv,
        mock_species,
        mock_latest_tag,
        mock_ver,
        mock_id,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for EEG dataset."""
        mock_fetch_files.return_value = [
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "sub-01/eeg/sub-01_task-math_eeg.edf",
            "sub-02/eeg/sub-02_task-rest_eeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/eeg/sub-01_task-math_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/eeg/sub-02_task-rest_eeg.edf",
            },
        ]

        def s3_url_side_effect(dataset_id, fpath, recording_id):
            parent_dir = str(Path(fpath).parent)
            return f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"

        mock_s3.side_effect = s3_url_side_effect
        mock_latest_tag.return_value = "1.0.0"
        mock_species.return_value = "homo sapiens"
        mock_participants_tsv.return_value = None
        mock_subject_info.return_value = {"age": 25, "sex": "M"}

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        result = eeg_pipeline_class.get_manifest(temp_dir, args)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/eeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/eeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/eeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            assert result.loc[recording_id, "species"] == "homo sapiens"
            assert result.loc[recording_id, "age"] == 25
            assert result.loc[recording_id, "sex"] == "M"
            assert result.loc[recording_id, "latest_snapshot_tag"] == "1.0.0"
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_latest_snapshot_tag")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    @patch("brainsets.utils.openneuro.pipeline.construct_s3_url_from_path")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_ieeg_recordings")
    def test_get_manifest_ieeg_success(
        self,
        mock_fetch_ieeg,
        mock_fetch_files,
        mock_s3,
        mock_subject_info,
        mock_participants_tsv,
        mock_species,
        mock_latest_tag,
        mock_ver,
        mock_id,
        ieeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for iEEG dataset."""
        mock_fetch_files.return_value = [
            "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_ieeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            },
        ]

        def s3_url_side_effect(dataset_id, fpath, recording_id):
            parent_dir = str(Path(fpath).parent)
            return f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"

        mock_s3.side_effect = s3_url_side_effect
        mock_latest_tag.return_value = "1.0.0"
        mock_species.return_value = "homo sapiens"
        mock_participants_tsv.return_value = None
        mock_subject_info.return_value = {"age": 25, "sex": "M"}

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        result = ieeg_pipeline_class.get_manifest(temp_dir, args)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/ieeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/ieeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/ieeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            assert result.loc[recording_id, "species"] == "homo sapiens"
            assert result.loc[recording_id, "age"] == 25
            assert result.loc[recording_id, "sex"] == "M"
            assert result.loc[recording_id, "latest_snapshot_tag"] == "1.0.0"
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch("sys.stdin.isatty", return_value=True)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_with_custom_openneuro_context(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_isatty,
        temp_dir,
    ):
        """get_manifest works when _openneuro_context is pre-populated."""

        class CustomContextPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "eeg"

        mock_fetch_files.return_value = [
            "sub-01/eeg/rec-001_eeg.edf",
            "sub-02/eeg/rec-001_eeg.edf",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "rec-001",
                "fpath": "sub-01/eeg/rec-001_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "rec-001",
                "fpath": "sub-02/eeg/rec-001_eeg.edf",
            },
        ]

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        result = CustomContextPipeline.get_manifest(temp_dir, args)

        assert len(result) == 2
        assert "rec-001" in result.index

    @patch("sys.stdin.isatty", return_value=True)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    def test_get_manifest_raises_on_unknown_modality(
        self, mock_fetch_files, mock_isatty, temp_dir
    ):
        """get_manifest raises ValueError for unknown modality."""

        class BadPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "unknown"

        mock_fetch_files.return_value = []

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        with pytest.raises(ValueError, match="Unknown modality"):
            BadPipeline.get_manifest(temp_dir, args)

    @patch("sys.stdin.isatty", return_value=True)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_recordings_found(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_isatty,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest raises ValueError when no recordings are found."""
        mock_fetch_files.return_value = []
        mock_fetch_eeg.return_value = []

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        with pytest.raises(ValueError, match="No EEG recordings found"):
            eeg_pipeline_class.get_manifest(temp_dir, args)

    @patch("sys.stdin.isatty", return_value=True)
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_recordings_returned_by_fetch(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_isatty,
        temp_dir,
    ):
        """get_manifest raises ValueError when recording parser returns no rows."""

        class EmptyRecordingsPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "eeg"

        mock_fetch_files.return_value = ["sub-01/eeg/rec-001.edf"]
        mock_fetch_eeg.return_value = []

        args = Namespace(
            on_version_mismatch="continue", redownload=False, reprocess=False
        )
        with pytest.raises(ValueError, match="No EEG recordings found"):
            EmptyRecordingsPipeline.get_manifest(temp_dir, args)


# ============================================================================
# Tests for download
# ============================================================================


class TestDownload:
    """Tests for download method."""

    def test_download_creates_raw_dir(self, eeg_pipeline_instance, manifest_row):
        """download creates raw_dir if it doesn't exist."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    eeg_pipeline_instance.download(manifest_row)

        assert eeg_pipeline_instance.raw_dir.exists()

    def test_download_returns_dict_with_required_keys(
        self, eeg_pipeline_instance, manifest_row
    ):
        """download returns manifest_item with subject_id and recording_id."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    result = eeg_pipeline_instance.download(manifest_row)

        assert hasattr(result, "subject_id")
        assert hasattr(result, "Index")
        assert result.subject_id == "sub-01"
        assert result.Index == "rec-001"

    def test_download_eeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download skips if files exist and redownload is False."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result.Index == "rec-001"

    def test_download_ieeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, ieeg_pipeline_class, manifest_row
    ):
        """download skips for iEEG if files exist and redownload is False."""
        pipeline = ieeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_ieeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result.Index == "rec-001"

    def test_download_eeg_redownloads_when_redownload_true(
        self, temp_dir, mock_args_with_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download redownloads when redownload=True even if files exist."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_with_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording"
        ) as mock_download:
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                result = pipeline.download(manifest_row)

        mock_download.assert_called_once()

    def test_download_raises_on_s3_error(self, eeg_pipeline_instance, manifest_row):
        """download raises RuntimeError on download failure."""
        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording",
            side_effect=Exception("S3 error"),
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                return_value=False,
            ):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    eeg_pipeline_instance.download(manifest_row)


# ============================================================================
# Tests for channel remapping methods
# ============================================================================


class TestChannelRemapping:
    """Tests for channel remapping methods."""

    def test_get_channel_name_remapping_returns_class_attribute(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns CHANNEL_NAME_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "eeg"
            CHANNEL_NAME_REMAPPING = {"PSG_F3": "F3", "PSG_F4": "F4"}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_channel_name_remapping() == {"PSG_F3": "F3", "PSG_F4": "F4"}

    def test_get_channel_name_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_channel_name_remapping() is None

    def test_get_channel_name_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_channel_name_remapping(
            recording_id="rec-001"
        )
        assert result is None

    def test_get_type_channels_remapping_returns_class_attribute(self):
        """get_type_channels_remapping returns TYPE_CHANNELS_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "eeg"
            TYPE_CHANNELS_REMAPPING = {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_type_channels_remapping() == {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG"],
        }

    def test_get_type_channels_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_type_channels_remapping() is None

    def test_get_type_channels_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_type_channels_remapping(
            recording_id="rec-001"
        )
        assert result is None


# ============================================================================
# Tests for process_common
# ============================================================================


class TestProcessCommon:
    """Tests for process_common method."""

    @patch("brainsets.utils.openneuro.pipeline.MNE_BIDS_AVAILABLE", False)
    def test_process_common_raises_when_mne_bids_unavailable(
        self, eeg_pipeline_instance
    ):
        """process_common raises ImportError if mne-bids not available."""
        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"
        download_output.species = "homo sapiens"
        download_output.age = 25
        download_output.sex = "M"
        download_output.latest_snapshot_tag = "1.0.0"

        with pytest.raises(ImportError, match="mne-bids"):
            eeg_pipeline_instance.process_common(download_output)

    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    def test_process_common_skips_if_already_processed(
        self, mock_read_raw, eeg_pipeline_instance
    ):
        """process_common returns None if file already exists and not reprocessing."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        (eeg_pipeline_instance.processed_dir / "rec-001.h5").touch()
        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"
        download_output.species = "homo sapiens"
        download_output.age = 25
        download_output.sex = "M"
        download_output.latest_snapshot_tag = "1.0.0"

        result = eeg_pipeline_instance.process_common(download_output)

        assert result is None
        mock_read_raw.assert_not_called()

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    @patch("brainsets.utils.openneuro.pipeline.extract_channels")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    def test_process_common_creates_data_object(
        self,
        mock_meas_date,
        mock_extract_channels,
        mock_extract_signal,
        mock_read_raw,
        mock_bids_path,
        eeg_pipeline_instance,
        mock_raw,
    ):
        """process_common creates and returns Data object."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"
        download_output.species = "homo sapiens"
        download_output.age = 25
        download_output.sex = "M"
        download_output.latest_snapshot_tag = "1.0.0"

        mock_bids_path.return_value = "path/to/bids"
        mock_read_raw.return_value = mock_raw
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )
        mock_extract_channels.return_value = {}

        result = eeg_pipeline_instance.process_common(download_output)

        assert result is not None
        assert len(result) == 2
        data, store_path = result
        assert isinstance(data, Data)
        assert isinstance(store_path, Path)

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    @patch("brainsets.utils.openneuro.pipeline.extract_channels")
    def test_channel_and_type_remapping_and_ignore_channels(
        self,
        mock_extract_channels,
        mock_extract_signal,
        mock_meas_date,
        mock_read_raw,
        mock_bids_path,
        temp_dir,
        mock_args_no_reprocessing,
        mock_raw,
    ):
        """Test that process_common remaps channel names/types and applies ignore channels."""

        class CustomEEGPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            derived_version = "1.0.0"
            modality = "eeg"
            CHANNEL_NAME_REMAPPING = {"EEG_1": "F3", "EEG_4": "F4"}
            TYPE_CHANNELS_REMAPPING = {
                "EEG": ["F3", "EEG_2", "EEG_3", "F4"],
                "EOG": ["EOG_L", "EOG_R", "EOG"],
                "EMG": ["EMG"],
                "STIM": ["STIM"],
                "MISC": ["unused"],
            }
            IGNORE_CHANNELS = ["EEG_IGNORE"]

        pipeline = CustomEEGPipeline(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"
        download_output.species = "homo sapiens"
        download_output.age = 25
        download_output.sex = "M"
        download_output.latest_snapshot_tag = "1.0.0"

        mock_bids_path.return_value = "path/to/bids_path"
        mock_read_raw.return_value = mock_raw
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )
        channels_obj = MagicMock()
        channels_obj.id = np.array(
            ["F3", "EEG_2", "EEG_3", "EEG_BAD", "EOG_L", "EOG_R", "EMG", "STIM"]
            + ["unused"] * 3
        )
        channels_obj.type = np.array(
            [
                "eeg",
                "eeg",
                "eeg",
                "eeg",
                "eog",
                "eog",
                "emg",
                "stim",
                "misc",
                "misc",
                "misc",
            ]
        )
        channels_obj.bad = np.array(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        mock_extract_channels.return_value = channels_obj

        data, _ = pipeline.process_common(download_output)

        assert all(
            data.channels.id
            == np.array(
                ["F3", "EEG_2", "EEG_3", "EEG_BAD", "EOG_L", "EOG_R", "EMG", "STIM"]
                + ["unused"] * 3
            )
        )
        assert all(
            data.channels.type
            == np.array(
                [
                    "eeg",
                    "eeg",
                    "eeg",
                    "eeg",
                    "eog",
                    "eog",
                    "emg",
                    "stim",
                    "misc",
                    "misc",
                    "misc",
                ]
            )
        )
        assert all(
            data.channels.bad
            == np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            )
        )


# ============================================================================
# Tests for process
# ============================================================================


class TestProcess:
    """Tests for process method."""

    def test_process_skips_when_process_common_returns_none(
        self, eeg_pipeline_instance
    ):
        """process returns early if process_common returns None."""
        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"

        with patch.object(eeg_pipeline_instance, "process_common", return_value=None):
            with patch("builtins.open") as mock_file:
                eeg_pipeline_instance.process(download_output)

        mock_file.assert_not_called()

    def test_process_saves_data_to_h5(self, eeg_pipeline_instance):
        """process saves Data object to HDF5 file."""
        download_output = MagicMock()
        download_output.Index = "rec-001"
        download_output.subject_id = "sub-01"
        mock_data = MagicMock(spec=Data)
        store_path = eeg_pipeline_instance.processed_dir / "rec-001.h5"

        with patch.object(
            eeg_pipeline_instance,
            "process_common",
            return_value=(mock_data, store_path),
        ):
            with patch("brainsets.utils.openneuro.pipeline.h5py.File"):
                eeg_pipeline_instance.process(download_output)

        mock_data.to_hdf5.assert_called_once()
