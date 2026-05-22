"""Unit tests for OpenNeuro S3 utility functions."""

import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd

from brainsets.utils.openneuro.openneuro_s3 import BOTO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BOTO_AVAILABLE, reason="boto3/botocore not installed"
)

from brainsets.utils.openneuro.openneuro_s3 import (
    ClientError,
    fetch_latest_snapshot_tag,
    fetch_species,
    fetch_all_filenames,
    fetch_participants_tsv,
    construct_s3_url_from_path,
    download_recording,
    download_dataset_description,
    _graphql_query_openneuro,
    OPENNEURO_S3_BUCKET,
)


# ============================================================================
# Shared Fixtures and Helpers
# ============================================================================


@pytest.fixture
def mock_s3_client():
    """Reusable mock S3 client for all S3-related tests."""
    return MagicMock()


def make_client_error(code: str, message: str = "") -> ClientError:
    """Helper to create consistent ClientError instances."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "GetObject",
    )


@pytest.fixture
def participants_tsv_bytes():
    """Helper fixture to generate valid participants.tsv content."""

    def _make_tsv(has_participant_id: bool = True) -> bytes:
        if has_participant_id:
            content = (
                "participant_id\tage\tsex\nparticipant_01\t25\tM\nparticipant_02\t30\tF"
            )
        else:
            content = "age\tsex\n25\tM\n30\tF"
        return content.encode("utf-8")

    return _make_tsv


def graphql_ok_response(tag: str) -> dict:
    """Helper to generate valid GraphQL response structure."""
    return {
        "data": {
            "dataset": {
                "latestSnapshot": {
                    "tag": tag,
                }
            }
        }
    }


def graphql_species_response(species: str) -> dict:
    """Helper to generate valid GraphQL species response structure."""
    return {
        "data": {
            "dataset": {
                "metadata": {
                    "species": species,
                }
            }
        }
    }


# ============================================================================
# Tests for fetch_latest_snapshot_tag
# ============================================================================


class TestFetchLatestSnapshotTag:
    """Tests for latest snapshot tag resolution via GraphQL."""

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_returns_latest_snapshot_tag(self, mock_graphql):
        """Returns tag value from latestSnapshot in GraphQL response."""
        mock_graphql.return_value = graphql_ok_response("1.2.3")

        result = fetch_latest_snapshot_tag("ds005085")

        assert result == "1.2.3"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_queries_with_correct_dataset_id_variable(self, mock_graphql):
        """GraphQL query receives the expected datasetId variable."""
        mock_graphql.return_value = graphql_ok_response("1.2.3")

        fetch_latest_snapshot_tag("ds005085")

        call_args = mock_graphql.call_args
        assert call_args[0][1]["datasetId"] == "ds005085"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_raises_when_latest_snapshot_tag_missing(self, mock_graphql):
        """Raises RuntimeError when latestSnapshot.tag is missing."""
        mock_graphql.return_value = {"data": {"dataset": {"latestSnapshot": {}}}}

        with pytest.raises(
            RuntimeError,
            match="Could not resolve latest snapshot tag for dataset 'ds005085'",
        ):
            fetch_latest_snapshot_tag("ds005085")


# ============================================================================
# Tests for fetch_all_filenames
# ============================================================================


class TestFetchAllFilenames:
    """Tests for fetching all filenames from a dataset."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_returns_filenames_when_non_empty(self, mock_get_object_list):
        """Returns list of filenames when dataset has files."""
        filenames = [
            "sub-01/ieeg/sub-01_task-VisualNaming_ieeg.edf",
            "sub-01/ieeg/sub-01_task-VisualNaming_channels.tsv",
            "sub-02/ieeg/sub-02_ses-01_task-Rest_acq-ecog_run-01_ieeg.vhdr",
            "sub-02/ieeg/sub-02_ses-01_task-Rest_acq-ecog_run-01_ieeg.vmrk",
            "participants.tsv",
        ]
        mock_get_object_list.return_value = filenames

        result = fetch_all_filenames("ds005085")

        assert result == filenames
        mock_get_object_list.assert_called_once_with(OPENNEURO_S3_BUCKET, "ds005085/")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_raises_runtime_error_on_empty_dataset(self, mock_get_object_list):
        """RuntimeError is raised when no files are found."""
        mock_get_object_list.return_value = []

        with pytest.raises(RuntimeError, match="No files found"):
            fetch_all_filenames("ds005085")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_raises_runtime_error_with_dataset_id_in_message(
        self, mock_get_object_list
    ):
        """RuntimeError message includes the dataset ID."""
        mock_get_object_list.return_value = []

        with pytest.raises(RuntimeError, match="ds005085"):
            fetch_all_filenames("ds005085")


# ============================================================================
# Tests for fetch_participants_tsv
# ============================================================================


class TestFetchParticipantsTsv:
    """Tests for fetching and parsing participants.tsv."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_indexed_dataframe_with_participant_id(
        self, mock_get_client, participants_tsv_bytes
    ):
        """Returns DataFrame indexed by participant_id when column exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=lambda: participants_tsv_bytes(has_participant_id=True)
            )
        }

        result = fetch_participants_tsv("ds005085")

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "participant_id"
        assert len(result) == 2
        assert list(result.index) == ["participant_01", "participant_02"]
        assert "age" in result.columns
        assert "sex" in result.columns

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_when_participant_id_column_missing(
        self, mock_get_client, participants_tsv_bytes, caplog
    ):
        """Returns None with warning when participant_id column is absent."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=lambda: participants_tsv_bytes(has_participant_id=False)
            )
        }

        result = fetch_participants_tsv("ds005085")

        assert result is None
        assert "No participant_id column found" in caplog.text

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_on_no_such_key_error(self, mock_get_client):
        """Returns None when participants.tsv does not exist (NoSuchKey)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("NoSuchKey")

        result = fetch_participants_tsv("ds005085")

        assert result is None

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_on_404_error(self, mock_get_client):
        """Returns None when participants.tsv returns 404."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("404")

        result = fetch_participants_tsv("ds005085")

        assert result is None

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_reraises_other_client_errors(self, mock_get_client):
        """Other ClientErrors are re-raised."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error(
            "AccessDenied", "Access Denied"
        )

        with pytest.raises(ClientError):
            fetch_participants_tsv("ds005085")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_parses_tsv_with_na_values(self, mock_get_client):
        """TSV parser respects na_values configuration."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        tsv_content = b"participant_id\tage\npart_01\tn/a\npart_02\tN/A"
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: tsv_content)
        }

        result = fetch_participants_tsv("ds005085")

        assert result is not None
        assert pd.isna(result.loc["part_01", "age"])
        assert pd.isna(result.loc["part_02", "age"])


# ============================================================================
# Tests for construct_s3_url_from_path and download_recording
# ============================================================================


class TestConstructS3UrlFromPath:
    """Tests for S3 URL construction."""

    def test_constructs_url_with_subdirectory(self):
        """URL is correctly constructed from dataset, path, and recording ID."""
        url = construct_s3_url_from_path(
            dataset_id="ds004019",
            data_file_path="sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1_eeg.edf",
            recording_id="sub-01_ses-01_task-nap_run-1",
        )

        assert (
            url
            == "s3://openneuro.org/ds004019/sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1"
        )

    def test_handles_deeply_nested_path(self):
        """Deeply nested file paths are handled correctly."""
        url = construct_s3_url_from_path(
            dataset_id="ds000001",
            data_file_path="derivatives/sub-A/ses-01/func/deep_file.nii.gz",
            recording_id="recording_x",
        )

        assert "derivatives/sub-A/ses-01/func" in url
        assert url.endswith("recording_x")
        assert (
            url
            == "s3://openneuro.org/ds000001/derivatives/sub-A/ses-01/func/recording_x"
        )


class TestDownloadRecording:
    """Tests for recording download delegation."""

    @patch("brainsets.utils.openneuro.openneuro_s3.download_prefix_from_url")
    def test_delegates_to_download_prefix_from_url(self, mock_download):
        """download_recording delegates to download_prefix_from_url."""
        s3_url = "s3://openneuro.org/ds005085/sub-01/eeg/file"
        target_dir = Path("/tmp/data")
        expected_files = [Path("/tmp/data/file1.edf")]
        mock_download.return_value = expected_files

        result = download_recording(s3_url, target_dir)

        mock_download.assert_called_once_with(s3_url, target_dir)
        assert result == expected_files


# ============================================================================
# Tests for download_dataset_description
# ============================================================================


class TestDownloadDatasetDescription:
    """Tests for dataset_description.json download and caching."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_short_circuits_when_file_exists(self, mock_get_client, tmp_path):
        """Returns existing file without downloading if already present."""
        target_file = tmp_path / "dataset_description.json"
        target_file.write_text('{"name": "existing"}')

        result = download_dataset_description("ds005085", tmp_path)

        assert result == target_file
        mock_get_client.assert_not_called()

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_downloads_and_writes_file_on_success(self, mock_get_client, tmp_path):
        """Downloads and writes file when not already present."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b'{"name": "test"}')
        }

        result = download_dataset_description("ds005085", tmp_path)

        assert result == tmp_path / "dataset_description.json"
        assert result.exists()
        assert result.read_bytes() == b'{"name": "test"}'

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_creates_parent_directories(self, mock_get_client, tmp_path):
        """Creates parent directories if they don't exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"{}")}
        nested_target = tmp_path / "a" / "b" / "c"

        result = download_dataset_description("ds005085", nested_target)

        assert result.parent.exists()
        assert result.exists()

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_no_such_key(self, mock_get_client, tmp_path):
        """RuntimeError is raised when file doesn't exist on S3 (NoSuchKey)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("NoSuchKey")

        with pytest.raises(RuntimeError, match="not found"):
            download_dataset_description("ds005085", tmp_path)

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_404(self, mock_get_client, tmp_path):
        """RuntimeError is raised on 404 error."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("404")

        with pytest.raises(RuntimeError, match="not found"):
            download_dataset_description("ds005085", tmp_path)

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_other_client_error(
        self, mock_get_client, tmp_path
    ):
        """RuntimeError is raised for other ClientErrors."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error(
            "AccessDenied", "Access Denied"
        )

        with pytest.raises(RuntimeError, match="Failed to download"):
            download_dataset_description("ds005085", tmp_path)


# ============================================================================
# Tests for fetch_species
# ============================================================================


class TestFetchSpecies:
    """Tests for OpenNeuro species fetching via GraphQL."""

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_returns_homo_sapiens_for_human_alias(self, mock_graphql):
        """Normalizes human aliases to canonical homo sapiens."""
        mock_graphql.return_value = graphql_species_response("homo sapiens")

        result = fetch_species("ds005085")

        assert result == "homo sapiens"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_returns_raw_non_human_species(self, mock_graphql):
        """Returns species value as provided by GraphQL."""
        mock_graphql.return_value = graphql_species_response("mus musculus")

        result = fetch_species("ds005085")

        assert result == "mus musculus"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_queries_with_correct_variables(self, mock_graphql):
        """GraphQL is queried with the correct dataset ID."""
        mock_graphql.return_value = graphql_species_response("mus musculus")

        fetch_species("ds005085")

        call_args = mock_graphql.call_args
        assert call_args[0][1]["datasetId"] == "ds005085"


# ============================================================================
# Tests for _graphql_query_openneuro
# ============================================================================


class TestGraphqlQueryOpenneuro:
    """Tests for GraphQL query execution and retry logic."""

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_succeeds_on_200_status(self, mock_post):
        """Returns response JSON on 200 status code."""
        expected_response = {"data": {"test": "value"}}
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: expected_response
        )

        result = _graphql_query_openneuro("query { test }", {})

        assert result == expected_response

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_raises_on_non_200_status(self, mock_post, mock_sleep):
        """Raises exception on non-200 status code."""
        mock_post.return_value = MagicMock(status_code=500)

        with pytest.raises(Exception, match="Query failed"):
            _graphql_query_openneuro("query { test }", {})

        # Non-200 responses are retried with the standard backoff schedule.
        assert mock_sleep.call_count == 4

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_retries_on_transient_failure_then_succeeds(self, mock_post, mock_sleep):
        """Retries on transient failure and succeeds on retry."""
        expected_response = {"data": {"test": "value"}}
        mock_post.side_effect = [
            Exception("Network error"),
            MagicMock(status_code=200, json=lambda: expected_response),
        ]

        result = _graphql_query_openneuro("query { test }", {})

        assert result == expected_response
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_raises_after_max_attempts(self, mock_post, mock_sleep):
        """Raises exception after exhausting retry attempts."""
        mock_post.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            _graphql_query_openneuro("query { test }", {})

        # 5 attempts total (initial + 4 retries)
        assert mock_post.call_count == 5
        # 4 sleeps between retries
        assert mock_sleep.call_count == 4

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_exponential_backoff_timing(self, mock_post, mock_sleep):
        """Retry delays follow exponential backoff pattern."""
        mock_post.side_effect = Exception("Network error")

        try:
            _graphql_query_openneuro("query { test }", {})
        except Exception:
            pass

        # Sleep calls should follow exponential backoff: 4, 8, 10, 10
        # (capped at max_wait=10)
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert sleep_calls == [4, 8, 10, 10]

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_passes_query_and_variables_correctly(self, mock_post):
        """Query and variables are passed correctly to requests.post."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
        query = "query { test }"
        variables = {"id": "123"}

        _graphql_query_openneuro(query, variables)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["query"] == query
        assert call_kwargs["json"]["variables"] == variables

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_handles_none_variables(self, mock_post):
        """None variables are handled correctly."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})

        _graphql_query_openneuro("query { test }", None)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["variables"] is None
