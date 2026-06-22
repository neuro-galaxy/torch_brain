"""OpenNeuro dataset utilities.

This module provides functions for dataset validation, file listing,
and downloading from OpenNeuro's S3 bucket.
"""

__all__ = [
    "OPENNEURO_S3_BUCKET",
    "construct_s3_url_from_path",
    "download_dataset_description",
    "download_meta",
    "download_participants_tsv",
    "download_recording",
    "fetch_latest_snapshot_tag",
    "fetch_all_filenames",
    "fetch_participants_tsv",
    "fetch_species",
]

__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}

from pathlib import Path

import pandas as pd
import requests

try:
    from botocore.exceptions import ClientError

    BOTO_AVAILABLE = True
except ImportError:
    ClientError = Exception
    BOTO_AVAILABLE = False

from torch_brain.utils.bids import _parse_participants_tsv
from torch_brain.utils.s3 import (
    _is_not_found_error,
    download_object,
    download_prefix_from_url,
    get_cached_s3_client,
    get_object_list,
)

OPENNEURO_S3_BUCKET = "openneuro.org"
r"""S3 bucket URL for OpenNeuro"""

GRAPHQL_ENDPOINT = "https://openneuro.org/crn/graphql"


def fetch_latest_snapshot_tag(dataset_id: str) -> str:
    """Fetch the latest snapshot tag for an OpenNeuro dataset.

    Args:
        dataset_id: OpenNeuro dataset identifier (for example, ``"ds005555"``).

    Returns:
        Latest snapshot tag available on OpenNeuro for ``dataset_id``.

    Raises:
        RuntimeError: If the dataset cannot be resolved from the GraphQL response.
    """
    query = """
        query Dataset($datasetId: ID!) {
            dataset(id: $datasetId) {
                latestSnapshot {
                    tag
                }
            }
        }
    """

    variables = {
        "datasetId": dataset_id,
    }

    response = _graphql_query_openneuro(
        query,
        variables,
    )

    dataset = response.get("data", {}).get("dataset")
    latest_snapshot_tag = ((dataset or {}).get("latestSnapshot") or {}).get("tag")
    if not latest_snapshot_tag:
        raise RuntimeError(
            f"Could not resolve latest snapshot tag for dataset '{dataset_id}'. "
            "The dataset may be missing, private, or the API response format changed."
        )

    return latest_snapshot_tag


def fetch_all_filenames(dataset_id: str) -> list[str]:
    """Fetch all filenames for a given OpenNeuro dataset using AWS S3.

    Note:
        OpenNeuro S3 exposes only the latest dataset snapshot.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        List of relative filenames in the dataset (excluding directories)
    """
    prefix = f"{dataset_id}/"

    filenames = get_object_list(OPENNEURO_S3_BUCKET, prefix)

    if len(filenames) == 0:
        raise RuntimeError(
            f"No files found for dataset {dataset_id}. "
            "The dataset may not exist or may be empty."
        )

    return filenames


def fetch_participants_tsv(dataset_id: str) -> pd.DataFrame | None:
    """Fetch and parse participants.tsv from OpenNeuro S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier

    Returns:
        DataFrame indexed by ``participant_id``, or ``None`` if the file does not
        exist or has no ``participant_id`` column.
    """
    s3_client = get_cached_s3_client()
    key = f"{dataset_id}/participants.tsv"

    try:
        response = s3_client.get_object(Bucket=OPENNEURO_S3_BUCKET, Key=key)
        content = response["Body"].read()
    except ClientError as e:
        if _is_not_found_error(e):
            return None
        raise

    return _parse_participants_tsv(
        content,
        missing_column_context=f"file in OpenNeuro dataset {dataset_id}",
    )


def fetch_species(dataset_id: str) -> str:
    """Fetch species metadata for an OpenNeuro dataset from GraphQL.

    Args:
        dataset_id: The OpenNeuro dataset identifier (e.g., 'ds005555').

    Returns:
        Raw species value returned by OpenNeuro metadata.
    """
    query = """
        query Dataset($datasetId: ID!) {
            dataset(id: $datasetId) {
                metadata {
                    species
                }
            }
        }
    """
    variables = {
        "datasetId": dataset_id,
    }

    response = _graphql_query_openneuro(
        query,
        variables,
    )
    species = response["data"]["dataset"]["metadata"]["species"]
    return species


def construct_s3_url_from_path(
    dataset_id: str,
    data_file_path: str,
    recording_id: str,
) -> str:
    """Construct an S3 URL prefix for a recording.

    Args:
        dataset_id: OpenNeuro dataset identifier
        data_file_path: Relative path to the EEG/iEEG file within the dataset
        recording_id: Recording identifier

    Example:
        >>> construct_s3_url_from_path(
        ...     dataset_id="ds004019",
        ...     data_file_path="sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1_eeg.edf",
        ...     recording_id="sub-01_ses-01_task-nap_run-1"
        ... )
        's3://openneuro.org/ds004019/sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1'

    Returns:
        S3 URL prefix for downloading recording-related files.
    """
    parent_dir = str(Path(data_file_path).parent)
    return f"s3://{OPENNEURO_S3_BUCKET}/{dataset_id}/{parent_dir}/{recording_id}"


def download_recording(s3_url: str, target_dir: Path) -> list[Path]:
    """Download all files matching an S3 prefix pattern for a recording.

    Args:
        s3_url: S3 URL prefix pattern (e.g., 's3://openneuro.org/ds005555/sub-1/eeg/sub-1_task-Sleep')
        target_dir: Local directory to download files to

    Returns:
        List of downloaded file paths

    Raises:
        RuntimeError: If download fails
    """
    return download_prefix_from_url(s3_url, target_dir)


def download_meta(
    dataset_id: str,
    target_dir: Path,
    filename: str,
    *,
    redownload: bool = False,
    required: bool = False,
) -> Path | None:
    """Download a metadata file from OpenNeuro S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        target_dir: Local directory to download to
        filename: Metadata filename at the dataset root (for example,
            ``"dataset_description.json"`` or ``"participants.tsv"``)
        redownload: If ``True``, re-download and overwrite any existing local file.
            If ``False`` (default), an existing local file is returned as-is.
        required: If ``True``, raise ``RuntimeError`` when the file is missing on
            S3. If ``False`` (default), return ``None`` when the file is absent.

    Returns:
        Path to the downloaded or existing local file, or ``None`` if the file is
        not present on S3 and ``required`` is ``False``.

    Raises:
        RuntimeError: If the download fails, or if the file is missing on S3 and
            ``required`` is ``True``.
    """
    target_dir = Path(target_dir)
    target_path = target_dir / filename
    key = f"{dataset_id}/{filename}"

    result = download_object(
        OPENNEURO_S3_BUCKET,
        key,
        target_path,
        redownload=redownload,
    )

    if result is None and required:
        raise RuntimeError(f"{filename} not found for {dataset_id} on OpenNeuro S3")

    return result


def download_dataset_description(
    dataset_id: str,
    target_dir: Path,
    *,
    redownload: bool = False,
) -> Path:
    """Download dataset_description.json from OpenNeuro S3.

    This file is required for mne-bids to recognize a valid BIDS dataset.
    If the file already exists locally, it is not re-downloaded unless
    ``redownload`` is ``True``.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        target_dir: Local directory to download to
        redownload: If ``True``, re-download and overwrite any existing local file.
            If ``False`` (default), an existing local file is returned as-is.

    Returns:
        Path to the downloaded or existing dataset_description.json file

    Raises:
        RuntimeError: If download fails or file doesn't exist on S3
    """
    return download_meta(
        dataset_id,
        target_dir,
        "dataset_description.json",
        redownload=redownload,
        required=True,
    )


def download_participants_tsv(
    dataset_id: str,
    target_dir: Path,
    *,
    redownload: bool = False,
) -> Path | None:
    """Download participants.tsv from OpenNeuro S3.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        target_dir: Local directory to download to
        redownload: If ``True``, re-download and overwrite any existing local file.
            If ``False`` (default), an existing local file is returned as-is.

    Returns:
        Path to the downloaded or existing participants.tsv file, or ``None`` if
        the dataset has no participants.tsv on OpenNeuro S3.

    Raises:
        RuntimeError: If the download fails for a reason other than the file being
            absent.
    """
    target_dir = Path(target_dir)
    target_path = target_dir / "participants.tsv"

    result = download_meta(
        dataset_id,
        target_dir,
        "participants.tsv",
        redownload=redownload,
    )

    if result is None and redownload and target_path.exists() and target_path.is_file():
        target_path.unlink()

    return result


def _graphql_query_openneuro(query: str, variables: dict | None = None) -> dict:
    """Execute an OpenNeuro GraphQL query with retry.

    Args:
        query: The GraphQL query to execute
        variables: Variables passed to the GraphQL query.

    Returns:
        Decoded JSON response from the GraphQL endpoint.

    Raises:
        Exception: If all retry attempts fail or the response contains GraphQL
            errors.
    """

    def _retry(max_attempts=5, initial_wait=4, max_wait=10):
        def decorator(func):
            import time

            def wrapper(*args, **kwargs):
                attempt = 0
                wait_time = initial_wait
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, max_wait)

            return wrapper

        return decorator

    @_retry(max_attempts=5, initial_wait=4, max_wait=10)
    def _graphql_query(query, variables=None):
        response = requests.post(
            GRAPHQL_ENDPOINT, json={"query": query, "variables": variables}
        )
        if response.status_code == 200:
            json_response = response.json()
            # Check for "errors" key in the GraphQL response
            if "errors" in json_response and json_response["errors"]:
                raise Exception(
                    f"GraphQL query returned errors: {json_response['errors']}"
                )
            return json_response

        else:
            raise Exception(f"Query failed with status code {response.status_code}")

    return _graphql_query(query, variables)
