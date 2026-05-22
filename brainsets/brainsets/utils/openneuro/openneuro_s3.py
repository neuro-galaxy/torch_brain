"""OpenNeuro dataset utilities.

This module provides functions for dataset validation, file listing,
and downloading from OpenNeuro's S3 bucket.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional
import logging
import requests
import pandas as pd

try:
    from botocore.exceptions import ClientError

    BOTO_AVAILABLE = True
except ImportError:
    ClientError = Exception
    BOTO_AVAILABLE = False

from brainsets.utils.s3_utils import (
    download_prefix_from_url,
    get_cached_s3_client,
    get_object_list,
)

OPENNEURO_S3_BUCKET = "openneuro.org"
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


def fetch_participants_tsv(dataset_id: str) -> Optional[pd.DataFrame]:
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

        df = pd.read_csv(
            BytesIO(content),
            sep="\t",
            na_values=["n/a", "N/A"],
            keep_default_na=True,
        )

        if "participant_id" not in df.columns:
            logging.warning(
                f"No participant_id column found in participants.tsv file in OpenNeuro dataset {dataset_id}. "
                "Returning None."
            )
            return None

        df = df.set_index("participant_id")
        return df

    except ClientError as e:
        if BOTO_AVAILABLE:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                return None
        raise


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
        >>>     dataset_id="ds004019",
        >>>     data_file_path="sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1_eeg.edf",
        >>>     recording_id="sub-01_ses-01_task-nap_run-1"
        >>> )
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


def download_dataset_description(dataset_id: str, target_dir: Path) -> Path:
    """Download dataset_description.json from OpenNeuro S3.

    This file is required for mne-bids to recognize a valid BIDS dataset.
    If the file already exists locally, it is not re-downloaded.

    Args:
        dataset_id: The OpenNeuro dataset identifier
        target_dir: Local directory to download to

    Returns:
        Path to the downloaded or existing dataset_description.json file

    Raises:
        RuntimeError: If download fails or file doesn't exist on S3
    """
    target_dir = Path(target_dir)
    target_path = target_dir / "dataset_description.json"

    if target_path.exists():
        return target_path

    s3_client = get_cached_s3_client()
    key = f"{dataset_id}/dataset_description.json"

    try:
        response = s3_client.get_object(Bucket=OPENNEURO_S3_BUCKET, Key=key)
        content = response["Body"].read()

        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(content)

        return target_path

    except ClientError as e:
        error_code = ""
        if BOTO_AVAILABLE:
            error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404"):
            raise RuntimeError(
                f"dataset_description.json not found for {dataset_id} on OpenNeuro S3"
            ) from e
        raise RuntimeError(
            f"Failed to download dataset_description.json for {dataset_id}: {e}"
        ) from e


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
            import random

            def wrapper(*args, **kwargs):
                attempt = 0
                wait_time = initial_wait
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
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
