"""Generic S3 utilities for downloading data from public buckets."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import BaseClient
    from botocore.config import Config
    from botocore.exceptions import ClientError

    BOTO_AVAILABLE = True
except ImportError:
    boto3 = None
    UNSIGNED = None
    BaseClient = None
    Config = None
    ClientError = None
    BOTO_AVAILABLE = False


def _check_boto_available(func_name: str) -> None:
    """Raise ImportError if boto3/botocore is not available."""
    if not BOTO_AVAILABLE:
        raise ImportError(
            f"{func_name} requires boto3 and botocore which are not installed. "
            "Install them with `pip install boto3`"
        )


@lru_cache(maxsize=1)
def get_cached_s3_client(
    retry_mode: str = "adaptive",
    max_attempts: int = 5,
    max_pool_connections: int = 30,
):
    """Get a cached S3 client configured for anonymous access to public buckets.

    Uses boto3's retry modes which include:
    - Exponential backoff with random jitter
    - Automatic retries on transient errors, throttling (429), and 5xx status codes

    Args:
        retry_mode: Retry mode ("standard" or "adaptive")
        max_attempts: Maximum number of retry attempts
        max_pool_connections: Maximum number of connections in the pool

    Returns:
        A configured boto3 S3 client for unsigned/anonymous access

    Raises:
        ImportError: If boto3/botocore is not installed.
    """
    _check_boto_available("get_cached_s3_client")
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={
                "mode": retry_mode,
                "total_max_attempts": max_attempts,
            },
            max_pool_connections=max_pool_connections,
        ),
    )


def get_object_list(
    bucket: str,
    prefix: str,
    s3_client: "BaseClient | None" = None,
) -> list[str]:
    """List all object keys under a prefix (excludes directories).

    Args:
        bucket: S3 bucket name
        prefix: Key prefix to filter objects (e.g., "ds005555/")
        s3_client: Optional pre-configured S3 client

    Returns:
        List of object keys (relative to the prefix)

    Raises:
        RuntimeError: If listing fails
        ImportError: If boto3/botocore is not installed.
    """
    _check_boto_available("get_object_list")
    if s3_client is None:
        s3_client = get_cached_s3_client()

    keys = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith("/") and key.startswith(prefix):
                    relative_path = key[len(prefix) :]
                    if relative_path:
                        keys.append(relative_path)

    except Exception as e:
        raise RuntimeError(f"Error listing objects in {bucket}/{prefix}: {e}") from e

    return keys


def download_prefix(
    bucket: str,
    prefix: str,
    target_dir: Path,
    strip_prefix: str = None,
    s3_client: "BaseClient | None" = None,
) -> list[Path]:
    """Download all files matching a prefix pattern.

    Args:
        bucket: S3 bucket name
        prefix: Key prefix to match files
        target_dir: Local directory to download files to
        strip_prefix: Prefix to strip from keys when creating local paths.
            If None, uses the first path component (dataset_id).
        s3_client: Optional pre-configured S3 client

    Returns:
        List of downloaded file paths

    Raises:
        RuntimeError: If download fails or no files match
        ImportError: If boto3/botocore is not installed.

    Examples:
        >>> # Basic usage
        >>> download_prefix(
                bucket="openneuro.org",
                prefix="ds005555/sub-1/eeg/sub-1_task-Sleep",
                target_dir=Path("~/data/raw/brainset_ds005555")
            )
        >>> # Custom strip_prefix
        >>> download_prefix(
                bucket="fcp-indi",
                prefix="data/Projects/EEG_Eyetracking_CMI_data/A00054400",
                target_dir=Path("~/data/raw/brainset_ds005555"),
                strip_prefix="data/Projects/"
            )
    """
    _check_boto_available("download_prefix")
    if s3_client is None:
        s3_client = get_cached_s3_client()

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if strip_prefix is None:
        # If prefix shows no sub-directories, use it as-is (eg. "ds005555/")
        if "/" not in prefix:
            strip_prefix = prefix
        else:
            strip_prefix = prefix.split("/")[0] + "/"
    strip_prefix = strip_prefix.rstrip("/") + "/"
    downloaded_files = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                obj_key = obj["Key"]
                if obj_key.endswith("/"):
                    continue

                if obj_key.startswith(strip_prefix):
                    rel_path = obj_key[len(strip_prefix) :]
                else:
                    rel_path = obj_key

                local_path = target_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    s3_client.download_file(bucket, obj_key, str(local_path))
                    downloaded_files.append(local_path)
                except ClientError as e:
                    raise RuntimeError(f"Failed to download {obj_key}: {e}") from e

        if not downloaded_files:
            raise RuntimeError(
                f"No files found matching prefix '{prefix}' in bucket '{bucket}'"
            )

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to download from {bucket}/{prefix}: {e}") from e

    return downloaded_files


def download_prefix_from_url(s3_url: str, target_dir: Path) -> list[Path]:
    """Download all files matching an S3 URL prefix pattern.

    Args:
        s3_url: S3 URL prefix pattern (e.g., 's3://bucket/prefix')
        target_dir: Local directory to download files to

    Returns:
        List of downloaded file paths

    Raises:
        ValueError: If URL is not a valid S3 URL
        RuntimeError: If download fails
        ImportError: If boto3/botocore is not installed.
    """
    _check_boto_available("download_prefix_from_url")
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URL: {s3_url}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    return download_prefix(bucket, prefix, target_dir)
