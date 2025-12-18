import boto3
from botocore import UNSIGNED
from botocore.config import Config


def get_s3_client_for_download(
    max_retries: int = 5,
    max_pool_connections: int = 10,
) -> boto3.client:
    """
    Create an S3 client configured for anonymous access to public buckets for downloading data.

    Args:
        max_retries: Maximum number of retry attempts for failed requests.
        max_pool_connections: Maximum number of connections in the pool.

    Returns:
        A configured boto3 S3 client.
    """
    config = Config(
        # Allow anonymous access to public buckets without AWS credentials
        signature_version=UNSIGNED,
        # Adaptive mode adjusts retry delays based on error types and throttling
        retries={"max_attempts": max_retries, "mode": "adaptive"},
        # Connection pool size for concurrent requests
        max_pool_connections=max_pool_connections,
    )
    return boto3.client("s3", config=config)
