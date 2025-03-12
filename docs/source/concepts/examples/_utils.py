import os
import requests
from typing import Optional


def download_file_from_s3(file_path: str, target_path: Optional[str] = None) -> None:
    """Downloads a file from the torch-brain S3 bucket.

    Args:
        file_path: Path to the file within the S3 bucket
        target_path: Local path where the file will be saved. If None, uses file_path
    """
    url = f"https://torch-brain.s3.amazonaws.com/{file_path}"

    output_path = target_path if target_path is not None else file_path

    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Warning: File {output_path} already exists, skipping download")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download the file
    response = requests.get(url)
    response.raise_for_status()

    # Save to disk
    with open(output_path, "wb") as f:
        f.write(response.content)
