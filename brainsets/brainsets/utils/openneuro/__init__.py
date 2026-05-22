"""OpenNeuro utilities subpackage.

This package provides utilities for working with OpenNeuro datasets:
- S3-based file listing and downloading
- BIDS filename parsing for EEG recordings discovery
- OpenNeuroPipeline base class and EEG/iEEG subclasses for building pipelines
"""

from .openneuro_s3 import (
    OPENNEURO_S3_BUCKET,
    construct_s3_url_from_path,
    download_dataset_description,
    download_recording,
    fetch_latest_snapshot_tag,
    fetch_all_filenames,
    fetch_participants_tsv,
    fetch_species,
)
from .pipeline import OpenNeuroPipeline, OpenNeuroDataModality, base_openneuro_parser

__all__ = [
    "OPENNEURO_S3_BUCKET",
    "construct_s3_url_from_path",
    "download_dataset_description",
    "download_recording",
    "fetch_latest_snapshot_tag",
    "fetch_all_filenames",
    "fetch_participants_tsv",
    "fetch_species",
    "OpenNeuroPipeline",
    "OpenNeuroDataModality",
    "base_openneuro_parser",
]
