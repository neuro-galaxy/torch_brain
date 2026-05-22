from pathlib import Path

import numpy as np

from brainsets.config import CONFIG_FILE, load_config
from temporaldata import Interval


def get_processed_dir(path: Path = CONFIG_FILE) -> str:
    """Return ``processed_dir`` from config, or raise if unavailable."""
    config = load_config(path)
    if config is None:
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Please run `brainsets config set` or pass `root` explicitly."
        )
    return config["processed_dir"]
