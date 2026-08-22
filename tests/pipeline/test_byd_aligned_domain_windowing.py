from pathlib import Path

import numpy as np
import pytest
from _utils import add_pipelines_to_path
from pynwb import NWBHDF5IO

add_pipelines_to_path()
from keles_byd_2024.pipeline import (  # noqa: E402
    EDGE_OFFSET_S,
    _apply_task_window,
    _build_windows,
    _extract_lfp,
)

RAW_CS41_DIR = Path(
    "/home/geeling/Projects/tb_buildathon/data/BYD_brainsets/raw/keles_byd_2024/sub-CS41"
)


def _pick_cs41_nwb() -> Path:
    nwb_files = sorted(RAW_CS41_DIR.glob("*.nwb"))
    if len(nwb_files) == 0:
        pytest.skip(f"No NWB files found under {RAW_CS41_DIR}")
    return nwb_files[0]


def test_aligned_domain_start_keeps_early_windows_that_legacy_zero_would_drop():
    """Aligned-domain split generation should preserve valid early windows."""
    nwb_path = _pick_cs41_nwb()

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        lfp_data, lfp_time, _, _ = _extract_lfp(nwbfile)
        trials_df = (
            nwbfile.trials.to_dataframe() if nwbfile.trials is not None else None
        )

    _, lfp_time_use = _apply_task_window(lfp_data, lfp_time, trials_df, EDGE_OFFSET_S)
    aligned_domain_start = float(np.min(lfp_time_use))
    aligned_domain_end = float(np.max(lfp_time_use))

    # Fixed-shift BYD alignment should provide negative start context.
    assert aligned_domain_start < 0.0

    # Construct a valid early label whose 1s-pre window starts < 0 but > aligned_domain_start.
    times = np.array([0.2], dtype=np.float64)
    labels = np.array([1], dtype=np.int64)

    start_aligned, end_aligned, labels_aligned = _build_windows(
        times,
        labels,
        pre_offset_s=1.0,
        post_offset_s=1.0,
        domain_start=aligned_domain_start,
        domain_end=aligned_domain_end,
    )
    assert len(start_aligned) == 1
    assert len(end_aligned) == 1
    assert len(labels_aligned) == 1
    assert start_aligned[0] < 0.0

    # Legacy behavior (domain_start=0.0) would incorrectly drop this window.
    start_legacy, end_legacy, labels_legacy = _build_windows(
        times,
        labels,
        pre_offset_s=1.0,
        post_offset_s=1.0,
        domain_start=0.0,
        domain_end=aligned_domain_end,
    )
    assert len(start_legacy) == 0
    assert len(end_legacy) == 0
    assert len(labels_legacy) == 0
