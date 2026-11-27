import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path
from pynwb import NWBHDF5IO

add_pipelines_to_path()
from keles_byd_2024.pipeline import (  # noqa: E402
    EDGE_OFFSET_S,
    LFP_MOVIE_PAD_S,
    _apply_task_window,
    _build_windows,
    _extract_encoding_interval,
    _extract_lfp,
)


@pytest.fixture(scope="module")
def aligned_byd_recording():
    raw_dir_value = os.environ.get("BYD_CS41_RAW_DIR")
    if not raw_dir_value:
        pytest.skip("Set BYD_CS41_RAW_DIR to run the real-data alignment tests")
    raw_dir = Path(raw_dir_value).expanduser()
    nwb_files = sorted(raw_dir.glob("*.nwb"))
    if len(nwb_files) == 0:
        pytest.skip(f"No NWB files found under {raw_dir}")
    with NWBHDF5IO(str(nwb_files[0]), "r") as io:
        nwbfile = io.read()
        lfp_data, lfp_time, _, _ = _extract_lfp(nwbfile)
        trials_df = (
            nwbfile.trials.to_dataframe() if nwbfile.trials is not None else None
        )
        enc_start, enc_stop = _extract_encoding_interval(trials_df)

    lfp_data_use, lfp_time_use = _apply_task_window(
        lfp_data, lfp_time, trials_df, EDGE_OFFSET_S
    )
    return {
        "signal": lfp_data_use,
        "time": lfp_time_use,
        "raw_time": lfp_time,
        "encoding_start": enc_start,
        "encoding_stop": enc_stop,
    }


def test_apply_task_window_real_data_applies_fixed_10s_shift(aligned_byd_recording):
    """Real BYD NWB: prepare path applies fixed 10s movie<->SEEG correction."""
    lfp_data_use = aligned_byd_recording["signal"]
    lfp_time_use = aligned_byd_recording["time"]
    lfp_time = aligned_byd_recording["raw_time"]
    enc_start = aligned_byd_recording["encoding_start"]
    enc_stop = aligned_byd_recording["encoding_stop"]

    assert lfp_data_use.shape[0] == lfp_time_use.shape[0]
    assert lfp_time_use.size > 0

    # Tolerance tied to real sampling interval.
    deltas = np.diff(lfp_time)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    dt = float(np.median(deltas))

    # Left/right bounds after fixed shift and encoding-window expansion.
    raw_min_aligned = float(lfp_time.min() - LFP_MOVIE_PAD_S)
    raw_max_aligned = float(lfp_time.max() - LFP_MOVIE_PAD_S)

    # Alignment contract:
    # keep_start = enc_start - EDGE_OFFSET_S
    # keep_stop = enc_stop + EDGE_OFFSET_S
    # after fixed aligned timebase t_aligned = t_raw - LFP_MOVIE_PAD_S
    expected_min = max(raw_min_aligned, float(enc_start - EDGE_OFFSET_S))
    expected_max = min(raw_max_aligned, float(enc_stop + EDGE_OFFSET_S))
    assert np.isclose(float(lfp_time_use.min()), expected_min, atol=dt + 1e-6)
    assert np.isclose(float(lfp_time_use.max()), expected_max, atol=dt + 1e-6)

    # Aligned timeline should include sample(s) near movie t=0.
    assert float(np.min(np.abs(lfp_time_use))) <= (dt + 1e-6)

    # When encoding starts at movie t=0 (true for current CS41 files),
    # fixed LFP_MOVIE_PAD_S should expose about 10s pre-movie context.
    if np.isclose(enc_start, 0.0, atol=dt + 1e-6):
        assert np.isclose(
            float(lfp_time_use.min()), -float(EDGE_OFFSET_S), atol=2.0 * dt + 1e-6
        )

    # Cropping may discard the raw minimum; map the retained sample back instead.
    first_raw_time = float(lfp_time_use[0] + LFP_MOVIE_PAD_S)
    assert float(np.min(np.abs(lfp_time - first_raw_time))) <= dt + 1e-6


def test_aligned_domain_start_keeps_early_windows_that_legacy_zero_would_drop(
    aligned_byd_recording,
):
    """Aligned-domain split generation should preserve valid early windows."""
    lfp_time_use = aligned_byd_recording["time"]
    aligned_domain_start = float(np.min(lfp_time_use))
    aligned_domain_end = float(np.max(lfp_time_use))
    assert aligned_domain_start < 0.0

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

    # A zero lower bound would discard available pre-movie context.
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


def test_synthetic_alignment_preserves_boundary_windows_and_drops_partial_support():
    raw_time = np.arange(5.0, 17.0)
    signal = np.column_stack([raw_time, raw_time + 100.0])
    trials = pd.DataFrame(
        {"stim_phase": ["encoding"], "start_time": [0.0], "stop_time": [3.0]}
    )

    aligned_signal, aligned_time = _apply_task_window(
        signal, raw_time, trials, edge_offset_s=1.0
    )

    np.testing.assert_array_equal(aligned_time, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(aligned_signal, signal[4:10])

    start, end, labels = _build_windows(
        np.array([-0.5, 0.0, 0.2, 3.0, 3.5]),
        np.array([1, 2, 3, 4, 5]),
        pre_offset_s=1.0,
        post_offset_s=1.0,
        domain_start=float(aligned_time[0]),
        domain_end=float(aligned_time[-1]),
    )

    # Both exact edges and available negative-time context are valid.
    np.testing.assert_allclose(start, [-1.0, -0.8, 2.0])
    np.testing.assert_allclose(end, [1.0, 1.2, 4.0])
    np.testing.assert_array_equal(labels, [2, 3, 4])
