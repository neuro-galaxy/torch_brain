from pathlib import Path

import numpy as np
import pytest
from _utils import add_pipelines_to_path
from pynwb import NWBHDF5IO

add_pipelines_to_path()
from keles_byd_2024.pipeline import (  # noqa: E402
    EDGE_OFFSET_S,
    LFP_MOVIE_PAD_S,
    _apply_task_window,
    _extract_encoding_interval,
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


def test_apply_task_window_real_data_applies_fixed_10s_shift():
    """Real BYD NWB: prepare path applies fixed 10s movie<->SEEG correction."""
    nwb_path = _pick_cs41_nwb()

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        lfp_data, lfp_time, _, _ = _extract_lfp(nwbfile)
        trials_df = (
            nwbfile.trials.to_dataframe() if nwbfile.trials is not None else None
        )
        enc_start, enc_stop = _extract_encoding_interval(trials_df)

    lfp_data_use, lfp_time_use = _apply_task_window(
        lfp_data, lfp_time, trials_df, EDGE_OFFSET_S
    )

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

    # Sanity check fixed-shift relation at left edge of retained samples.
    # (First retained aligned sample should correspond to raw sample minus fixed pad.)
    observed_shift_s = float(lfp_time.min() - lfp_time_use.min())
    assert np.isclose(observed_shift_s, float(LFP_MOVIE_PAD_S), atol=2.0 * dt + 1e-6)
