import numpy as np
import pytest
from _utils import add_pipelines_to_path

add_pipelines_to_path()
from karpowicz_pandarinath_falcon_h1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    mask_to_interval,
    trials_from_numbers,
)
from karpowicz_pandarinath_falcon_h1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_h1,
)
from karpowicz_pandarinath_falcon_m1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_m1,
)
from karpowicz_pandarinath_falcon_m2_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_m2,
)


def _inside(interval, t):
    return np.array([((interval.start <= x) & (x < interval.end)).any() for x in t])


def test_mask_to_interval_matches_mask():
    t = np.arange(10) * 0.02
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    interval = mask_to_interval(mask, t)
    assert len(interval.start) == 3
    assert (_inside(interval, t) == mask).all()
    assert interval.is_sorted() and interval.is_disjoint()


def test_mask_to_interval_empty_and_length_check():
    t = np.arange(5) * 0.02
    assert len(mask_to_interval(np.zeros(5, dtype=bool), t).start) == 0
    with pytest.raises(ValueError):
        mask_to_interval(np.ones(4, dtype=bool), t)


def test_trials_from_numbers():
    t = np.arange(7) * 0.02
    trials = trials_from_numbers(np.array([3, 3, 4, 4, 4, 7, 7]), t)
    assert list(trials.number) == [3, 4, 7]
    assert (_inside(trials, t)).all()
    assert trials.is_disjoint()


def test_parse_filenames():
    assert parse_h1(
        "sub-HumanPitt/sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"
    ) == ("held_in_calib", "19250108T110520")
    assert parse_m1("sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys.nwb") == (
        "held_out_calib",
        "20121004",
    )
    assert parse_m2(
        "sub-MonkeyN-held-in-minival_ses-2020-10-19-Run2_behavior+ecephys.nwb"
    ) == ("held_in_minival", "20201019", "2")
    with pytest.raises(ValueError):
        parse_m1("sub-MonkeyL_ses-20121004.nwb")
