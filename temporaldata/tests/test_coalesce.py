import pytest
import numpy as np
from temporaldata import Interval


def test_interval_coalesce():
    data = Interval(
        start=np.array([0.0, 1.0, 2.0]),
        end=np.array([1.0, 2.0, 3.0]),
        go_cue_time=np.array([0.5, 1.5, 2.5]),
        drifting_gratings_dir=np.array([0, 45, 90]),
        timekeys=["start", "end", "go_cue_time"],
    )

    coalesced_data = data.coalesce()
    assert len(coalesced_data) == 1
    # only keep start and end
    assert len(coalesced_data.keys()) == 2
    assert np.allclose(coalesced_data.start, np.array([0.0]))
    assert np.allclose(coalesced_data.end, np.array([3.0]))

    data = Interval(
        start=np.array([0.0, 1.0, 2.0, 4.0, 4.5, 5.0, 10.0]),
        end=np.array([0.5, 2.0, 2.5, 4.5, 5.0, 6.0, 11.0]),
    )

    coalesced_data = data.coalesce()
    assert len(coalesced_data) == 4
    assert np.allclose(coalesced_data.start, np.array([0.0, 1.0, 4.0, 10.0]))
    assert np.allclose(coalesced_data.end, np.array([0.5, 2.5, 6.0, 11.0]))
