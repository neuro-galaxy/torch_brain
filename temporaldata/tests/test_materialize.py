import pytest
import os
import h5py
import numpy as np
import tempfile
from temporaldata import (
    ArrayDict,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    Data,
)
from temporaldata.temporaldata import LazyInterval


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        # clean up the temporary file after the test
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def test_materialize(test_filepath):
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        some_numpy_array=np.array([1, 2, 3]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            raw=np.zeros((1000, 3)),
            sampling_rate=250.0,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        trials=Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)

        # check that the data is lazy loaded
        assert all(
            isinstance(data.spikes.__dict__[key], h5py.Dataset)
            for key in data.spikes.keys()
        )
        assert all(
            isinstance(data.lfp.__dict__[key], h5py.Dataset) for key in data.lfp.keys()
        )
        assert all(
            isinstance(data.units.__dict__[key], h5py.Dataset)
            for key in data.units.keys()
        )
        assert all(
            isinstance(data.trials.__dict__[key], h5py.Dataset)
            for key in data.trials.keys()
        )

        # materialize the data
        data.materialize()

    # check that the data is now materialized
    # Note: It is important to assert this when file is closed
    assert all(
        isinstance(data.spikes.__dict__[key], np.ndarray) for key in data.spikes.keys()
    )
    assert all(
        isinstance(data.lfp.__dict__[key], np.ndarray) for key in data.lfp.keys()
    )
    assert all(
        isinstance(data.units.__dict__[key], np.ndarray) for key in data.units.keys()
    )
    assert all(
        isinstance(data.trials.__dict__[key], np.ndarray) for key in data.trials.keys()
    )

    # Check that none of the objects in the data object are the Lazy variant
    for key in data.keys():
        obj = getattr(data, key)
        if isinstance(obj, (Data, RegularTimeSeries, IrregularTimeSeries, Interval)):
            assert (
                "Lazy" not in obj.__class__.__name__
            ), f"{key} is still a Lazy object: {obj.__class__.__name__}"

    assert not isinstance(
        data.domain, LazyInterval
    ), f"data.domain is still a LazyInterval object"
