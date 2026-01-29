import pytest
import os
import copy
import h5py
import numpy as np
import tempfile
import logging
from temporaldata import (
    ArrayDict,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    Data,
)


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


def test_data():
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
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

    assert data.keys() == [
        "session_id",
        "spikes",
        "lfp",
        "units",
        "trials",
        "drifting_gratings_imgs",
    ]

    data = data.slice(1.0, 3.0)
    assert data.absolute_start == 1.0

    assert ["session_id", "spikes", "lfp", "units", "trials", "drifting_gratings_imgs"]

    assert len(data.spikes) == 3
    assert np.allclose(data.spikes.timestamps, np.array([1.1, 1.2, 1.3]))
    assert np.allclose(data.spikes.unit_index, np.array([0, 1, 2]))

    assert len(data.lfp) == 500

    assert len(data.trials) == 2
    assert np.allclose(data.trials.start, np.array([0, 1]))

    data = data.slice(0.4, 1.4)
    assert data.absolute_start == 1.4


def test_data_copy():
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

    ### test copy
    data_copy = copy.copy(data)
    data_copy.some_numpy_array[0] = 10
    # this is a shallow copy, so the original object should be modified
    assert data.some_numpy_array[0] == 10

    data_copy.spikes.unit_index[0] = 2
    # this is a shallow copy, so the original object should be modified
    assert data.spikes.unit_index[0] == 2

    data_copy.spikes.unit_index = np.array([0, 0, 0, 0, 0, 0])
    # the unit_index variable is not shared between the two objects
    assert data.spikes.unit_index[0] == 2

    ### test deepcopy
    data_deepcopy = copy.deepcopy(data)
    data_deepcopy.some_numpy_array[1] = 20
    # this is a deep copy, so the original object should not be modified
    assert data.some_numpy_array[1] == 2

    data_deepcopy.spikes.unit_index[1] = 2
    # this is a deep copy, so the original object should not be modified
    assert data.spikes.unit_index[1] == 0


def test_lazy_data_copy(test_filepath):
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
        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)

        # this will copy all references to any h5py datasets
        data_copy = copy.copy(data)
        data_copy.some_numpy_array[0] = 10
        # TODO Data does not lazy load numpy arrays that are not wrapped in an
        # ArrayDict object, this will change in the future.
        # because some_numpy_array is not a h5py dataset, changing it will affect
        # the original object
        assert isinstance(data.__dict__["some_numpy_array"], np.ndarray)
        # this is a shallow copy, so the original object should be modified
        assert data.some_numpy_array[0] == 10

        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)
        data_copy.spikes.unit_index[0] = 2
        assert isinstance(data.spikes.__dict__["unit_index"], h5py.Dataset)
        # this is a shallow copy, but the unit_index is a h5py dataset, so
        # the original object will not be modified
        assert data.spikes.unit_index[0] == 0

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)

        data_deepcopy = copy.deepcopy(data)
        data_deepcopy.some_numpy_array[0] = 10
        # this is a deep copy, so the original object should not be modified
        assert data.some_numpy_array[0] == 1

        data_deepcopy.spikes.unit_index[0] = 2
        # this is a deep copy, so the original object should not be modified
        assert data.spikes.unit_index[0] == 0


def test_data_absolute_start(test_filepath):
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

    data = data.slice(1.0, 3.0)

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f, lazy=True)
        assert data.absolute_start == 1.0


def test_timeless_data(test_filepath):
    # when defining a Data object that has no time-based attributes, we do no need to
    # specify a domain
    subject = Data(
        id="jenkins",
        age=5.0,
        species="HUMAN",
        description="À89!ÜÞ",
        image=np.ones((32, 32, 3)),
    )

    # we cannot slice this object because it has no domain or time-based attributes
    with pytest.raises(ValueError):
        subject.slice(0.1, 0.2)

    with h5py.File(test_filepath, "w") as f:
        subject.to_hdf5(f)

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f)

        assert data.id == "jenkins"
        assert data.age == 5.0
        assert data.species == "HUMAN"
        assert data.description == "À89!ÜÞ"

        # TODO(mehdi) image is a numpy array so it should be lazy loaded
        # assert isinstance(data.__dict__["image"], h5py.Dataset)
        assert np.allclose(data.image, np.ones((32, 32, 3)))

    data = Data(
        subject=subject,
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    with h5py.File(test_filepath, "r") as f:
        data = Data.from_hdf5(f)

        assert data.subject.id == "jenkins"
        assert data.subject.age == 5.0
        assert data.subject.species == "HUMAN"
        assert np.allclose(data.subject.image, np.ones((32, 32, 3)))

        assert len(data.spikes) == 6
        assert np.allclose(
            data.spikes.timestamps, np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3])
        )
        assert np.allclose(data.spikes.unit_index, np.array([0, 0, 1, 0, 1, 2]))


def test_precision(caplog):
    with caplog.at_level(logging.WARNING):
        spikes = IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3], dtype=np.float64),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        )
    assert caplog.text == ""

    with caplog.at_level(logging.WARNING):
        spikes = IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3], dtype=np.float16),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        )
    assert (
        f"timestamps is of type {spikes.timestamps.dtype} not of type float64."
        in caplog.text
    )

    with caplog.at_level(logging.WARNING):
        trials = Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        )
        assert (
            f"start is of type {trials.start.dtype} not of type float64." in caplog.text
        )

    lfp = RegularTimeSeries(
        raw=np.zeros((1000, 3)),
        sampling_rate=250.0,
        domain="auto",
    )
    assert lfp.timestamps.dtype == np.float64


def test_nested_attributes():
    data = Data(
        session_id="session_0",
        domain=Interval.from_list([(0, 3)]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
    )

    # test get_nested_attribute
    out = data.get_nested_attribute("spikes.timestamps")
    assert np.allclose(out, data.spikes.timestamps)

    out = data.get_nested_attribute("units.id")
    for i, id in enumerate(out):
        assert id == data.units.id[i]

    # test has_nested_attribute
    assert data.has_nested_attribute("spikes.timestamps")
    assert data.has_nested_attribute("units.id")
    assert not data.has_nested_attribute("spikes.unit_index")

    # test that an error is raised if the attribute does not exist
    with pytest.raises(AttributeError):
        data.get_nested_attribute("spikes.unit_index")

    # test that it is possible to access attributes that are not nested
    assert data.get_nested_attribute("session_id") == "session_0"
    assert data.has_nested_attribute("session_id")


class TestSetNestedAttribute:
    @pytest.fixture
    def data(self):
        return Data(
            not_nested="not_nested_attrib",
            session=Data(id="session_0"),
            domain=Interval.from_list([(0, 3)]),
            spikes=IrregularTimeSeries(
                timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
                waveforms=np.zeros((6, 48)),
                domain="auto",
            ),
            units=ArrayDict(
                id=np.array(["unit_0", "unit_1", "unit_2"]),
                brain_region=np.array(["M1", "M1", "PMd"]),
            ),
        )

    def test_changes_happen_inplace(self, data):
        data.set_nested_attribute("session.id", "new_session_id")
        assert data.session.id == "new_session_id"
        data.set_nested_attribute("not_nested", "new_not_nested_attrib")
        assert data.not_nested == "new_not_nested_attrib"

    def test_return_value(self, data):
        data_ret = data.set_nested_attribute("session.id", "new_session_id")
        assert id(data_ret) == id(data)

    def test_error_on_nonexistent_attrib(self, data):
        with pytest.raises(AttributeError):
            data.set_nested_attribute("non.existent", None)

    def test_create_new_attribute(self, data):
        # Set a string attribute that does not exist
        data.set_nested_attribute("session.new_attrib", "a_new_attribute")
        assert data.session.new_attrib == "a_new_attribute"

        # Set an array attribute that does not exist
        unit_index = np.random.randint(0, 3, len(data.spikes))
        data.set_nested_attribute("spikes.unit_index", unit_index)
        assert (data.spikes.unit_index == unit_index).all()

        new_attrib = np.array([1, 2, 3])
        data.set_nested_attribute("units.new_attrib", new_attrib)
        assert (data.units.new_attrib == new_attrib).all()

        # Validation should happen on object creation
        with pytest.raises(ValueError):
            data.set_nested_attribute(
                "spikes.unit_index2", np.arange(len(data.spikes) + 4)
            )
        with pytest.raises(ValueError):
            data.set_nested_attribute(
                "units.new_attrib2", np.arange(len(data.units) - 1)
            )

    def test_overwrite_type(self, data):
        # Test that we should be able to overwrite type of object being set
        data.set_nested_attribute("spikes", 0)
        assert data.spikes == 0


def test_data_has_nested_attribute_lazy(test_filepath):
    """Tests the Data.has_nested_attribute method with lazily loaded objects."""
    data_to_save = Data(
        session_id="session_lazy_hsna_test",
        some_numpy_array=np.array([10, 20, 30]),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["u0_hsna", "u1_hsna"]),
            type=np.array(["typeA_hsna", "typeB_hsna"]),
        ),
        nested_data=Data(
            level2_attr="hello_nested_hsna",
            level2_array_dict=ArrayDict(l2_field=np.array([100, 200])),
            level2_primitive=42,
            domain=Interval(0.0, 1.0),
        ),
        domain=Interval(0.0, 3.0),
    )

    with h5py.File(test_filepath, "w") as f:
        data_to_save.to_hdf5(f)
    del data_to_save

    with h5py.File(test_filepath, "r") as f:
        lazy_data = Data.from_hdf5(f, lazy=True)

        # Pre-checks for laziness: ensure some attributes are indeed h5py.Dataset
        assert isinstance(lazy_data.spikes.__dict__["unit_index"], h5py.Dataset)
        assert isinstance(lazy_data.units.__dict__["id"], h5py.Dataset)
        assert isinstance(
            lazy_data.nested_data.level2_array_dict.__dict__["l2_field"], h5py.Dataset
        )

        # === Test existing paths ===
        assert lazy_data.has_nested_attribute("session_id")
        assert lazy_data.has_nested_attribute("some_numpy_array")
        assert lazy_data.has_nested_attribute("spikes")
        assert lazy_data.has_nested_attribute("spikes.unit_index")
        assert lazy_data.has_nested_attribute("units.id")
        assert lazy_data.has_nested_attribute("nested_data")
        assert lazy_data.has_nested_attribute("nested_data.level2_attr")
        assert lazy_data.has_nested_attribute("nested_data.level2_array_dict")
        assert lazy_data.has_nested_attribute("nested_data.level2_array_dict.l2_field")
        assert lazy_data.has_nested_attribute("nested_data.level2_primitive")

        # Check attributes remain lazy after has_nested_attribute calls
        assert isinstance(
            lazy_data.spikes.__dict__["unit_index"], h5py.Dataset
        ), "spikes.unit_index was loaded"
        assert isinstance(
            lazy_data.units.__dict__["id"], h5py.Dataset
        ), "units.id was loaded"
        assert isinstance(
            lazy_data.nested_data.level2_array_dict.__dict__["l2_field"], h5py.Dataset
        ), "nested_data.l2_array_dict.l2_field was loaded"

        # === Test non-existent paths (should return False) ===
        assert not lazy_data.has_nested_attribute("non_existent_toplevel")
        assert not lazy_data.has_nested_attribute("spikes.foo_bar_baz")
        assert not lazy_data.has_nested_attribute("units.id.deeper_false")
        assert not lazy_data.has_nested_attribute(
            "nested_data.level2_array_dict.l2_field.deeper_false_too"
        )
        assert not lazy_data.has_nested_attribute("nested_data.non_existent_attr")

        # === Test empty path ===
        assert not lazy_data.has_nested_attribute("")

        # === Test paths that should fail due to AttributeError on __dict__ access for primitives ===
        with pytest.raises(
            AttributeError, match="'str' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("session_id.foo")

        with pytest.raises(
            AttributeError, match="'numpy.ndarray' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("some_numpy_array.shape")
        with pytest.raises(
            AttributeError, match="'numpy.ndarray' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("some_numpy_array.foo")

        with pytest.raises(
            AttributeError, match="'numpy.int64' object has no attribute '__dict__'"
        ):
            lazy_data.has_nested_attribute("nested_data.level2_primitive.foo")


def test_data_auto_domain():
    data = Data(
        session_id="session_0",
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
            start=np.array([0, 1, 5]),
            end=np.array([1, 2, 6]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
        ),
        drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
        domain="auto",
    )

    assert np.allclose(data.domain.start, np.array([0, 5]))
    assert np.allclose(data.domain.end, np.array([3.996, 6]))
