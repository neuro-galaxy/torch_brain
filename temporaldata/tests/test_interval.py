import pytest
import numpy as np
import h5py
from temporaldata import Interval, LazyInterval


@pytest.fixture
def test_filepath(request):
    import os, tempfile

    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        # clean up the temporary file after the test
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def test_interval():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        go_cue_time=np.array([0.5, 1.5, 2.5]),
        drifting_gratings_dir=np.array([0, 45, 90]),
        timekeys=["start", "end", "go_cue_time"],
    )

    assert data.keys() == ["start", "end", "go_cue_time", "drifting_gratings_dir"]
    assert len(data) == 3

    assert data.is_sorted()
    assert data.is_disjoint()

    # setting an incorrect attribute
    with pytest.raises(ValueError):
        data.wrong_len = np.array([0, 1, 2, 3])

    with pytest.raises(AssertionError):
        data = Interval(
            start=np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6]),
            end=np.array([1, 2, 3, 4, 5, 6]),
        )


def test_interval_select_by_mask():
    # test masking
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    mask = data.drifting_gratings_dir == 90

    data = data.select_by_mask(mask)

    assert len(data) == 3
    assert np.array_equal(data.start, np.array([2, 5, 7]))
    assert np.array_equal(data.end, np.array([3, 6, 8]))


def test_interval_slice():
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    data = data.slice(2.0, 6.0)

    assert len(data) == 4
    assert np.allclose(data.start, np.array([0, 1, 2, 3]))
    assert np.allclose(data.end, np.array([1, 2, 3, 4]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5, 2.5, 3.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45, 180, 90]))

    data = data.slice(0.0, 2.0)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0, 1]))
    assert np.allclose(data.end, np.array([1, 2]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))

    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    data = data.slice(2.0, 6.0, reset_origin=False)

    assert len(data) == 4
    assert np.allclose(data.start, np.array([2, 3, 4, 5]))
    assert np.allclose(data.end, np.array([3, 4, 5, 6]))
    assert np.allclose(data.go_cue_time, np.array([2.5, 3.5, 4.5, 5.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45, 180, 90]))

    data = data.slice(2.0, 4.0, reset_origin=True)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0, 1]))
    assert np.allclose(data.end, np.array([1, 2]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))


def test_lazy_interval(test_filepath):
    data = Interval(
        start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        go_cue_time=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        drifting_gratings_dir=np.array([0, 45, 90, 45, 180, 90, 0, 90, 45]),
        timekeys=["start", "end", "go_cue_time"],
    )

    with h5py.File(test_filepath, "w") as f:
        data.to_hdf5(f)

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        assert len(data) == 9

        # make sure that nothing is loaded yet
        assert all(isinstance(data.__dict__[key], h5py.Dataset) for key in data.keys())

        # try loading one attribute
        start = data.start
        # make sure that the attribute is loaded
        assert isinstance(start, np.ndarray)
        # make sure that the attribute is loaded correctly
        assert np.array_equal(start, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        # make sure that the loaded attribute replaced the h5py.Dataset reference
        assert isinstance(data.__dict__["start"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "start"
        )

        assert data.__class__ == LazyInterval

        data.end
        assert data.__class__ == LazyInterval

        data.go_cue_time
        data.drifting_gratings_dir
        # final attribute was accessed, the object should automatically convert to ArrayDict
        assert data.__class__ == Interval

    del data

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        # try masking
        mask = data.drifting_gratings_dir == 90
        data = data.select_by_mask(mask)

        # make sure only brain_region was loaded
        assert isinstance(data.__dict__["drifting_gratings_dir"], np.ndarray)
        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key != "drifting_gratings_dir"
        )

        assert len(data) == 3
        assert np.array_equal(data.start, np.array([2, 5, 7]))

        assert np.array_equal(data._lazy_ops["mask"], mask)

        # load another attribute
        go_cue_time = data.go_cue_time
        assert isinstance(go_cue_time, np.ndarray)
        assert np.array_equal(go_cue_time, np.array([2.5, 5.5, 7.5]))

        # mask again!
        mask = data.start >= 6

        # make a new object data2 (mask is not inplace)
        data2 = data.select_by_mask(mask)

        assert len(data2) == 1
        # make sure that the attribute was never accessed, is still not accessed
        assert isinstance(data2.__dict__["end"], h5py.Dataset)

        # check if the mask was applied twice correctly!
        assert np.allclose(data2.end, np.array([8]))

        # make sure that data is still intact
        assert len(data) == 3
        assert np.array_equal(data.end, np.array([3, 6, 8]))

    del data, data2

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        data = data.slice(2.0, 6.0)

        assert len(data) == 4
        assert np.allclose(data.start, np.array([0, 1, 2, 3]))
        assert np.allclose(data.end, np.array([1, 2, 3, 4]))

        assert all(
            isinstance(data.__dict__[key], h5py.Dataset)
            for key in data.keys()
            if key not in ["start", "end"]
        )

        assert np.allclose(data.go_cue_time, np.array([0.5, 1.5, 2.5, 3.5]))

        data = data.slice(0.0, 2.0)

        assert len(data) == 2
        assert np.allclose(data.start, np.array([0, 1]))
        assert np.allclose(data.end, np.array([1, 2]))
        assert np.allclose(data.go_cue_time, np.array([0.5, 1.5]))
        assert np.allclose(data.drifting_gratings_dir, np.array([90, 45]))

    del data

    # try slicing and masking

    with h5py.File(test_filepath, "r") as f:
        data = LazyInterval.from_hdf5(f)

        data = data.slice(2.0, 6.0)
        mask = data.drifting_gratings_dir == 90
        data = data.select_by_mask(mask)

        assert np.allclose(data.start, np.array([0, 3]))


def test_interval_select_by_interval():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        go_cue_time=np.array([0.5, 1.5, 2.5]),
        drifting_gratings_dir=np.array([0, 45, 90]),
        timekeys=["start", "end", "go_cue_time"],
    )

    selection_interval = Interval(
        start=np.array([0.2, 2.5]),
        end=np.array([0.4, 3.12]),
    )
    data = data.select_by_interval(selection_interval)

    assert len(data) == 2
    assert np.allclose(data.start, np.array([0.0, 2.0]))
    assert np.allclose(data.end, np.array([1.0, 3.0]))
    assert np.allclose(data.go_cue_time, np.array([0.5, 2.5]))
    assert np.allclose(data.drifting_gratings_dir, np.array([0, 90]))


def test_interval_iter():
    data = Interval(
        start=np.array([0.0, 1, 2]),
        end=np.array([1, 2, 3]),
        some_other_attribute=np.array([0, 1, 2]),
    )

    assert list(data) == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    # test a single interval
    data = Interval(0.0, 1.0)
    assert list(data) == [(0.0, 1.0)]

    # test an empty interval
    data = Interval(np.array([]), np.array([]))
    assert list(data) == []


def test_linspace():
    result = Interval.linspace(0, 1, 10)
    expected = Interval(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.1, 0.1))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_arange():
    result = Interval.arange(0.0, 1.0, 0.1)
    expected = Interval(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.1, 0.1))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    result = Interval.arange(0.0, 1.0, 0.3)
    expected = Interval(np.array([0.0, 0.3, 0.6, 0.9]), np.array([0.3, 0.6, 0.9, 1.0]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    result = Interval.arange(0.0, 1.0, 0.3, include_end=False)
    expected = Interval(np.array([0.0, 0.3, 0.6]), np.array([0.3, 0.6, 0.9]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_split():
    interval = Interval.linspace(0, 1, 10)

    # split into 3 sets using an int list
    result = interval.split([6, 2, 2])
    expected = [
        Interval.linspace(0, 0.6, 6),
        Interval.linspace(0.6, 0.8, 2),
        Interval.linspace(0.8, 1, 2),
    ]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        )

    # split into 2 sets using a float list
    result = interval.split([0.8, 0.2])
    expected = [Interval.linspace(0, 0.8, 8), Interval.linspace(0.8, 1, 2)]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        )

    # shuffle
    result = interval.split([0.5, 0.5], shuffle=True, random_seed=42)
    print(result[0].start, result[1].start)
    print(result[0].end, result[1].end)
    expected = [
        Interval(
            start=np.array([0.0, 0.3, 0.5, 0.6, 0.7]),
            end=np.array([0.1, 0.4, 0.6, 0.7, 0.8]),
        ),
        Interval(
            start=np.array([0.1, 0.2, 0.4, 0.8, 0.9]),
            end=np.array([0.2, 0.3, 0.5, 0.9, 1.0]),
        ),
    ]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        ), (
            f"result: {result[i].start} {result[i].end} "
            f"expected: {expected[i].start} {expected[i].end}"
        )


def test_and():
    op = lambda x, y: x & y

    I1 = Interval.from_list([(1.0, 2.3)])
    I2 = Interval.from_list([(1.7, 6.9)])
    Iexp = Interval.from_list([(1.7, 2.3)])
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list(
        [
            (1.0, 2.3),
            (3.0, 4.0),
            (5.6, 6.9),
            (8.0, 10.0),
        ]
    )
    I2 = Interval.from_list(
        [
            (1.7, 2.1),
            (3.2, 4.2),
            (5.4, 6.7),
            (8.2, 9.0),
            (9.5, 10.2),
        ]
    )
    Iexp = Interval.from_list(
        [
            (1.7, 2.1),
            (3.2, 4.0),
            (5.6, 6.7),
            (8.2, 9.0),
            (9.5, 10.0),
        ]
    )
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (1.7, 6.9)])
    I2 = Interval.from_list([(0.0, 1.0), (6.9, 8.4)])
    Iexp = Interval.from_list(
        [
            (0.0, 1.0),
        ]
    )
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)])
    I2 = Interval.from_list([(10.0, 11.0), (12.0, 13.0), (14.0, 15.0), (16.0, 17.0)])

    Iexp = Interval(np.array([]), np.array([]))
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (2.0, 3.0), (10.0, 11.0), (12.0, 13.0)])
    I2 = Interval.from_list([(1.0, 2.0), (3.0, 4.0), (11.0, 12.0), (13.0, 14.0)])
    Iexp = Interval(start=np.array([]), end=np.array([]))
    easy_symmetric_check(I1, I2, Iexp, op)


def test_or():
    op = lambda x, y: x | y

    I1 = Interval.from_list([(1.0, 2.3)])
    I2 = Interval.from_list([(1.7, 6.9)])
    Iexp = Interval.from_list([(1.0, 6.9)])
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list(
        [
            (1.0, 2.3),
            (3.0, 4.0),
            (5.6, 6.9),
            (8.0, 10.0),
        ]
    )
    I2 = Interval.from_list(
        [
            (1.7, 2.1),
            (3.2, 4.2),
            (5.4, 6.7),
            (8.2, 9.0),
            (9.5, 10.2),
        ]
    )
    Iexp = Interval.from_list(
        [
            (1.0, 2.3),
            (3.0, 4.2),
            (5.4, 6.9),
            (8.0, 10.2),
        ]
    )
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)])
    I2 = Interval.from_list([(10.0, 11.0), (12.0, 13.0), (14.0, 15.0), (16.0, 17.0)])

    Iexp = Interval.from_list(
        [
            (0.0, 1.0),
            (2.0, 3.0),
            (4.0, 5.0),
            (6.0, 7.0),
            (10.0, 11.0),
            (12.0, 13.0),
            (14.0, 15.0),
            (16.0, 17.0),
        ]
    )
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (2.0, 3.0), (10.0, 11.0), (12.0, 13.0)])
    I2 = Interval.from_list([(1.0, 2.0), (3.0, 4.0), (11.0, 12.0), (13.0, 14.0)])
    Iexp = Interval.from_list([(0.0, 4.0), (10.0, 14.0)])
    easy_symmetric_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (1.0, 2.0), (10.0, 11.0), (11.0, 12.0)])
    I2 = Interval.from_list([(2.0, 3.0), (3.0, 4.0), (12.0, 13.0), (13.0, 14.0)])
    Iexp = Interval.from_list([(0.0, 4.0), (10.0, 14.0)])
    easy_symmetric_check(I1, I2, Iexp, op)


def test_difference():
    op = lambda x, y: x.difference(y)

    I1 = Interval.from_list([(1.0, 2.3)])
    I2 = Interval.from_list([(1.7, 6.9)])
    Iexp = Interval.from_list([(1.0, 1.7)])
    easy_check(I1, I2, Iexp, op)

    I1 = Interval.from_list(
        [
            (1.0, 2.3),
            (3.0, 4.0),
            (5.6, 6.9),
            (8.0, 10.0),
            (12.0, 13.0),
        ]
    )
    I2 = Interval.from_list(
        [
            (1.7, 2.1),
            (3.2, 4.2),
            (5.4, 6.7),
            (8.2, 9.0),
            (9.5, 10.2),
        ]
    )
    Iexp = Interval.from_list(
        [
            (1.0, 1.7),
            (2.1, 2.3),
            (3.0, 3.2),
            (6.7, 6.9),
            (8.0, 8.2),
            (9.0, 9.5),
            (12.0, 13.0),
        ]
    )
    easy_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(1.0, 10.0)])
    I2 = Interval.from_list([(1.7, 6.9), (6.9, 8.4)])
    Iexp = Interval.from_list([(1.0, 1.7), (8.4, 10.0)])
    easy_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(2.0, 10.0)])
    I2 = Interval.from_list([(1.0, 20.0)])
    Iexp = Interval(np.array([]), np.array([]))
    easy_check(I1, I2, Iexp, op)

    I1 = Interval.from_list([(1.0, 3.0)])
    I2 = Interval.from_list([(3.0, 5.0)])
    easy_check(I1, I2, I1, op)

    I1 = Interval(1700.0, 1740.0)
    I2 = Interval(np.array([1716.0, 1722.5]), np.array([1722.0, 1740.0]))
    Iexp = Interval(np.array([1700.0, 1722.0]), np.array([1716.0, 1722.5]))
    easy_check(I1, I2, Iexp, op)

    I1 = Interval(1700.0, 1726.0)
    I2 = Interval(np.array([1716.0, 1722.0]), np.array([1722.0, 1730.0]))
    Iexp = Interval(1700.0, 1716.0)
    easy_check(I1, I2, Iexp, op)


# helper function
def easy_eq(interval1, interval2):
    return (
        len(interval1) == len(interval2)
        and np.allclose(interval1.start, interval2.start)
        and np.allclose(interval1.end, interval2.end)
    )


# helper function
def easy_str(interval):
    return str([(interval.start[i], interval.end[i]) for i in range(len(interval))])


# helper function
def easy_check(I1, I2, Iexp, op):
    assert easy_eq(op(I1, I2), Iexp), (
        f"Did not match \n"
        f"I1:     {easy_str(I1)}, \n"
        f"I2:     {easy_str(I2)}, \n"
        f"result: {easy_str(op(I1, I2))}, \n"
        f"expect: {easy_str(Iexp)}"
    )


# helper function
def easy_symmetric_check(I1, I2, Iexp, op):
    easy_check(I1, I2, Iexp, op)
    easy_check(I2, I1, Iexp, op)


def test_dilate():
    data = Interval(np.array([1.0, 5.0, 11.0]), np.array([2.0, 7.0, 12.0]))

    result = data.dilate(0.5)
    expected = Interval(np.array([0.5, 4.5, 10.5]), np.array([2.5, 7.5, 12.5]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    result = data.dilate(4.0)
    expected = Interval(np.array([-3.0, 3.5, 9.0]), np.array([3.5, 9.0, 16.0]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    result = data.dilate(4.0, max_len=2.0)
    expected = Interval(np.array([0.5, 5.0, 10.5]), np.array([2.5, 7.0, 12.5]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    data = Interval(np.array([]), np.array([]))

    result = data.dilate(0.5)
    expected = Interval(np.array([]), np.array([]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_is_dijoint_is_sorted():
    # Test sorted and disjoint interval
    sorted_disjoint = Interval(np.array([1.0, 3.0, 5.0]), np.array([2.0, 4.0, 6.0]))
    assert sorted_disjoint.is_disjoint() == True
    assert sorted_disjoint.is_sorted() == True

    # Test not sorted but disjoint interval
    unsorted_disjoint = Interval(np.array([3.0, 1.0, 5.0]), np.array([4.0, 2.0, 6.0]))
    assert unsorted_disjoint.is_disjoint() == True
    assert unsorted_disjoint.is_sorted() == False

    # Test not sorted and not disjoint interval
    unsorted_overlapping = Interval(
        np.array([3.0, 1.0, 2.0]), np.array([5.0, 4.0, 6.0])
    )
    assert unsorted_overlapping.is_disjoint() == False
    assert unsorted_overlapping.is_sorted() == False

    # Test sorted but not disjoint interval
    sorted_overlapping = Interval(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert sorted_overlapping.is_disjoint() == False
    assert sorted_overlapping.is_sorted() == True

    # Test edge case: interval of one point
    point_interval = Interval(np.array([1.0, 1.0, 3.0]), np.array([1.0, 3.0, 3.0]))
    assert point_interval.is_disjoint() == True
    assert point_interval.is_sorted() == True

    # Test edge case: mixed point and non-point intervals
    mixed_interval = Interval(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.5, 3.0, 5.0])
    )
    assert mixed_interval.is_disjoint() == True
    assert mixed_interval.is_sorted() == True

    # Test edge case: adjacent intervals
    adjacent_interval = Interval(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert adjacent_interval.is_disjoint() == True
    assert adjacent_interval.is_sorted() == True

    # Test interval object with one interval
    single_interval = Interval(np.array([1.0]), np.array([2.0]))
    assert single_interval.is_disjoint() == True
    assert single_interval.is_sorted() == True

    # Test empty interval
    empty_interval = Interval(np.array([]), np.array([]))
    assert empty_interval.is_disjoint() == True
    assert empty_interval.is_sorted() == True


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
