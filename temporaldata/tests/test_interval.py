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


def test_and_edge_cases():
    op = lambda x, y: x & y
    empty = Interval(np.array([]), np.array([]))

    # both empty
    easy_symmetric_check(empty, empty, empty, op)

    # one empty
    I1 = Interval.from_list([(1.0, 2.0)])
    easy_symmetric_check(I1, empty, empty, op)

    # full containment: one interval fully inside another
    I1 = Interval.from_list([(0.0, 10.0)])
    I2 = Interval.from_list([(3.0, 5.0)])
    Iexp = Interval.from_list([(3.0, 5.0)])
    easy_symmetric_check(I1, I2, Iexp, op)

    # identical intervals
    I1 = Interval.from_list([(1.0, 3.0), (5.0, 7.0)])
    easy_symmetric_check(I1, I1, I1, op)

    # single-segment with many-segment: Data.slice() hot path
    I1 = Interval.from_list([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])
    I2 = Interval.from_list([(1.5, 4.5)])
    Iexp = Interval.from_list([(2.0, 3.0), (4.0, 4.5)])
    easy_check(I1, I2, Iexp, op)
    Iexp_rev = Interval.from_list([(2.0, 3.0), (4.0, 4.5)])
    easy_check(I2, I1, Iexp_rev, op)

    # point interval intersected with a containing interval
    point = Interval(np.array([2.0]), np.array([2.0]))
    other = Interval(np.array([1.0]), np.array([3.0]))
    Iexp = Interval(np.array([2.0]), np.array([2.0]))
    easy_symmetric_check(point, other, Iexp, op)


def test_or_edge_cases():
    op = lambda x, y: x | y
    empty = Interval(np.array([]), np.array([]))

    # both empty
    easy_symmetric_check(empty, empty, empty, op)

    # one empty
    I1 = Interval.from_list([(1.0, 2.0)])
    easy_symmetric_check(I1, empty, I1, op)

    # full containment
    I1 = Interval.from_list([(0.0, 10.0)])
    I2 = Interval.from_list([(3.0, 5.0)])
    Iexp = Interval.from_list([(0.0, 10.0)])
    easy_symmetric_check(I1, I2, Iexp, op)

    # identical intervals
    I1 = Interval.from_list([(1.0, 3.0), (5.0, 7.0)])
    Iexp = Interval.from_list([(1.0, 3.0), (5.0, 7.0)])
    easy_symmetric_check(I1, I1, Iexp, op)

    # multiple containments in a row
    I1 = Interval.from_list([(0.0, 20.0)])
    I2 = Interval.from_list([(1.0, 3.0), (5.0, 7.0), (9.0, 11.0)])
    Iexp = Interval.from_list([(0.0, 20.0)])
    easy_symmetric_check(I1, I2, Iexp, op)

    # adjacent intervals with empty operand must still merge
    I1 = Interval.from_list([(0.0, 1.0), (1.0, 2.0)])
    Iexp = Interval.from_list([(0.0, 2.0)])
    easy_symmetric_check(I1, empty, Iexp, op)

    I1 = Interval.from_list([(0.0, 1.0), (1.0, 2.0), (5.0, 6.0), (6.0, 7.0)])
    Iexp = Interval.from_list([(0.0, 2.0), (5.0, 7.0)])
    easy_symmetric_check(I1, empty, Iexp, op)

    # adjacent self-union (doc example: adjacent | adjacent → merged)
    adjacent = Interval(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    Iexp = Interval.from_list([(1.0, 3.0)])
    easy_symmetric_check(adjacent, adjacent, Iexp, op)

    # point interval union
    point = Interval(np.array([2.0]), np.array([2.0]))
    other = Interval(np.array([1.0]), np.array([3.0]))
    Iexp = Interval.from_list([(1.0, 3.0)])
    easy_symmetric_check(point, other, Iexp, op)


def test_overlapping_input_raises():
    """Operations on non-disjoint or unsorted intervals must raise ValueError."""
    valid = Interval.from_list([(1.0, 2.0)])
    overlapping = Interval(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    unsorted = Interval(np.array([3.0, 1.0]), np.array([4.0, 2.0]))

    for bad in [overlapping, unsorted]:
        with pytest.raises(ValueError):
            bad | valid
        with pytest.raises(ValueError):
            valid | bad
        with pytest.raises(ValueError):
            bad & valid
        with pytest.raises(ValueError):
            valid & bad
        with pytest.raises(ValueError):
            bad.difference(valid)
        with pytest.raises(ValueError):
            valid.difference(bad)


def test_difference_edge_cases():
    op = lambda x, y: x.difference(y)
    empty = Interval(np.array([]), np.array([]))

    # both empty
    easy_check(empty, empty, empty, op)

    # self empty
    I1 = Interval.from_list([(1.0, 2.0)])
    easy_check(empty, I1, empty, op)

    # other empty
    I1 = Interval.from_list([(1.0, 2.0)])
    easy_check(I1, empty, I1, op)

    # self fully contains other
    I1 = Interval.from_list([(0.0, 10.0)])
    I2 = Interval.from_list([(3.0, 5.0)])
    Iexp = Interval.from_list([(0.0, 3.0), (5.0, 10.0)])
    easy_check(I1, I2, Iexp, op)

    # identical intervals: difference should be empty
    I1 = Interval.from_list([(1.0, 3.0), (5.0, 7.0)])
    easy_check(I1, I1, empty, op)

    # no overlap
    I1 = Interval.from_list([(0.0, 1.0)])
    I2 = Interval.from_list([(5.0, 6.0)])
    easy_check(I1, I2, I1, op)


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


class TestIntervalCoalesce:
    def test_contiguous_intervals(self):
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

    def test_mixed_overlapping_and_separated(self):
        data = Interval(
            start=np.array([0.0, 1.0, 2.0, 4.0, 4.5, 5.0, 10.0]),
            end=np.array([0.5, 2.0, 2.5, 4.5, 5.0, 6.0, 11.0]),
        )

        coalesced_data = data.coalesce()
        assert len(coalesced_data) == 4
        assert np.allclose(coalesced_data.start, np.array([0.0, 1.0, 4.0, 10.0]))
        assert np.allclose(coalesced_data.end, np.array([0.5, 2.5, 6.0, 11.0]))

    def test_already_disjoint(self):
        data = Interval(
            start=np.array([0.0, 5.0, 10.0]),
            end=np.array([1.0, 6.0, 11.0]),
        )

        coalesced_data = data.coalesce()
        assert len(coalesced_data) == 3
        assert np.allclose(coalesced_data.start, np.array([0.0, 5.0, 10.0]))
        assert np.allclose(coalesced_data.end, np.array([1.0, 6.0, 11.0]))

    def test_empty_interval(self):
        data = Interval(start=np.array([]), end=np.array([]))
        coalesced_data = data.coalesce()
        assert len(coalesced_data) == 0
        assert np.array_equal(coalesced_data.start, np.array([]))
        assert np.array_equal(coalesced_data.end, np.array([]))

    def test_custom_eps(self):
        data = Interval(
            start=np.array([0.0, 1.001, 5.0]),
            end=np.array([1.0, 2.0, 6.0]),
        )

        # default eps=1e-6: gap of 0.001 is too large, no coalescing
        coalesced_default = data.coalesce()
        assert len(coalesced_default) == 3

        # eps=0.01: gap of 0.001 is within threshold, first two coalesce
        coalesced_custom = data.coalesce(eps=0.01)
        assert len(coalesced_custom) == 2
        assert np.allclose(coalesced_custom.start, np.array([0.0, 5.0]))
        assert np.allclose(coalesced_custom.end, np.array([2.0, 6.0]))

    def test_gap_exactly_at_eps(self):
        eps = 0.01
        data = Interval(
            start=np.array([0.0, 1.0 + eps]),
            end=np.array([1.0, 2.0]),
        )

        # gap == eps is NOT less than eps, so intervals stay separate
        coalesced_data = data.coalesce(eps=eps)
        assert len(coalesced_data) == 2
        assert np.allclose(coalesced_data.start, np.array([0.0, 1.0 + eps]))
        assert np.allclose(coalesced_data.end, np.array([1.0, 2.0]))

    def test_gap_just_under_eps(self):
        eps = 0.01
        gap = eps - 1e-10
        data = Interval(
            start=np.array([0.0, 1.0 + gap]),
            end=np.array([1.0, 2.0]),
        )

        # gap < eps, so intervals coalesce
        coalesced_data = data.coalesce(eps=eps)
        assert len(coalesced_data) == 1
        assert np.allclose(coalesced_data.start, np.array([0.0]))
        assert np.allclose(coalesced_data.end, np.array([2.0]))

    def test_negative_eps_raises(self):
        data = Interval(
            start=np.array([0.0, 2.0]),
            end=np.array([1.0, 3.0]),
        )
        with pytest.raises(ValueError, match="eps must be non-negative"):
            data.coalesce(eps=-0.1)


def test_subdivide():
    interval = Interval(start=np.array([0.0]), end=np.array([10.0]))
    result = interval.subdivide(2.0)
    expected = Interval(
        start=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        end=np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(
        start=np.array([0.0, 20.0]),
        end=np.array([10.0, 30.0]),
    )
    result = interval.subdivide(2.5)
    expected = Interval(
        start=np.array([0.0, 2.5, 5.0, 7.5, 20.0, 22.5, 25.0, 27.5]),
        end=np.array([2.5, 5.0, 7.5, 10.0, 22.5, 25.0, 27.5, 30.0]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0]), end=np.array([10.5]))
    result = interval.subdivide(3.0)
    expected = Interval(
        start=np.array([0.0, 3.0, 6.0, 9.0]),
        end=np.array([3.0, 6.0, 9.0, 10.5]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0]), end=np.array([0.5]))
    result = interval.subdivide(2.0)
    expected = Interval(start=np.array([0.0]), end=np.array([0.5]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([]), end=np.array([]))
    result = interval.subdivide(2.0)
    expected = Interval(start=np.array([]), end=np.array([]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0]), end=np.array([2.0]))
    result = interval.subdivide(2.0)
    expected = Interval(start=np.array([0.0]), end=np.array([2.0]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_subdivide_drop_short():
    interval = Interval(start=np.array([0.0]), end=np.array([10.5]))
    result = interval.subdivide(3.0, drop_short=False)
    expected = Interval(
        start=np.array([0.0, 3.0, 6.0, 9.0]),
        end=np.array([3.0, 6.0, 9.0, 10.5]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0]), end=np.array([10.5]))
    result = interval.subdivide(3.0, drop_short=True)
    expected = Interval(
        start=np.array([0.0, 3.0, 6.0]),
        end=np.array([3.0, 6.0, 9.0]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0, 20.0]), end=np.array([10.5, 25.0]))
    result = interval.subdivide(3.0, drop_short=True)
    expected = Interval(
        start=np.array([0.0, 3.0, 6.0, 20.0]),
        end=np.array([3.0, 6.0, 9.0, 23.0]),
    )
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    interval = Interval(start=np.array([0.0]), end=np.array([0.5]))
    result = interval.subdivide(2.0, drop_short=True)
    expected = Interval(start=np.array([]), end=np.array([]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_subdivide_attributes_preservation():
    interval = Interval(
        start=np.array([0.0, 20.0]),
        end=np.array([6.0, 26.0]),
        trial_id=np.array([1, 2]),
        condition=np.array(["A", "B"]),
        timekeys=["start", "end"],
    )
    result = interval.subdivide(2.0)

    expected_start = np.array([0.0, 2.0, 4.0, 20.0, 22.0, 24.0])
    expected_end = np.array([2.0, 4.0, 6.0, 22.0, 24.0, 26.0])
    expected_trial_id = np.array([1, 1, 1, 2, 2, 2])
    expected_condition = np.array(["A", "A", "A", "B", "B", "B"])

    assert np.allclose(result.start, expected_start)
    assert np.allclose(result.end, expected_end)
    assert np.array_equal(result.trial_id, expected_trial_id)
    assert np.array_equal(result.condition, expected_condition)

    interval = Interval(
        start=np.array([0.0]),
        end=np.array([5.5]),
        session_id=np.array([42]),
        timekeys=["start", "end"],
    )
    result = interval.subdivide(2.0, drop_short=False)

    expected_start = np.array([0.0, 2.0, 4.0])
    expected_end = np.array([2.0, 4.0, 5.5])
    expected_session_id = np.array([42, 42, 42])

    assert np.allclose(result.start, expected_start)
    assert np.allclose(result.end, expected_end)
    assert np.array_equal(result.session_id, expected_session_id)

    interval = Interval(
        start=np.array([0.0]),
        end=np.array([5.5]),
        session_id=np.array([42]),
        timekeys=["start", "end"],
    )
    result = interval.subdivide(2.0, drop_short=True)

    expected_start = np.array([0.0, 2.0])
    expected_end = np.array([2.0, 4.0])
    expected_session_id = np.array([42, 42])

    assert np.allclose(result.start, expected_start)
    assert np.allclose(result.end, expected_end)
    assert np.array_equal(result.session_id, expected_session_id)


def test_subdivide_empty_interval_preserves_attributes():
    interval = Interval(
        start=np.array([]),
        end=np.array([]),
        trial_id=np.array([], dtype=int),
        condition=np.array([], dtype=str),
        timekeys=["start", "end"],
    )
    result = interval.subdivide(2.0)

    assert len(result) == 0
    assert np.allclose(result.start, np.array([]))
    assert np.allclose(result.end, np.array([]))
    assert "trial_id" in result.keys()
    assert "condition" in result.keys()
    assert len(result.trial_id) == 0
    assert len(result.condition) == 0
    assert result.timekeys() == ["start", "end"]


def test_subdivide_timekeys_preservation():
    interval = Interval(
        start=np.array([0.0, 10.0]),
        end=np.array([4.0, 14.0]),
        go_cue_time=np.array([1.0, 11.0]),
        timekeys=["start", "end", "go_cue_time"],
    )
    result = interval.subdivide(2.0)

    expected_start = np.array([0.0, 2.0, 10.0, 12.0])
    expected_end = np.array([2.0, 4.0, 12.0, 14.0])
    expected_go_cue_time = np.array([1.0, 1.0, 11.0, 11.0])

    assert np.allclose(result.start, expected_start)
    assert np.allclose(result.end, expected_end)
    assert np.array_equal(result.go_cue_time, expected_go_cue_time)
    assert "go_cue_time" in result.timekeys()


class TestPointIntervals:
    """Point intervals (start == end) must be handled correctly."""

    # -- union ---------------------------------------------------------

    def test_union_point_or_empty_is_identity(self):
        point = Interval(np.array([5.0]), np.array([5.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(point, empty, point, lambda x, y: x | y)

    def test_union_two_distinct_points(self):
        p1 = Interval(np.array([1.0]), np.array([1.0]))
        p2 = Interval(np.array([3.0]), np.array([3.0]))
        result = p1 | p2
        assert len(result) == 2
        assert np.allclose(result.start, [1.0, 3.0])
        assert np.allclose(result.end, [1.0, 3.0])

    def test_union_point_absorbed_by_range(self):
        point = Interval(np.array([2.0]), np.array([2.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        expected = Interval.from_list([(1.0, 3.0)])
        easy_symmetric_check(point, rng, expected, lambda x, y: x | y)

    def test_union_duplicate_points_merge(self):
        p = Interval(np.array([2.0]), np.array([2.0]))
        result = p | p
        assert len(result) == 1
        assert result.start[0] == 2.0 and result.end[0] == 2.0

    # -- intersection --------------------------------------------------

    def test_intersect_point_with_containing_range(self):
        point = Interval(np.array([2.0]), np.array([2.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        expected = Interval(np.array([2.0]), np.array([2.0]))
        easy_symmetric_check(point, rng, expected, lambda x, y: x & y)

    def test_intersect_point_with_point_same(self):
        p = Interval(np.array([2.0]), np.array([2.0]))
        expected = Interval(np.array([2.0]), np.array([2.0]))
        easy_symmetric_check(p, p, expected, lambda x, y: x & y)

    def test_intersect_point_with_point_different(self):
        p1 = Interval(np.array([1.0]), np.array([1.0]))
        p2 = Interval(np.array([3.0]), np.array([3.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(p1, p2, empty, lambda x, y: x & y)

    def test_intersect_point_outside_range(self):
        point = Interval(np.array([5.0]), np.array([5.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(point, rng, empty, lambda x, y: x & y)

    def test_intersect_point_at_range_boundary(self):
        point = Interval(np.array([3.0]), np.array([3.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        expected = Interval(np.array([3.0]), np.array([3.0]))
        easy_symmetric_check(point, rng, expected, lambda x, y: x & y)

    def test_intersect_point_at_range_start(self):
        point = Interval(np.array([1.0]), np.array([1.0]))
        rng = Interval(np.array([1.0]), np.array([5.0]))
        expected = Interval(np.array([1.0]), np.array([1.0]))
        easy_symmetric_check(point, rng, expected, lambda x, y: x & y)

    def test_intersect_point_with_multi_segment(self):
        """Exact repro from issue #110: point & multi-segment must be commutative."""
        x = Interval(0, 1) | Interval(3, 10)
        y = Interval(5, 5)
        expected = Interval(np.array([5.0]), np.array([5.0]))
        easy_symmetric_check(x, y, expected, lambda a, b: a & b)

    def test_intersect_point_with_multi_segment_at_boundary(self):
        """Point at the exact end of a segment in a multi-segment interval."""
        x = Interval.from_list([(0.0, 5.0), (7.0, 10.0)])
        point_at_end = Interval(np.array([5.0]), np.array([5.0]))
        expected = Interval(np.array([5.0]), np.array([5.0]))
        easy_symmetric_check(x, point_at_end, expected, lambda a, b: a & b)

    def test_intersect_point_with_multi_segment_at_start(self):
        """Point at the exact start of a segment in a multi-segment interval."""
        x = Interval.from_list([(0.0, 5.0), (7.0, 10.0)])
        point_at_start = Interval(np.array([7.0]), np.array([7.0]))
        expected = Interval(np.array([7.0]), np.array([7.0]))
        easy_symmetric_check(x, point_at_start, expected, lambda a, b: a & b)

    def test_intersect_point_outside_multi_segment(self):
        """Point in a gap of a multi-segment interval → empty."""
        x = Interval.from_list([(0.0, 3.0), (7.0, 10.0)])
        point_in_gap = Interval(np.array([5.0]), np.array([5.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(x, point_in_gap, empty, lambda a, b: a & b)

    def test_intersect_multiple_points_with_range(self):
        """Multiple point intervals intersected with a range."""
        points = Interval(np.array([1.0, 5.0, 9.0]), np.array([1.0, 5.0, 9.0]))
        rng = Interval.from_list([(0.0, 2.0), (4.0, 6.0)])
        expected = Interval(np.array([1.0, 5.0]), np.array([1.0, 5.0]))
        easy_symmetric_check(points, rng, expected, lambda a, b: a & b)

    def test_intersect_touching_segments_no_false_point(self):
        """Boundary-touching non-point intervals must NOT produce a false point."""
        I1 = Interval.from_list([(0.0, 5.0)])
        I2 = Interval.from_list([(5.0, 10.0)])
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(I1, I2, empty, lambda a, b: a & b)

    def test_intersect_mixed_points_and_ranges(self):
        """Both operands contain a mix of point and non-point intervals."""
        I1 = Interval(np.array([1.0, 3.0, 5.0]), np.array([2.0, 3.0, 7.0]))
        I2 = Interval(np.array([0.0, 3.0, 6.0]), np.array([1.5, 4.0, 6.0]))
        expected = Interval(np.array([1.0, 3.0, 6.0]), np.array([1.5, 3.0, 6.0]))
        easy_symmetric_check(I1, I2, expected, lambda a, b: a & b)

    def test_intersect_multiple_points_vs_multiple_points(self):
        """Both operands are all point intervals."""
        p1 = Interval(np.array([1.0, 3.0, 5.0]), np.array([1.0, 3.0, 5.0]))
        p2 = Interval(np.array([2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0]))
        expected = Interval(np.array([3.0, 5.0]), np.array([3.0, 5.0]))
        easy_symmetric_check(p1, p2, expected, lambda a, b: a & b)

    def test_intersect_one_range_containing_many(self):
        """One large interval fully contains many segments from the other."""
        big = Interval.from_list([(0.0, 100.0)])
        many = Interval.from_list(
            [(1.0, 2.0), (10.0, 20.0), (50.0, 60.0), (90.0, 95.0)]
        )
        easy_symmetric_check(big, many, many, lambda a, b: a & b)

    def test_intersect_asymmetric_segment_counts(self):
        """Highly asymmetric: 1 segment vs many, with partial overlaps."""
        one = Interval.from_list([(2.5, 7.5)])
        many = Interval.from_list(
            [
                (0.0, 1.0),
                (2.0, 3.0),
                (4.0, 5.0),
                (6.0, 7.0),
                (8.0, 9.0),
            ]
        )
        expected = Interval.from_list([(2.5, 3.0), (4.0, 5.0), (6.0, 7.0)])
        easy_symmetric_check(one, many, expected, lambda a, b: a & b)

    def test_intersect_point_between_segments(self):
        """Point sitting in the gap between two segments → empty."""
        segments = Interval.from_list([(0.0, 3.0), (5.0, 8.0)])
        point = Interval(np.array([4.0]), np.array([4.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(segments, point, empty, lambda a, b: a & b)

    def test_intersect_multi_segment_touching_no_false_point(self):
        """Multi-segment boundaries that touch must not create false points."""
        I1 = Interval.from_list([(0.0, 3.0), (6.0, 9.0)])
        I2 = Interval.from_list([(3.0, 6.0), (9.0, 12.0)])
        empty = Interval(np.array([]), np.array([]))
        easy_symmetric_check(I1, I2, empty, lambda a, b: a & b)

    # -- difference ----------------------------------------------------

    def test_difference_point_minus_covering_range(self):
        point = Interval(np.array([2.0]), np.array([2.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_check(point, rng, empty, lambda x, y: x.difference(y))

    def test_difference_point_minus_same_point(self):
        p = Interval(np.array([2.0]), np.array([2.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_check(p, p, empty, lambda x, y: x.difference(y))

    def test_difference_point_minus_disjoint(self):
        point = Interval(np.array([5.0]), np.array([5.0]))
        rng = Interval(np.array([1.0]), np.array([3.0]))
        easy_check(point, rng, point, lambda x, y: x.difference(y))

    def test_difference_range_minus_inner_point(self):
        rng = Interval(np.array([1.0]), np.array([3.0]))
        point = Interval(np.array([2.0]), np.array([2.0]))
        expected = Interval.from_list([(1.0, 2.0), (2.0, 3.0)])
        easy_check(rng, point, expected, lambda x, y: x.difference(y))

    def test_difference_point_minus_empty(self):
        point = Interval(np.array([2.0]), np.array([2.0]))
        empty = Interval(np.array([]), np.array([]))
        easy_check(point, empty, point, lambda x, y: x.difference(y))

    def test_difference_multiple_points_minus_some(self):
        points = Interval(np.array([1.0, 3.0, 5.0]), np.array([1.0, 3.0, 5.0]))
        other = Interval(np.array([3.0]), np.array([3.0]))
        result = points.difference(other)
        assert len(result) == 2
        assert np.allclose(result.start, [1.0, 5.0])
        assert np.allclose(result.end, [1.0, 5.0])
