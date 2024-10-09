import pytest
import numpy as np
from temporaldata import Interval


# def test_indexing():
#     # same code but with numpy arrays
#     interval = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

#     # Test single index
#     result = interval[0]
#     expected = Interval(np.array([0]), np.array([1]))
#     assert np.allclose(result.start, expected.start) and np.allclose(
#         result.end, expected.end
#     )

#     # Test slice indexing
#     result = interval[0:2]
#     expected = Interval(np.array([0, 1]), np.array([1, 2]))
#     assert np.allclose(result.start, expected.start) and np.allclose(
#         result.end, expected.end
#     )

#     # Test list indexing
#     result = interval[[0, 2]]
#     expected = Interval(np.array([0, 2]), np.array([1, 3]))
#     assert np.allclose(result.start, expected.start) and np.allclose(
#         result.end, expected.end
#     )

#     # Test boolean indexing
#     result = interval[[True, False, True]]
#     expected = Interval(np.array([0, 2]), np.array([1, 3]))
#     assert np.allclose(result.start, expected.start) and np.allclose(
#         result.end, expected.end
#     )


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
