.. _advanced_interval_operations:

Interval Operations - Edge Cases
================================


Let's explore how interval operations handle various edge cases:

Adjacent Intervals
~~~~~~~~~~~~~~~~~~

When intervals are exactly adjacent (end of one equals start of another), the union operation will merge them into a single interval:

.. code-block:: python

    # Create adjacent intervals [1, 2] and [2, 3]
    adjacent = Interval(start=[1, 2], end=[2, 3])
    
    # Union will merge them into [1, 3]
    merged = adjacent | adjacent
    print(merged.start)  # [1]
    print(merged.end)    # [3]

Point Intervals
~~~~~~~~~~~~~~~

Intervals where start equals end (zero duration) are handled gracefully:

.. code-block:: python

    # Create point interval [2, 2]
    point = Interval(start=[2], end=[2])
    
    # Intersection with another interval containing that point
    other = Interval(start=[1], end=[3])
    intersection = point & other
    print(intersection.start)  # [2]
    print(intersection.end)    # [2]

Empty Intervals
~~~~~~~~~~~~~~~

Operations with empty intervals (no time periods) return appropriate results:

.. code-block:: python

    # Create an empty interval
    empty = Interval(start=[], end=[])
    
    # Any intersection with an empty interval is empty
    some_interval = Interval(start=[1], end=[2])
    intersection = empty & some_interval
    print(len(intersection.start))  # 0
    
    # Union with an empty interval returns the non-empty interval
    union = empty | some_interval
    print(union.start)  # [1]
    print(union.end)    # [2]

Overlapping Input Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you try to perform operations on intervals that aren't disjoint or sorted, the operations will raise a ValueError:

.. code-block:: python

    # Create overlapping intervals [1, 3] and [2, 4]
    overlapping = Interval(start=[1, 2], end=[3, 4])
    
    try:
        # This will raise a ValueError
        overlapping & some_interval
    except ValueError as e:
        print(e)  # "left Interval object must be disjoint."
    
    # Fix by making intervals disjoint first
    fixed = overlapping | overlapping  # Merges overlapping intervals
    result = fixed & some_interval     # Now works correctly

Exact Matches
~~~~~~~~~~~~~

When interval boundaries exactly match, both intersection and union handle them appropriately:

.. code-block:: python

    # Create two identical intervals
    a = Interval(start=[1, 5], end=[3, 7])
    b = Interval(start=[1, 5], end=[3, 7])
    
    # Intersection returns the same intervals
    intersection = a & b
    print(intersection.start)  # [1, 5]
    print(intersection.end)    # [3, 7]
    
    # Union also returns the same intervals
    union = a | b
    print(union.start)  # [1, 5]
    print(union.end)    # [3, 7]

These edge cases are important to consider when working with intervals, especially in data processing pipelines where unexpected interval patterns might occur.
