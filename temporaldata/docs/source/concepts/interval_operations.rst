Interval Operations
===================

The :obj:`Interval <temporaldata.Interval>` class provides several operations to manipulate and combine interval objects.

First, let's create some simple intervals:

.. code-block:: python

    from temporaldata import Interval
    import numpy as np

    # Create two intervals: [1, 8] and [12, 18]
    interval1 = Interval(start=np.array([1, 12]), end=np.array([8, 18]))
    
    # Create three intervals: [2, 5], [6, 10], and [10, 17]
    interval2 = Interval(start=np.array([2, 6, 10]), end=np.array([5, 10, 17]))

Typically for most operations that involve multiple :obj:`Interval <temporaldata.Interval>` objects, 
each :obj:`Interval <temporaldata.Interval>` object must be disjoint and sorted.

You can check these conditions using:

.. code-block:: python

    interval.is_disjoint()  # Returns True if intervals don't overlap
    interval.is_sorted()    # Returns True if intervals are sorted by start time



Intersection
------------

The intersection operation (``&``) returns a new :obj:`Interval <temporaldata.Interval>` 
containing only the overlapping time periods between two :obj:`Interval <temporaldata.Interval>` objects.

.. image:: /_static/intersection.png
   :width: 800
   :align: center
   :alt: Visualization of interval intersection operation

.. code-block:: python

    # Compute intersection
    intersection = interval1 & interval2
    
    # Result will contain [2, 5] and [12, 17] as these are the overlapping periods
    print(intersection.start)  # [2, 12]
    print(intersection.end)    # [5, 17]

Union
-----

The union operation (``|``) returns a new :obj:`Interval <temporaldata.Interval>` 
containing the union of all intervals in both objects.

.. image:: /_static/union.png
   :width: 800
   :align: center
   :alt: Visualization of interval union operation


.. code-block:: python

    # Compute union
    union = interval1 | interval2
    
    # Result will contain [1, 10] and [10, 18]
    print(union.start)  # [1, 10]
    print(union.end)    # [10, 18]

Difference
----------

The difference operation (``.difference()``) returns a new :obj:`Interval <temporaldata.Interval>` containing time periods that are in the first interval but not in the second interval.

.. image:: /_static/difference.png
   :width: 800
   :align: center
   :alt: Visualization of interval difference operation


.. code-block:: python

    # Create two intervals
    a = Interval(start=[1, 12], end=[8, 18])
    b = Interval(start=[2, 6, 10], end=[5, 10, 17])
    
    # Compute difference
    difference = a.difference(b)
    
    # Results in intervals: [1, 2], [5, 6], [17, 18]
    print(difference.start)  # [1, 5, 17]
    print(difference.end)    # [2, 6, 18]

Dilation
--------

The dilation operation (``.dilate()``) expands each interval by a specified amount on both sides. This is useful for creating buffer periods around intervals or merging nearby intervals.

.. image:: /_static/dilate.png
   :width: 800
   :align: center
   :alt: Visualization of interval dilation operation

.. code-block:: python

    # Create an interval
    interval = Interval(start=[1, 5], end=[2, 6])
    
    # Dilate by 0.5 on each side
    dilated = interval.dilate(0.5)
    
    print(dilated.start)  # [0.5, 4.5]
    print(dilated.end)    # [2.5, 6.5]

The dilation operation is particularly useful when you need to:
- Create buffer periods around events
- Account for uncertainty in interval boundaries
- Merge intervals that are close together

Coalescing
----------

The coalesce operation (``.coalesce()``) merges overlapping or touching intervals into single continuous intervals. This is useful for simplifying interval sets and removing gaps below a certain threshold.

.. image:: /_static/coalesce.png
   :width: 800
   :align: center
   :alt: Visualization of interval coalesce operation

.. code-block:: python

    # Create intervals with small gaps
    interval = Interval(
        start=[1, 2.1, 4, 8],
        end=[2, 3, 5, 9]
    )
    
    # Coalesce intervals that are within 0.2 of each other
    coalesced = interval.coalesce(0.2)
    
    # [1-3] and [4-5] merged, [8-9] unchanged
    print(coalesced.start)  # [1, 4, 8]
    print(coalesced.end)    # [3, 5, 9]

The coalesce operation is useful for:
- Cleaning up noisy interval data
- Merging intervals that are effectively continuous
- Simplifying interval representations

You can combine coalesce with dilate to merge intervals within a certain distance:

.. code-block:: python

    # Merge intervals within distance 0.5
    interval = Interval(start=[1, 3], end=[2, 4])
    merged = interval.dilate(0.25).coalesce(0)
    
    print(merged.start)  # [0.75]  
    print(merged.end)    # [4.25]


There are multiple edge cases that can occur when performing interval operations. For more details, see :ref:`advanced_interval_operations`.