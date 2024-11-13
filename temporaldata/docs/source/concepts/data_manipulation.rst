Data manipulation
-----------------

The **temporaldata** package provides several ways for manipulating data objects.

Time-based Slicing
~~~~~~~~~~~~~~~~~~

All temporal data objects, :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`, 
:obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>`, :obj:`Interval <temporaldata.Interval>`, 
as well as :obj:`Data <temporaldata.Data>` objects support time-based slicing through the ``.slice()`` method:

.. code-block:: python

    # slice data between 1.0 and 3.0 seconds
    sliced_data = data.slice(start=1.0, end=3.0)

The returned slice is a new object of the same type as the original data object. The timestamps
are reset relative to the new start time. To keep the original timestamps, set ``reset_origin=False``:

.. code-block:: python

    sliced_data = data.slice(start=1.0, end=3.0, reset_origin=False)


For point-based objects, :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>` and :obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>`, the timestamps that are in [start, end) are included in the slice.

For interval-based objects, :obj:`Interval <temporaldata.Interval>`, the intervals that overlap with the slice window are kept.

For :obj:`Data <temporaldata.Data>` objects, the slice operation propagates to all nested data objects.

Under the hood, the slicing operation is performed using a hybrid of binary search and a 
kd-tree algorithm on the timestamps, making it very fast.


Masking Operations
~~~~~~~~~~~~~~~~~~

Another way of manipulating data is through boolean masks. While slicing selects data 
based on time windows, masking allows selecting data points or intervals based on other attributes. 
For example, you may want to select only spikes from certain neurons, or intervals with specific properties.

All data objects support masking through the ``.select_by_mask()`` method. 
The mask must be a 1D boolean array matching the length of the first dimension of the data arrays:

.. code-block:: python

    # Create a boolean mask
    mask = data.amplitude > 0.5

    # Apply mask to select data
    masked_data = data.select_by_mask(mask)

The mask must be a 1D boolean array matching the length of the first dimension of the data.
