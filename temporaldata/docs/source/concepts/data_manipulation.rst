Data manipulation
-----------------

The temporaldata package provides several ways to manipulate data objects, including slicing by time intervals and selecting data using boolean masks.

Time-based Slicing
~~~~~~~~~~~~~~~~~

All time-series objects (IrregularTimeSeries, RegularTimeSeries, Interval) and Data objects support slicing operations through the ``slice()`` method:

.. code-block:: python

    # Slice data between 1.0 and 3.0 seconds
    sliced_data = data.slice(start=1.0, end=3.0)

    # By default, times are reset relative to the new start time
    # To keep original timestamps, set reset_origin=False
    sliced_data = data.slice(start=1.0, end=3.0, reset_origin=False)

Under the Hood
^^^^^^^^^^^^^

When slicing is performed, the following happens:

1. For IrregularTimeSeries:
   - Binary search is used to find indices corresponding to the time window
   - If ``reset_origin=True``, timestamps are shifted to start at 0
   - For lazy objects (LazyIrregularTimeSeries), the slice operation is stored and only applied when data is accessed

2. For Interval objects:
   - Intervals that overlap with the slice window are kept
   - Start/end times are adjusted relative to the new origin if ``reset_origin=True``

3. For Data objects:
   - The slice operation propagates to all nested time-series objects
   - The domain is updated to reflect the new time window
   - The absolute start time is tracked for reference

Masking Operations
~~~~~~~~~~~~~~~~

Boolean masks can be applied to select specific data points using ``select_by_mask()``:

.. code-block:: python

    # Create a boolean mask
    mask = data.spikes.timestamps < 2.0

    # Apply mask to select data
    masked_data = data.select_by_mask(mask)

The mask must be a 1D boolean array matching the length of the first dimension of the data.

Lazy Operations
^^^^^^^^^^^^^

For large datasets stored in HDF5 files, the package provides lazy versions of data objects (LazyArrayDict, LazyIrregularTimeSeries, LazyInterval). These objects:

- Defer actual data loading until attributes are accessed
- Store operations (slicing, masking) and apply them only when needed
- Combine multiple operations efficiently

For example:

.. code-block:: python

    # These operations don't load data immediately
    lazy_data = lazy_data.slice(0, 2.0)
    lazy_data = lazy_data.select_by_mask(mask)

    # Data is loaded and operations are applied only when accessed
    timestamps = lazy_data.timestamps  # Now data is loaded

Split Masks
~~~~~~~~~~

For machine learning applications, you can create split masks to separate data into training/validation/test sets:

.. code-block:: python

    # Create a split mask using an interval
    interval = Interval(0, 5.0)
    data.add_split_mask("train", interval)

    # Check for data leakage between splits
    data._check_for_data_leakage("train")

This ensures proper data separation and helps prevent data leakage in machine learning pipelines.