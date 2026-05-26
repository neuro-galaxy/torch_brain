Data manipulation
-----------------

The **temporaldata** package provides several ways for manipulating data objects.

Time-based Slicing
~~~~~~~~~~~~~~~~~~

All temporal data objects, :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`, 
:obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>`, :obj:`Interval <temporaldata.Interval>`, 
as well as :obj:`Data <temporaldata.Data>` objects support time-based slicing through the ``.slice()`` method:

.. code-block:: python

    import numpy as np
    from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries

    # Create a complex data object
    user_session = Data(
        clicks=IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.1]),
            position=np.array([[100, 200], [150, 300], [200, 150]]),
            domain=Interval(start=0, end=4),
        ),
        sensor=RegularTimeSeries(
            sampling_rate=100,
            accelerometer=np.random.randn(400, 3),
            domain=Interval(start=0, end=4),
        ),
        activities=Interval(
            start=np.array([0.0, 2.0, 4.5]),
            end=np.array([1.0, 3.0, 6.0]),
            activity=np.array(["typing", "scrolling", "typing"]),
        ),
        user_id="user123",
        device="laptop",
        domain="auto",
    )


    # slice data between 1.0 and 2.5 seconds
    sliced_user_session = user_session.slice(1, 2.5)

    # IrregularTimeSeries
    print(sliced_user_session.clicks.timestamps)  # [0.2, 1.3]
    print(sliced_user_session.clicks.position)  # [[100, 200], [150, 300]]
    print(sliced_user_session.clicks.domain.start)  # [0.]
    print(sliced_user_session.clicks.domain.end)  # [1.5]

    # RegularTimeSeries
    print(sliced_user_session.sensor.accelerometer.shape)  # (150, 3)
    print(sliced_user_session.sensor.domain.start)  # [-1.]
    print(sliced_user_session.sensor.domain.end)  # [3.]

    # Interval
    print(sliced_user_session.activities.activity)  # ['scrolling']
    print(sliced_user_session.activities.start)  # [1.]
    print(sliced_user_session.activities.end)  # [2.]


The returned slice is a new object of the same type as the original data object. The timestamps
are reset relative to the new start time. To keep the original timestamps, set ``reset_origin=False``:

.. code-block:: python

    sliced_data = user_session.slice(start=1.0, end=3.0, reset_origin=False)

    sliced_user_session = user_session.slice(1, 2.5, reset_origin=False)

    # IrregularTimeSeries
    print(sliced_user_session.clicks.timestamps)  # [1.2, 2.3]
    print(sliced_user_session.clicks.position)  # [[100, 200], [150, 300]]
    print(sliced_user_session.clicks.domain.start)  # [1.]
    print(sliced_user_session.clicks.domain.end)  # [2.5]

    # RegularTimeSeries
    print(sliced_user_session.sensor.accelerometer.shape)  # (150, 3)
    print(sliced_user_session.sensor.domain.start)  # [0.]
    print(sliced_user_session.sensor.domain.end)  # [4.]

    # Interval
    print(sliced_user_session.activities.activity)  # ['scrolling']
    print(sliced_user_session.activities.start)  # [2.]
    print(sliced_user_session.activities.end)  # [3.]


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

All temporal data objects, :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`, 
:obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>`, and :obj:`Interval <temporaldata.Interval>` support masking through the ``.select_by_mask()`` method. 
The mask must be a 1D boolean array matching the length of the first dimension of the data arrays:

.. code-block:: python

    import numpy as np

    from temporaldata import Interval, IrregularTimeSeries

    data = Interval(
        start=np.array([0.0, 2.0, 4.5]),
        end=np.array([1.0, 3.0, 6.0]),
        activity=np.array(["typing", "scrolling", "typing"]),
    )

    # Create a boolean mask
    mask = data.activity == "scrolling"


    # Apply mask to select data
    masked_data = data.select_by_mask(mask)

    print(masked_data.activity)  # ['scrolling']
    print(masked_data.start)  # [2.]
    print(masked_data.end)  # [3.]


    data = IrregularTimeSeries(
        timestamps=np.array([1.2, 2.3, 3.8, 6.4, 7.1, 8.9]),
        amplitude=np.array([0.5, 0.7, 0.6, 0.8, 0.4, 0.6]),
        domain="auto",
    )

    mask = data.amplitude > 0.5

    masked_data = data.select_by_mask(mask)

    print(masked_data.timestamps)  # [2.3, 3.8, 6.4, 8.9]
    print(masked_data.amplitude)  # [0.7, 0.6, 0.8, 0.6]

