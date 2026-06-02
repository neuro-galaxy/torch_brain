Meet the Data Objects
=====================

The :obj:`torch_brain.data` module defines several key of data objects.
Here we'll look at the different ways to create and interact with each type of object.

.. contents:: Contents
   :depth: 1
   :local:

.. .. note::
..    All timestamps should be expressed in seconds. Sampling rates are specified in Hz (samples per second).


ArrayDict
---------
:obj:`~torch_brain.data.ArrayDict` is a simple container for
arbitrary arrays that share the same *first* dimension.
By itself, this object is not meant to store *temporal* attributes, but
forms the base for all other objects in our library.

.. code-block:: pycon

   >>> from torch_brain.data import ArrayDict

   >>> # Create an ArrayDict with any attributes you want
   >>> channels = ArrayDict(
   ...     channel_id=[1, 2, 3],
   ...     brain_region=['V1', 'V2', 'V1'],
   ...     position=[[0., 1.], [0.1, 0.9], [1.2, 3.2]]
   ... )

   >>> # Printing the object shows the shapes of the underlying data
   >>> channels
   ArrayDict(
     channel_id=[3],
     brain_region=[3],
     position=[3, 2]
   )

   >>> # Access any attributes
   >>> channels.brain_region
   array([[0. , 1. ],
          [0.1, 0.9],
          [1.2, 3.2]])

RegularTimeSeries
-----------------
:obj:`~torch_brain.data.RegularTimeSeries` is the first time-oriented data
object we will look at. As the name suggests, it is meant to store time-series
that are regularly sampled.

.. code-block:: pycon

   >>> import numpy as np
   >>> from torch_brain.data import RegularTimeSeries

   >>> behavior = RegularTimeSeries(
   ...     sampling_rate=100.0,  # in Hz
   ...     hand_vel=np.random.randn(1000, 2),
   ...     eye_pos=np.random.randn(1000, 2),
   ...     pupil_size=np.random.randn(1000),
   ... )
   >>> behavior
   RegularTimeSeries(
     hand_vel=[1000, 2],
     eye_pos=[1000, 2],
     pupil_size=[1000]
   )

   >>> # length represents the number of timepoints
   >>> len(behavior)
   1000

   >>> # timestamps are automatically created from the sampling rate
   >>> behavior.timestamps
   array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, ...])

Here, we have created a 10-second long collection of behavioral measurements
(hand velocity, eye position, and pupil size), in the *same* data object.
This allows us to get time-slices of the entire set of signals!

Let's get a slice starting at 2 seconds and ending at 3 seconds.
Since our signals are sampled at 100Hz, we should get 100 samples.

.. code-block:: pycon

   >>> sliced = behavior.slice(2., 3.)
   >>> sliced
   RegularTimeSeries(
     hand_vel=[100, 2],
     eye_pos=[100, 2],
     pupil_size=[100]
   )

   >>> len(sliced)
   100

   >>> sliced.pupil_size
   array([-0.07094018,  1.1442879 ,  1.26022563,  1.57259098, ..., ])


:obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`
-------------------------------------------------------------
An :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>` represents event-based or irregularly sampled time series data, it is also well suited for time series with missing data.

.. tab:: Generic

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval

        # Create with timestamps and additional data
        events = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.1]),
            event_type=np.array(['click', 'scroll', 'click']),
            user_id=np.array([1, 2, 1]),
            timekeys=['timestamps'],
            domain=Interval(start=0, end=4)
        )

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval

        # Create with timestamps and additional data
        spikes = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.1]),
            unit_id=np.array([1, 2, 1]),
            amplitude=np.array([0.5, 0.7, 0.6]),
            waveforms=np.random.randn(3, 32),
            timekeys=['timestamps'],
            domain=Interval(start=0, end=4)
        )


Choosing ``timekeys``
^^^^^^^^^^^^^^^^^^^^^

The ``timekeys`` parameter specifies which attributes represent timestamps that should be adjusted during temporal operations.
Include any attributes that represent absolute times, this will ensure that when the data is sliced and shifted, the timestamps are updated accordingly:

.. tab:: Generic

    .. code-block:: python

        # Both timestamps and response_times are time attributes
        trials = IrregularTimeSeries(
            timestamps=np.array([1.0, 3.0, 5.0]),      # stimulus onset times
            response_times=np.array([1.5, 3.8, 5.7]),  # response times
            accuracy=np.array([1, 0, 1]),              # not a time attribute
            reaction_time=np.array([0.5, 0.8, 0.7]),   # duration between timestamps and response_times
            timekeys=['timestamps', 'response_times']
        )

.. tab:: Neuroscience

    .. code-block:: python

        # Both timestamps and response_times are time attributes
        trials = IrregularTimeSeries(
            timestamps=np.array([1.0, 3.0, 5.0]),      # stimulus onset times
            response_times=np.array([1.5, 3.8, 5.7]),  # response times
            spike_rate=np.array([45.2, 32.1, 67.8]),   # not a time attribute
            reaction_time=np.array([0.5, 0.8, 0.7]),   # duration between timestamps and response_times
            timekeys=['timestamps', 'response_times']
        )

Note the distinction between **durations** and **times**: Only include attributes representing absolute times, not durations.


Choosing ``domain``
^^^^^^^^^^^^^^^^^^^

The ``domain`` parameter specifies the time range over which the time series is defined. It is an :obj:`Interval <temporaldata.Interval>` object that defines the start and end times of the data.

For example, if you have event data from 0 to 10 seconds, but all events occur between 2-8 seconds, setting ``domain=Interval(start=0, end=10)`` makes it explicit that the recording spans the full 10 seconds:

.. tab:: Generic

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval

        # Events only occur between 2-8 seconds
        events = IrregularTimeSeries(
            timestamps=np.array([2.1, 3.4, 7.8]),
            event_type=np.array(['click', 'scroll', 'click']),
            domain=Interval(start=0, end=10)  # But recording is 0-10 seconds
        )

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval

        # Spikes only occur between 2-8 seconds
        spikes = IrregularTimeSeries(
            timestamps=np.array([2.1, 3.4, 7.8]),
            amplitude=np.array([0.5, 0.7, 0.6]),
            domain=Interval(start=0, end=10)  # But recording is 0-10 seconds
        )

Without specifying the domain, operations might incorrectly assume the time series only spans from 2.1 to 7.8 seconds.

It is also useful for when the data is not contiguous, where you have a chunk of data that is missing from the recording:

.. tab:: Generic

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval
        # Recording with a gap between 4-6 seconds
        events = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.8, 6.4, 7.1, 8.9]),
            event_type=np.array(['click', 'scroll', 'click', 'scroll', 'click', 'scroll']),
            domain=Interval(
                start=np.array([0.0, 6.0]),  # Two intervals
                end=np.array([4.0, 10.0])    # Gap between 4-6 seconds
            )
        )

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import IrregularTimeSeries, Interval

        # Recording with a gap between 4-6 seconds
        spikes = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.8, 6.4, 7.1, 8.9]),
            amplitude=np.array([0.5, 0.7, 0.6, 0.8, 0.4, 0.6]),
            domain=Interval(
                start=np.array([0.0, 6.0]),  # Two intervals
                end=np.array([4.0, 10.0])    # Gap between 4-6 seconds
            )
        )

Finally, you can also set ``domain="auto"`` to infer the domain from the data, as ``[min(timestamps), max(timestamps))``. However, explicitly setting it is recommended when you know the true temporal extent of your recording.

.. tab:: Generic

    .. code-block:: python

        from temporaldata import IrregularTimeSeries

        # Recording with auto-inferred domain
        events = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.8, 6.4, 7.1, 8.9]),
            event_type=np.array(['click', 'scroll', 'click', 'scroll', 'click', 'scroll']),
            domain="auto"
        )

        print(events.domain)
        # Output: Interval(start=1.2, end=8.9)

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import IrregularTimeSeries

        # Recording with auto-inferred domain
        spikes = IrregularTimeSeries(
            timestamps=np.array([1.2, 2.3, 3.8, 6.4, 7.1, 8.9]),
            amplitude=np.array([0.5, 0.7, 0.6, 0.8, 0.4, 0.6]),
            domain="auto"
        )

        print(spikes.domain)
        # Output: Interval(start=1.2, end=8.9)





:obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>`
---------------------------------------------------------
A :obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>` represents uniformly sampled time series data. There is no need to provide
``timestamps`` as they are infered from the sampling rate.

.. tab:: Generic

    .. code-block:: python

        from temporaldata import RegularTimeSeries

        # Create with sampling rate and data
        sensor_data = RegularTimeSeries(
            sampling_rate=100,  # Hz
            temperature=np.random.randn(1000),  # 10 seconds of temperature data
            humidity=np.random.randn(1000),  # 10 seconds of humidity data
            domain_start=0,  # Start time
        )

        print(sensor_data.timestamps)
        # Output: array([ 0,  0.01,  0.02,  0.03, ...,  9.98,  9.99])

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import RegularTimeSeries

        # Create with sampling rate and data
        lfp = RegularTimeSeries(
            sampling_rate=1000,  # Hz
            raw=np.random.randn(10000, 3),  # 10 seconds of 3-channel LFP
            domain_start=0,  # Start time
        )

        print(lfp.timestamps)
        # Output: array([ 0,  0.001,  0.002,  0.003, ...,  9.998,  9.999])


Choosing ``domain``
^^^^^^^^^^^^^^^^^^^
:obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>` does not need a ``domain``
argument. The domain is always computed automatically as
:math:`[t_0,\ t_0 + N / f_s)`, where :math:`t_0` is ``domain_start``, :math:`N` is the
number of samples, and :math:`f_s` is ``sampling_rate``, so that its boundaries stay
aligned to the sample grid. Use ``domain_start`` to set the start time, as in the
examples above.

.. note::

    Passing ``domain="auto"`` is still accepted for backward compatibility but
    emits a :class:`DeprecationWarning` and will be removed in a future version.
    Passing a custom :obj:`Interval <temporaldata.Interval>` raises ``ValueError``.


Converting to :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is easy to convert a :obj:`RegularTimeSeries <temporaldata.RegularTimeSeries>` to an :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>` using the :meth:`to_irregular` method:

.. code-block:: python

    # Convert RegularTimeSeries to IrregularTimeSeries
    irregular_data = regular_data.to_irregular()




:obj:`Interval <temporaldata.Interval>`
---------------------------------------
An :obj:`Interval <temporaldata.Interval>` represents time periods. The only required attributes are ``start`` and ``end``.

.. tab:: Generic

    .. code-block:: python

        from temporaldata import Interval

        # Create with start/end times and additional data
        meetings = Interval(
            start=np.array([0, 60, 120]),
            end=np.array([45, 105, 180]),
            title=np.array(['Team Sync', 'Planning', 'Review']),
            room=np.array(['A101', 'B202', 'A101']),
            timekeys=['start', 'end']
        )

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import Interval

        # Create with start/end times and additional data
        trials = Interval(
            start=np.array([0, 2, 4]),
            end=np.array([1, 3, 5]),
            stimulus=np.array(['left', 'right', 'left']),
            outcome=np.array(['correct', 'error', 'correct']),
            timekeys=['start', 'end']
        )


Intervals can also be created from a list of tuples using :meth:`from_list`:

.. code-block:: python

    # Create from list of (start, end) tuples
    intervals = Interval.from_list([
        (0, 1),
        (1, 2),
        (2, 3)
    ])

Or from a pandas DataFrame using :meth:`from_dataframe`:

.. code-block:: python

    import pandas as pd

    # Create from DataFrame with 'start' and 'end' columns
    df = pd.DataFrame({
        'start': [0, 1, 2],
        'end': [1, 2, 3],
        'label': ['A', 'B', 'C']
    })
    intervals = Interval.from_dataframe(df)

Or using :meth:`linspace` or :meth:`arange` to create evenly spaced intervals:

.. code-block:: python

    # Create 5 evenly spaced intervals from 0 to 10
    intervals = Interval.linspace(0, 10, 5)

    # Create intervals with step size 2 from 0 to 10
    intervals = Interval.arange(0, 10, 2)

When you have a single interval, you can simply provide float values:

.. code-block:: python

    # Create a single interval from 0 to 10
    interval = Interval(start=0, end=10)


Choosing ``timekeys``
^^^^^^^^^^^^^^^^^^^^^

Like for :obj:`IrregularTimeSeries <temporaldata.IrregularTimeSeries>`, the ``timekeys`` parameter specifies which attributes represent timestamps that should be adjusted during temporal operations.

.. tab:: Generic

    .. code-block:: python

        # start, end, and event_time are time attributes
        segments = Interval(
            start=np.array([1.0, 3.0, 5.0]),      # segment start times
            end=np.array([2.0, 4.0, 6.0]),        # segment end times
            event_time=np.array([1.5, 3.5, 5.5]), # important event within segment
            label=np.array(['A', 'B', 'C']),      # not a time attribute
            timekeys=['start', 'end', 'event_time']
        )

.. tab:: Neuroscience

    .. code-block:: python

        # start, end, and go_cue are time attributes
        trials = Interval(
            start=np.array([1.0, 3.0, 5.0]),      # trial start times
            end=np.array([2.0, 4.0, 6.0]),        # trial end times
            go_cue=np.array([1.2, 3.3, 5.1]),     # go cue presentation time
            condition=np.array(['cue1', 'cue2', 'cue1']),  # not a time attribute
            timekeys=['start', 'end', 'go_cue']
        )

No ``domain``
^^^^^^^^^^^^^

There is no need to set a ``domain`` for :obj:`Interval <temporaldata.Interval>` objects, as the intervals themselves represent their own domain.


:obj:`Data <temporaldata.Data>`
-------------------------------
The :obj:`Data <temporaldata.Data>` class is a container that holds and organizes all temporaldata objects, including other :obj:`Data <temporaldata.Data>` objects, strings, numbers, floats, numpy arrays, and more.

.. tab:: Generic

    .. code-block:: python

        from temporaldata import Data

        # Create a complex data object
        user_session = Data(
            clicks=IrregularTimeSeries(
                timestamps=np.array([1.2, 2.3, 3.1]),
                position=np.array([[100,200], [150,300], [200,150]]),
                domain=Interval(start=0, end=4)
            ),
            sensor=RegularTimeSeries(
                sampling_rate=100,
                accelerometer=np.random.randn(400, 3),
            ),
            activities=Interval(
                start=np.array([0, 2]),
                end=np.array([1, 3]),
                activity=np.array(['typing', 'scrolling'])
            ),

            user_id='user123',
            device='laptop',
            domain="auto",
        )

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import Data

        # Create a complex data object
        session = Data(
            spikes=IrregularTimeSeries(
                timestamps=np.array([1.2, 2.3, 3.1]),
                unit_id=np.array([1, 2, 1]),
                domain=Interval(start=0, end=4)
            ),
            units=ArrayDict(
                unit_id=np.array([1, 2, 1]),
                brain_region=np.array(['V1', 'V2', 'V1']),
            ),
            lfp=RegularTimeSeries(
                sampling_rate=1000,
                raw=np.random.randn(4000, 3),
            ),
            trials=Interval(
                start=np.array([0, 2]),
                end=np.array([1, 3]),
                condition=np.array(['A', 'B'])
            ),
            subject_id='mouse1',
            date='2023-01-01',
            domain="auto",
        )

Choosing ``domain``
^^^^^^^^^^^^^^^^^^^

The recommended way to set the domain is to set ``domain="auto"``, which will infer the domain from the data.
Note that ``domain`` is not required when the data object does not contain any time-based data.
