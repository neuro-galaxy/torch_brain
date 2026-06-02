.. currentmodule:: torch_brain.data

Meet the Data Objects
=====================

The :obj:`torch_brain.data` module defines several key of data objects.
Here we'll look at the different ways to create and interact with each type of object.

.. contents:: Contents
   :depth: 1
   :local:


ArrayDict
---------
:obj:`ArrayDict` is a simple container for arbitrary arrays *that share the same
first dimension*.
This data structure is useful for storing things like metadata assosciated with
different recording channels, or any data in a tabular form.

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
:obj:`RegularTimeSeries` is the first time-oriented data object we will look
at. As the name suggests, it is meant to store time-series that are regularly
sampled.

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


IrregularTimeSeries
-------------------
An :obj:`IrregularTimeSeries` represents event-based or irregularly sampled
time series data, it is also well suited for representing events and spike-trains.

.. code-block:: pycon

   >>> from temporaldata import IrregularTimeSeries

   >>> # Create with timestamps and additional data
   >>> events = IrregularTimeSeries(
   ...     timestamps=[1.2, 2.3, 3.1],  # required
   ...     event_type=['click', 'scroll', 'click'],
   ...     user_id=[1, 2, 1],
   ... )

   >>> # Real spike-trains would be much longer ofcourse,
   >>> # here we show a spike-train with only 3 spikes.
   >>> spikes = IrregularTimeSeries(
   ...     timestamps=[1.2, 2.3, 3.1],  # required
   ...     unit_id=[1, 2, 1],
   ...     amplitude=[0.5, 0.7, 0.6],
   ...     waveforms=np.random.randn(3, 32),
   ... )

IrregularTimeSeries objects can also be time-sliced.
In this example, slicing ``spikes`` from 2 to 4 seconds should give us 2 spikes.

.. code-block:: pycon

   >>> sliced = spikes.slice(2.0, 4.0)
   >>> sliced
   IrregularTimeSeries(
     timestamps=[2],
     unit_id=[2],
     amplitude=[2],
     waveforms=[2, 32]
   )

   >>> sliced.timestamps
   array([0.3, 1.1])

   >>> sliced.unit_id
   array([2, 1])

Notice how slicing *shifted* the timestamps by the start-time of the slice.
You can choose to avoid this, by using ``reset_origin=False``:

.. code-block:: pycon

   >>> sliced = spikes.slice(2.0, 4.0, reset_origin=False)
   >>> sliced.timestamps
   array([2.3, 3.1])

.. TODO Add link to some tutorial explaining timekeys

.. note::

   Note that in ``torch_brain``, we always represent time in the units of seconds.



Interval
--------
Our final *core* object is :obj:`Interval`, which is meant to represent
finite-time periods and their attached metadata.
A common example of this would be represent the trial-structure of
an experiment:

.. code-block:: pycon

   >>> from temporaldata import Interval

   >>> trials = Interval(
   ...     start=[0., 2., 4.],  # required
   ...     end=[1., 3., 5.],  # required
   ...     stimulus=['left', 'right', 'left'],
   ...     outcome=['correct', 'error', 'correct'],
   ... )

Trials can also be sliced, but they have a quirk


Data
----
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
