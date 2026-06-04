.. currentmodule:: torch_brain.data

Meet the Data Objects
=====================

The :obj:`torch_brain.data` module defines several key data objects.
Here we'll look at the different ways to create and interact with each type of object.


RegularTimeSeries
-----------------
:obj:`RegularTimeSeries` is the first time-oriented data object we will look
at. As the name suggests, it is meant to store time-series that are regularly
sampled. This could be anything from behavior measurements to EEG signals.

.. code-block:: pycon

   >>> import numpy as np
   >>> from torch_brain.data import RegularTimeSeries

   >>> behavior = RegularTimeSeries(
   ...     sampling_rate=100.0,  # in Hz
   ...     hand_vel=np.random.randn(1000, 2),
   ...     eye_pos=np.random.randn(1000, 2),
   ...     pupil_size=np.random.randn(1000),
   ... )

   >>> # Printing the object shows the shapes of the underlying data
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


Here, we have created a 10-second-long collection of behavioral measurements
(hand velocity, eye position, and pupil size) inside the *same* data object.
This allows us to get time-slices of the entire set of signals at once.

Let's grab a slice starting at 2 seconds and ending at 4 seconds.
Since our signals are sampled at 100Hz, we should get 200 samples.

.. code-block:: pycon

   >>> sliced = behavior.slice(2., 4.)
   >>> sliced
   RegularTimeSeries(
     hand_vel=[200, 2],
     eye_pos=[200, 2],
     pupil_size=[200]
   )

   >>> len(sliced)
   200

   >>> sliced.pupil_size
   array([-0.07094018,  1.1442879 ,  1.26022563,  1.57259098, ..., ])


.. admonition:: Other constructors
   :class: tip

   :obj:`RegularTimeSeries.from_gappy_timeseries()`: If your raw data has missing
   timesteps. Also see :doc:`gappy_regular_ts`.


IrregularTimeSeries
-------------------
An :obj:`IrregularTimeSeries` represents event-based or irregularly sampled
time-series data, and is well suited for things like discrete events and
spike-trains.

.. code-block:: pycon

   >>> from torch_brain.data import IrregularTimeSeries

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
In this example, slicing ``spikes`` from 2 to 4 seconds should give us the
2 spikes that fall within that window.

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

Notice how slicing *shifted* the timestamps so that the slice starts at zero.
You can opt out of this by passing ``reset_origin=False``:

.. code-block:: pycon

   >>> sliced = spikes.slice(2.0, 4.0, reset_origin=False)
   >>> sliced.timestamps
   array([2.3, 3.1])

.. TODO Add link to some tutorial explaining timekeys


.. admonition:: Other constructors
   :class: tip

   :obj:`IrregularTimeSeries.from_dataframe()`


Interval
--------
:obj:`Interval` represents time-periods and any metadata attached to them. A
common use of this is to represent the trial structure of an experiment:

.. code-block:: pycon

   >>> from torch_brain.data import Interval

   >>> trials = Interval(
   ...     start=[0., 2., 4.],  # required
   ...     end=[1., 3., 5.],  # required
   ...     stimulus=['left', 'right', 'left'],
   ...     outcome=['correct', 'error', 'correct'],
   ... )
   >>> trials
   Interval(
     start=[3],
     end=[3],
     stimulus=[3],
     outcome=[3]
   )

This says that during the interval :math:`[0, 1)` the stimulus was ``'left'``
and the outcome was ``'correct'``; during :math:`[2, 3)` the stimulus was
``'right'`` and the outcome was ``'error'``, and so on.

In other words, each *pair* of start and end timestamps describes the boundary
of a period, and the remaining attributes act as metadata for what happened
within that period.

Trials can also be sliced:

.. code-block:: pycon

   >>> sliced = trials.slice(2.0, 4.0)
   >>> len(sliced)
   1

   >>> sliced.start, sliced.end, sliced.stimulus, sliced.outcome
   (array([0.]),
    array([1.]),
    array(['right'], dtype='<U5'),
    array(['error'], dtype='<U7'))

.. admonition:: Other constructors
   :class: tip

   :obj:`Interval.from_list()`, :obj:`Interval.from_dataframe()`

ArrayDict
---------
Our final *core* data object is :obj:`ArrayDict`. It is a simple container
for arbitrary arrays *that share the same first dimension*.
This data structure is useful for storing things like metadata associated with
different recording channels, or any other data in a tabular form.

.. code-block:: pycon

   >>> from torch_brain.data import ArrayDict

   >>> # Create an ArrayDict with any attributes you want
   >>> channels = ArrayDict(
   ...     channel_id=[1, 2, 3],
   ...     brain_region=['V1', 'V2', 'V1'],
   ...     position=[[0., 1.], [0.1, 0.9], [1.2, 3.2]]
   ... )

   >>> channels
   ArrayDict(
     channel_id=[3],
     brain_region=[3],
     position=[3, 2]
   )

   >>> # Access any attribute
   >>> channels.position
   array([[0. , 1. ],
          [0.1, 0.9],
          [1.2, 3.2]])

.. admonition:: Other constructors
   :class: tip

   :obj:`ArrayDict.from_dataframe()`


Data
----
:obj:`Data` objects act as containers for all four types of objects we
discussed above:

.. code-block:: pycon

   >>> from torch_brain.data import Data

   >>> data = Data(
   ...     channels=channels,
   ...     spikes=spikes,
   ...     behavior=behavior,
   ...     trials=trials,
   ...     domain="auto",  # don't worry about this for now; "auto" is the right choice most of the time
   ... )

Let's say this ``data`` object represents one entire neural recording.
The nice thing about this container is that it can be sliced *as a whole*:

.. code-block:: pycon

   >>> sliced = data.slice(2., 4.)
   >>> sliced
   Data(
     channels=ArrayDict(
       channel_id=[3],
       brain_region=[3],
       position=[3, 2]
     ),
     spikes=IrregularTimeSeries(
       timestamps=[2],
       unit_id=[2],
       amplitude=[2],
       waveforms=[2, 32]
     ),
     behavior=RegularTimeSeries(
       hand_vel=[200, 2],
       eye_pos=[200, 2],
       pupil_size=[200]
     ),
     trials=Interval(
       start=[1],
       end=[1],
       stimulus=[1],
       outcome=[1]
     ),
   )
   >>> sliced.spikes.timestamps
   array([0.3, 1.1])  # same as the IrregularTimeSeries example above

The sliced object also remembers the absolute time at which it was sliced:

.. code-block:: pycon

   >>> sliced.absolute_start
   2.0

The sliced data object is itself just another :obj:`Data` object, which means
you can slice it again:

.. code-block:: pycon

   >>> sliced_again = sliced.slice(1., 2.)
   >>> sliced_again.absolute_start
   3.0
   >>> sliced_again
   Data(
     channels=ArrayDict(
       channel_id=[3],
       brain_region=[3],
       position=[3, 2]
     ),
     spikes=IrregularTimeSeries(
       timestamps=[1],
       unit_id=[1],
       amplitude=[1],
       waveforms=[1, 32]
     ),
     behavior=RegularTimeSeries(
       hand_vel=[100, 2],
       eye_pos=[100, 2],
       pupil_size=[100]
     ),
     trials=Interval(
       start=[0],
       end=[0],
       stimulus=[0],
       outcome=[0]
     ),
   )

You can also store a few other things in :obj:`Data`: numpy arrays and Python
primitives (scalars, lists, tuples, strings, etc.).
Additionally, :obj:`Data` objects can be *nested* — a :obj:`Data` object can
itself contain another :obj:`Data` object — which lets you organize your data
in a hierarchy.


Conventions
-----------

Some conventions to note:

* In ``torch_brain``, time is always represented in the units of seconds.
* Slicing like ``data.slice(a, b)`` is inclusive on the left side, and
  exclusive on the right side.
