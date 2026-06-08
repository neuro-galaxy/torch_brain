.. currentmodule:: torch_brain.data

Meet the Data Objects
=====================

The :obj:`torch_brain.data` module defines several key data objects.
This guide covers how to create and interact with each type of object.


RegularTimeSeries
-----------------
:obj:`RegularTimeSeries` stores regularly sampled time-series, such as behavior
measurements or EEG signals. It can store multiple signals that are sampled at
the same timestamps.

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
   array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, ..., 9.99])


This creates a 10-second collection of behavioral measurements (hand velocity,
eye position, and pupil size) inside a single object.

This entire set of signals can be time-sliced at once. As an example, we slice
it from :math:`t=2s` to :math:`t=4s` below. The signals are sampled at 100Hz,
so the slice returns 200 samples.

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

:obj:`RegularTimeSeries` can also incorporate a starting time offset:

.. code-block:: pycon

   >>> behavior = RegularTimeSeries(
   ...     sampling_rate=100.0,  # in Hz
   ...     hand_vel=np.random.randn(1000, 2),
   ...     eye_pos=np.random.randn(1000, 2),
   ...     pupil_size=np.random.randn(1000),
   ...     domain_start=10.0,  # <- the starting time offset
   ... )

   >>> # timestamps start from 10s
   >>> behavior.timestamps
   array([10.  , 10.01, 10.02, 10.03, 10.04, ..., 19.99]])


.. admonition:: Other constructors
   :class: tip

   :obj:`RegularTimeSeries.from_gappy_timeseries()`: If your raw data has missing
   timestamps. Also see the :doc:`gappy_regular_ts` guide.


**torch_brain** follows some common conventions throughout:

* Time is always represented in seconds.
* ``data.slice(a, b)`` is inclusive on the left side and exclusive on the right
  side.

IrregularTimeSeries
-------------------
An :obj:`IrregularTimeSeries` represents event-based or irregularly sampled
time-series data, and is well suited for things like discrete events and
spike-trains. It can also store multiple time-series that share the same
timestamps.

.. code-block:: pycon

   >>> from torch_brain.data import IrregularTimeSeries

   >>> # Create with timestamps and additional data
   >>> events = IrregularTimeSeries(
   ...     timestamps=[1.2, 2.3, 3.1],  # required
   ...     event_type=['click', 'scroll', 'click'],
   ...     user_id=[1, 2, 1],
   ... )

   >>> # Real spike-trains are much longer; this one has only 3 spikes.
   >>> spikes = IrregularTimeSeries(
   ...     timestamps=[1.2, 2.3, 3.1],  # required
   ...     unit_id=[1, 2, 1],
   ...     amplitude=[0.5, 0.7, 0.6],
   ...     waveforms=np.random.randn(3, 32),
   ... )

Inputs passed as Python lists are automatically converted to :obj:`numpy`
arrays internally.

:obj:`IrregularTimeSeries` objects also support time-slicing. We slice
``spikes`` from :math:`t=2s` to :math:`t=4s`, which returns the 2 spikes within
that window.

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

Slicing *shifts* the timestamps to the left by the start time of the slicing
window. Pass ``reset_origin=False`` to keep the original timestamps:

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
:obj:`Interval` represents time-periods and any metadata attached to them, such
as the trial structure of an experiment:

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

Each *pair* of start and end timestamps defines the boundary of a time-period,
and the remaining attributes hold metadata for that period. By convention,
start times are inclusive and end times are non-inclusive. Here, during the
interval :math:`[0s, 1s)` the stimulus is ``'left'`` and the outcome is
``'correct'``; during :math:`[2s, 3s)` the stimulus is ``'right'`` and the
outcome is ``'error'``, and so on.

Intervals also support slicing:

.. code-block:: pycon

   >>> sliced = trials.slice(2.0, 4.0)
   >>> len(sliced)
   1

   >>> sliced.start, sliced.end, sliced.stimulus, sliced.outcome
   (array([0.]),
    array([1.]),
    array(['right'], dtype='<U5'),
    array(['error'], dtype='<U7'))

When a slicing window partially overlaps a time-period, the slice includes the
entire period, and the start time of the slice sets the shift. Below, we slice
from :math:`t=2.5s` to :math:`t=4s`. The time-period :math:`[2s, 3s)` partially
overlaps this window, so the slice includes it and shifts its timestamps to the
left by :math:`2.5s`.

.. code-block:: pycon

   >>> sliced = trials.slice(2.5, 4.0)
   >>> len(sliced)
   1

   >>> sliced.start, sliced.end, sliced.stimulus, sliced.outcome
   (array([-0.5]),
    array([0.5]),
    array(['right'], dtype='<U5'),
    array(['error'], dtype='<U7'))

.. admonition:: Other constructors
   :class: tip

   :obj:`Interval.from_list()`, :obj:`Interval.from_dataframe()`

ArrayDict
---------
:obj:`ArrayDict` is the final *core* data object. It is a simple container for
arbitrary arrays *that share the same first dimension*. Use it to store tabular
data, such as metadata associated with different recording channels.

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
:obj:`Data` objects act as containers for all four object types above:

.. code-block:: pycon

   >>> from torch_brain.data import Data

   >>> data = Data(
   ...     channels=channels,
   ...     spikes=spikes,
   ...     behavior=behavior,
   ...     trials=trials,
   ...     domain="auto",  # don't worry about this for now; "auto" is the right choice most of the time
   ... )

This ``data`` object represents one entire neural recording. We can slice the
container *as a whole*:

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

The sliced object remembers the absolute time at which it was sliced:

.. code-block:: pycon

   >>> sliced.absolute_start
   2.0

The sliced data object is itself another :obj:`Data` object, so you can slice it
again:

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

:obj:`Data` can also store numpy arrays and Python primitives (scalars, lists,
tuples, strings, etc.). :obj:`Data` objects also *nest* — a :obj:`Data` object
can contain another :obj:`Data` object — which can be used to organize data in
a meaningful hierarchy.

