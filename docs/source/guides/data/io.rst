.. currentmodule:: torch_brain.data

Saving and Loading Data
=======================

:obj:`Data` objects can be saved to and loaded from `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_
files. HDF5 is a specialized data format that allows streaming chunks of data
from disk without loading all of it into memory (RAM), giving us an efficient
way to work with large datasets.


Saving
------

To save a data object to disk, use the :obj:`~Data.save()` method:

.. code-block:: pycon

   >>> import numpy as np
   >>> from torch_brain.data import RegularTimeSeries, IrregularTimeSeries, Data

   >>> # Create a complex data object
   >>> session = Data(
   ...     spikes=IrregularTimeSeries(
   ...         timestamps=np.array([1.2, 2.3, 3.1]),
   ...         unit_id=np.array([1, 2, 1]),
   ...     ),
   ...     behavior=RegularTimeSeries(
   ...         sampling_rate=100.0,
   ...         hand_vel=np.random.randn(400, 2),
   ...         eye_pos=np.random.randn(400, 2),
   ...         pupil_size=np.random.randn(400),
   ...     ),
   ...     domain="auto",
   ... )

   >>> # Save to a HDF5 file on disk
   >>> session.save("neural_data.h5")


Loading
-------

To load data back from disk, use :obj:`Data.load()`.
Let's first load it "non-lazily" by passing ``lazy=False``:

.. code-block:: pycon

    >>> # Read neural data from HDF5 file on disk
    >>> session = Data.load("neural_data.h5", lazy=False)

    >>> # Access neural data
    >>> session.spikes.timestamps
    array([1.2, 2.3, 3.1])
    >>> session.behavior.sampling_rate
    np.float64(100.0)

    >>> # Slice
    >>> sliced = session.slice(2., 4.)
    >>> sliced
    Data(
      behavior=RegularTimeSeries(
        eye_pos=[200, 2],
        hand_vel=[200, 2],
        pupil_size=[200]
      ),
      spikes=IrregularTimeSeries(
        timestamps=[2],
        unit_id=[2]
      ),
    )

By setting ``lazy=False``, we load the entire dataset into memory upfront.
This quickly becomes infeasible for datasets of any real size (a few hundred GBs
to a few TBs). To address this, we provide a *Lazy Loading* mode.


Lazy Loading
------------

Under lazy-loading:

- Data is read from disk only when an attribute is accessed.

- Slicing is also deferred: only the attributes you access get sliced, and
  only at the moment you access them. Importantly, only the sliced portion is
  read from disk, not the whole array. So slicing a small window out of a
  huge recording is cheap, both in terms of disk I/O and memory usage.

- The same goes for masking (via methods such as :obj:`ArrayDict.select_by_mask()`):
  the masking is deferred until the attribute is actually requested.


To load data in lazy mode, simply omit the ``lazy=False`` flag we used above:

.. code-block:: pycon

   >>> # omit lazy=False to load lazily
   >>> session = Data.load("neural_data.h5")
   >>> session
   Data(
     behavior=LazyRegularTimeSeries(
       eye_pos=<HDF5 dataset "eye_pos": shape (400, 2), type "<f8">,
       hand_vel=<HDF5 dataset "hand_vel": shape (400, 2), type "<f8">,
       pupil_size=<HDF5 dataset "pupil_size": shape (400,), type "<f8">
     ),
     spikes=LazyIrregularTimeSeries(
       timestamps=<HDF5 dataset "timestamps": shape (3,), type "<f8">,
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">
     ),
   )

First note that the internal objects are :obj:`LazyRegularTimeSeries` and
:obj:`LazyIrregularTimeSeries`. Secondly, the presence of ``<HDF5 dataset...>``
indicates that the arrays are yet to be loaded. Let's see what happens when we
access ``eye_pos``:

.. code-block:: pycon

   >>> session.behavior.eye_pos
   array([[ 1.14566523,  0.22616446],
          [-0.03963849,  0.11477352],
          ...

   >>> session
   Data(
     behavior=LazyRegularTimeSeries(
       eye_pos=[400, 2],
       hand_vel=<HDF5 dataset "hand_vel": shape (400, 2), type "<f8">,
       pupil_size=<HDF5 dataset "pupil_size": shape (400,), type "<f8">
     ),
     spikes=LazyIrregularTimeSeries(
       timestamps=<HDF5 dataset "timestamps": shape (3,), type "<f8">,
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">
     ),
   )

We can see that ``eye_pos`` has been loaded, and the remaining attributes
are still lazy. If we access both ``hand_vel`` and ``pupil_size``, ``behavior``
will then turn into a :obj:`RegularTimeSeries` object:

.. code-block:: pycon

   >>> session.behavior.hand_vel
   array([[ 1.86527056e-01,  1.54714182e-01],
          [ 2.75861600e-01, -5.30891532e-01],
          ...
   >>> session.behavior.pupil_size
   array([ 1.66121169e-01,  2.06565774e-01, -7.85847571e-01, ...])

   >>> session
   Data(
     behavior=RegularTimeSeries(
       eye_pos=[400, 2],
       hand_vel=[400, 2],
       pupil_size=[400]
     ),
     spikes=LazyIrregularTimeSeries(
       timestamps=<HDF5 dataset "timestamps": shape (3,), type "<f8">,
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">
     ),
   )

We can also slice a lazy object:

.. code-block:: pycon

   >>> sliced = session.slice(2., 4.)
   >>> sliced
   Data(
     behavior=RegularTimeSeries(
       eye_pos=[200, 2],
       hand_vel=[200, 2],
       pupil_size=[200]
     ),
     spikes=LazyIrregularTimeSeries(  # Note that this remains lazy!!
       timestamps=<HDF5 dataset "timestamps": shape (3,), type "<f8">,
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">
     ),
   )

   >>> sliced.spikes.timestamps
   array([0.3, 1.1])

Here, ``spikes`` stayed lazy after slicing. When we finally access
``sliced.spikes.timestamps``, only the two timestamps that fall within the
:math:`[2, 4)` window are read from disk and not the full timestamps array.
This is what makes lazy loading efficient: you only pay for the slice you ask for.

