.. currentmodule:: torch_brain.data

Saving and (Lazy) Loading Data
==============================

:obj:`Data` objects can be saved to and loaded from `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_
files. HDF5 is a specialized data format that allows streaming data chunks from
disk without loading it all in memory (RAM), thus providing an efficient way to
work with large datasets.


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

To load our previously written data from disk, use :obj:`Data.load()`.
Let's first load the data "non-lazily" by using ``lazy=False``:

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

By setting, ``lazy=False``, we instantly load the entire data into memory.
This quickly becomes infeasible for datasets of reasonable size (a few 100GBs
to a few TBs). To address this, we introduce our *Lazy Loading* feature.


Lazy Loading
------------

Under lazy-loading,

- Data is read from disk only when attributes are accessed.
  Once an attribute is accessed, it is then kept in memory.

- Slicing is also done lazily, where only attributes that you access will
  be sliced whenever you access them.

- Same goes for masking (via methods such as :obj:`ArrayDict.select_by_mask()`).
  Actual masking is deferred until attributes are actually requested.


To load data in lazy mode, simply omit the ``lazy=False`` flag that we used above:

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

The presence of ``<HDF5 dataset...>`` indicates that the arrays are yet
to be loaded. Let's see what happens when we access ``eye_pos``:

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

We can see that ``eye_pos`` has now been loaded, and the remaining attributes
are still lazy.
