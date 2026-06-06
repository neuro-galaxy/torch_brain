.. currentmodule:: torch_brain.data

Saving and Loading Data
=======================

:obj:`Data` objects can be saved to and loaded from `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_
files. HDF5 is a specialized data format that allows streaming chunks of data
from disk without loading all of it into memory (RAM), providing an efficient
way to work with large datasets.


Saving
------

The :obj:`~Data.save()` method saves a :obj:`Data` object to disk.

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

:obj:`Data.load()` loads data from disk. We first load it "non-lazily" by
passing ``lazy=False``:

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

Setting ``lazy=False`` loads the entire data file into memory upfront.
This quickly becomes infeasible for datasets of any real size (a few hundred GBs
to a few TBs). To address this, **torch_brain** features a *Lazy Loading* mode.


Lazy Loading
------------

Lazy loading provides data access efficiencies in terms of, both, disk I/O and
memory usage. To load data in lazy mode, simply omit the ``lazy=False`` flag
used above (or explicitly provide ``lazy=True``):

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

Note that:

1. The internal objects are :obj:`LazyRegularTimeSeries` and
   :obj:`LazyIrregularTimeSeries`.

2. The arrays are printed as ``<HDF5 dataset...>``, which indicates that the
   arrays are yet to be loaded.

Let's access ``eye_pos``:

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

Since ``eye_pos`` has been loaded into memory, it is no longer an ``<HDF5
dataset...>``, while the remaining attributes remain lazy. Accessing the
remainig attributes of ``behavior`` will load its data into memory and turn it
into a normal (non-lazy) :obj:`RegularTimeSeries` object:

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

Slicing preserves laziness for any attributes that are still lazy:

.. code-block:: pycon

   >>> sliced = session.slice(2., 4.)
   >>> sliced
   Data(
     behavior=RegularTimeSeries(
       eye_pos=[200, 2],
       hand_vel=[200, 2],
       pupil_size=[200]
     ),
     spikes=LazyIrregularTimeSeries(  # Note that this remains lazy!
       timestamps=<HDF5 dataset "timestamps": shape (3,), type "<f8">,
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">
     ),
   )

Here, ``behavior`` is already in memory (non-lazy), so it is sliced immediately
(indicated by the change in array shapes). On the other hand, ``spikes`` remains
lazy.
Upon accessing ``sliced.spikes.timestamps``, only the two timestamps that fall
within the :math:`[2s, 4s)` window are read from disk and not the full
timestamps array.

.. code-block:: pycon

   >>> sliced
   Data(
     behavior=RegularTimeSeries(
       eye_pos=[200, 2],
       hand_vel=[200, 2],
       pupil_size=[200]
     ),
     spikes=LazyIrregularTimeSeries(
       timestamps=[2],
       unit_id=<HDF5 dataset "unit_id": shape (3,), type "<i8">  # still lazy
     ),
   )

   >>> sliced.spikes.timestamps
   array([0.3, 1.1])


In summary, under lazy-loading:

- Data is read from disk only when an attribute is accessed.

- Slicing is deferred: only the attributes you access get sliced, and only when
  you access them. Importantly, only the sliced portion is read from disk, not
  the whole array.

- Masking (via methods such as :obj:`ArrayDict.select_by_mask()`) is also
  deferred until an attribute is accessed.


