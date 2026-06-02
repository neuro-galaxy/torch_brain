.. currentmodule:: torch_brain.data

Saving and Loading Data Objects
===============================

:obj:`Data` objects can be saved to and loaded from *`HDF5
<https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_* files. This is a
specialized data format that allows us to stream data from disk without loading
it all in memory (RAM), thus providing an efficient way to work with large
datasets.


Saving
------

To save a data object to disk, use the :obj:`~Data.save()` method:

.. code-block:: python

   import numpy as np
   from torch_brain.data import RegularTimeSeries, IrregularTimeSeries, Data, Interval

   # Create a complex data object
   session = Data(
       spikes=IrregularTimeSeries(
           timestamps=np.array([1.2, 2.3, 3.1]),
           unit_id=np.array([1, 2, 1]),
           domain=Interval(start=0, end=4)
       ),
       lfp=RegularTimeSeries(
           sampling_rate=1000,
           raw=np.random.randn(4000, 3),
       ),
   )

   # Save to a HDF5 file on disk
   session.save("neural_data.h5")


Loading
-------

To load our previously written data from disk, use :obj:`Data.load()`:

.. code-block:: python

    # Read neural data from HDF5 file on disk
    session = Data.load("neural_data.h5")

    # Access neural data
    print(session.spikes.timestamps)  # [1.2, 2.3, 3.1]
    print(session.lfp.sampling_rate)  # 1000
    print(session.subject_id)  # 'mouse1'

    # Get spikes from specific unit
    unit1_spikes = session.spikes.select_by_mask(session.spikes.unit_id == 1)
    print(unit1_spikes.timestamps)  # [1.2, 3.1]

.. tip::

   Note that, when reading from an HDF5 file, the data is not loaded into memory
   immediately. Instead, it is loaded on demand when you access an attribute.
   We discuss this further in :ref:`lazy_loading`.
