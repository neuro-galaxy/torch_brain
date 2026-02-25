I/O Operations
--------------

All data objects in **temporaldata** can be saved to and loaded from HDF5 files. This provides an efficient way to store and retrieve large datasets.


Writing
~~~~~~~


To save a data object to disk, use the ``save`` method:

.. tab:: Generic

    .. code-block:: python

        from temporaldata import RegularTimeSeries, IrregularTimeSeries, Data, Interval
        import numpy as np

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
                domain=Interval(start=0, end=4)
            ),
            user_id='user123',
            device='laptop'
        )

        # Save to a HDF5 file on disk
        user_session.save("user_data.h5")

.. tab:: Neuroscience

    .. code-block:: python

        from temporaldata import RegularTimeSeries, IrregularTimeSeries, Data, Interval
        import numpy as np

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
                domain=Interval(start=0, end=4)
            ),
            subject_id='mouse1',
            date='2023-01-01'
        )

        # Save to a HDF5 file on disk
        session.save("neural_data.h5")

The data structure is preserved in the HDF5 file, including all attributes and metadata.

Reading
~~~~~~~

To read data from an HDF5 file, use the ``load`` method:

.. tab:: Generic

    .. code-block:: python

        # Read from HDF5 file on disk
        user_session = Data.load("user_data.h5")
        
        # Access data as normal
        print(user_session.clicks.timestamps)  # [1.2, 2.3, 3.1]
        print(user_session.sensor.sampling_rate)  # 100
        print(user_session.user_id)  # 'user123'

        # Perform operations
        subset = user_session.clicks.slice(0, 2.0)
        print(subset.timestamps)  # [1.2]

.. tab:: Neuroscience

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

The loaded objects maintain all the functionality of the original objects, allowing you to perform operations, slicing, and access all attributes.

Note that, when reading from an HDF5 file, the data is not loaded into memory immediately. 
Instead, it is loaded on demand when you access an attribute. This lazy loading mechanism 
allows you to work with large datasets without loading the entire file into memory at once. 
For more details, see :ref:`lazy_loading`.