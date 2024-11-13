.. _lazy_loading:

Lazy Loading
============

**temporaldata** uses hdf5 to store and load data. When loading data from hdf5, **temporaldata** uses lazy loading to defer loading data until it is needed:

- Loading data only when attributes are accessed
- Lazy slicing without loading the entire data object by creating an internal cached kd-tree for fast timestamp lookup
- Deferring operations like slicing and masking until data is needed (when an attribute is accessed)
- Managing an internal queue of operations (successive slices, masks, etc.) that is only resolved when the attribute is accessed

Attribute-Based Lazy Loading
----------------------------

Data is loaded on a per-attribute basis. Only the attributes you access are loaded into memory:

.. tab:: Generic

    .. code-block:: python

        with h5py.File("user_data.h5", "r") as f:
            user_session = Data.from_hdf5(f)
            
            # No data loaded, just returns attribute names
            print(user_session.keys())  # ['clicks', 'sensor', 'user_id', 'device']
            
            # Only loads timestamps array from clicks
            print(user_session.clicks.timestamps)  # [1.2, 2.3, 3.1]
            
            # Only loads accelerometer data when accessed
            print(user_session.sensor.accelerometer)

.. tab:: Neuroscience

    .. code-block:: python

        with h5py.File("neural_data.h5", "r") as f:
            session = Data.from_hdf5(f)
            
            # No data loaded, just returns attribute names
            print(session.keys())  # ['spikes', 'lfp', 'subject_id', 'date']
            
            # Only loads timestamps array from spikes
            print(session.spikes.timestamps)  # [1.2, 2.3, 3.1]
            
            # Only loads raw LFP data when accessed
            print(session.lfp.raw)

Time-Based Lazy Loading
-----------------------

For time series data, you can efficiently load specific time windows without loading the entire dataset:

.. tab:: Generic

    .. code-block:: python

        with h5py.File("user_data.h5", "r") as f:
            user_session = Data.from_hdf5(f)
            
            # Define time window without loading
            window = user_session.slice(start=0.0, end=2.0)
            
            # Data loaded only for requested window
            print(window.clicks.timestamps)  # [1.2]
            print(window.sensor.accelerometer)  # First 200 samples

.. tab:: Neuroscience

    .. code-block:: python

        with h5py.File("neural_data.h5", "r") as f:
            session = Data.from_hdf5(f)
            
            # Define time window without loading
            window = session.slice(start=0.0, end=2.0)
            
            # Data loaded only for requested window
            print(window.spikes.timestamps)  # [1.2]
            print(window.lfp.raw)  # First 2000 samples

Best Practices
--------------

1. Always use ``with`` statements when working with HDF5 files to ensure proper file handling

2. Use lazy loading when:
   - Working with large datasets that may not fit in memory
   - Only needing specific time windows or attributes
   - Performing multiple operations before accessing data

3. Consider materializing the data using ``.materialize()`` if you need to load the full data object into memory
