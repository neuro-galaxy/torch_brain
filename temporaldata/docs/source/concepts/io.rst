I/O Operations
--------------

Data Persistence
===============

All data objects in temporaldata can be saved to and loaded from HDF5 files. This provides an efficient way to store and retrieve large datasets.

Saving Data
~~~~~~~~~~~~~~~~

To save a data object to disk, use the ``to_hdf5`` method::

    import h5py
    from temporaldata import RegularTimeSeries, Data
    
    # Create a regular time series
    data = RegularTimeSeries(
        raw=np.zeros((1000, 128)),
        sampling_rate=250.,
        domain=Interval(0., 4.)
    )
    
    # Save to HDF5
    with h5py.File("data.h5", "w") as f:
        data.to_hdf5(f)

The data structure is preserved in the HDF5 file, including all attributes and metadata.

Lazy Loading
~~~~~~~~~~~~~~~~

For large datasets, loading everything into memory might not be feasible. temporaldata provides lazy loading capabilities through the ``Lazy`` variants of its data classes (``LazyRegularTimeSeries``, ``LazyIrregularTimeSeries``, ``LazyInterval``).

Time-Based Lazy Loading
~~~~~~~~~~~~~~~~

When working with time series data, you can load only specific time windows without loading the entire dataset::

    with h5py.File("data.h5", "r") as f:
        # Create a lazy reference to the data
        lazy_data = LazyRegularTimeSeries.from_hdf5(f)
        
        # Select a time window - this doesn't load the data yet
        subset = lazy_data.select(start=0.0, end=2.0)
        
        # Data is only loaded when actually accessed
        print(subset.raw)  # This triggers the loading

Attribute-Based Lazy Loading
~~~~~~~~~~~~~~~~

Lazy loading also works on a per-attribute basis. Data is only loaded when a specific attribute is accessed::

    with h5py.File("data.h5", "r") as f:
        lazy_data = LazyIrregularTimeSeries.from_hdf5(f)
        
        # This doesn't load any data
        print(lazy_data.keys())
        
        # Only the 'timestamps' attribute is loaded
        print(lazy_data.timestamps)
        
        # 'raw' is still not loaded until accessed
        print(lazy_data.raw)  # Now 'raw' is loaded

Implementation Details
~~~~~~~~~~~~~~~~

The lazy loading mechanism works through several key components:

1. **Delayed Loading**: When a lazy object is created from an HDF5 file, it only stores references to the datasets, not the actual data.

2. **Attribute Interception**: The ``__getattribute__`` method intercepts attribute access and performs loading only when needed.

3. **Operation Queueing**: Operations like slicing and masking are queued in ``_lazy_ops`` and only applied when the data is actually loaded::

    lazy_data = LazyIrregularTimeSeries.from_hdf5(f)
    masked_data = lazy_data.select_by_mask(mask)  # Operation is queued
    print(masked_data.raw)  # Loading happens here, with mask applied

Best Practices
~~~~~~~~~~~~~~~~

1. Always use ``with`` statements when working with HDF5 files to ensure proper file handling.

2. Use lazy loading when:
   - Working with large datasets
   - Only needing specific time windows
   - Accessing only a subset of attributes

3. Consider memory constraints when deciding between regular and lazy variants.

4. Keep HDF5 files in read-only mode when using lazy loading to prevent concurrent modification issues.
