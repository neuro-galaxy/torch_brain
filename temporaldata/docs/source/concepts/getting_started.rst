When to use temporaldata
------------------------

**temporaldata** is a Python package for working with temporal data. In particular it is
designed for complex, multi-modal, multi-resolution data. 

Working with temporal data in real-world applications presents several challenges:

- **Multi-resolution data**: Different sensors often record at different frequencies (e.g., 1000Hz accelerometer data alongside 30Hz video)
- **Missing or irregular data**: Gaps in recordings, manual measurements, or variable sampling rates
- **Complex relationships**: Multiple data streams that need to be aligned and analyzed together
- **Large datasets**: Efficient access and processing of temporal segments

Traditional approaches using simple array operations become cumbersome when dealing with these challenges. For example:

.. code-block:: python
    :linenos:

    # Traditional approach has limitations
    import numpy as np
    
    # Need separate arrays for different sensors
    accelerometer = np.random.randn(1000000)  # 1000Hz
    video = np.random.randn(30000)  # 30Hz
    manual_readings = np.random.randn(10)  # Sporadic
    
    # Difficult to align and slice across different timescales
    # How do we get corresponding data for a 5-second window?

temporaldata provides a solution by:

1. Offering flexible data structures that handle multi-resolution and irregular data naturally
2. Providing efficient temporal indexing across all data streams
3. Maintaining temporal relationships between different data sources
4. Enabling fast, on-the-fly access to temporal segments without preprocessing

The package is particularly useful for researchers and engineers working with sensor data, time series analysis, or any application involving complex temporal data structures.
