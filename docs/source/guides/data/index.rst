Representing and Storing Neural Data
====================================

NeuroAI training pipelines typically requires accessing time-based slices of
the neural recording.
For example, training a behavior decoder on trialized data might
involve slicing and loading data chunks around a trial-onset.
Or self-supervised learning, like MAE or contrastive learning, might involve
randomly sampling fixed-duration chunks of data from anywhere within the recording.
In TorchBrain, we want to support all these use-cases without requiring re-processing
or re-shaping of the underlying data on-disk.

To achieve this, we created our own data format which stores data *temporally* and
includes APIs optimized for *lazily loading* time-slices.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   meet_data_objects
   data_manipulation
   io


.. toctree::
   :maxdepth: 1
   :caption: Advanced Concepts

   interval_operations
   gappy_regular_ts
   lazy_loading
   advanced_interval_operations
