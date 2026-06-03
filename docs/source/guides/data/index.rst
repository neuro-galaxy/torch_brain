Representing Neural Data
========================

NeuroAI training pipelines typically require accessing time-based slices of
neural recordings.
For example, training a behavior decoder on trialized data might involve
slicing and loading data chunks around trial-onsets.
Self-supervised approaches, like masked autoencoding or contrastive learning,
might instead involve randomly sampling fixed-duration chunks from anywhere
within the recording.

In TorchBrain, we want to support all of these use-cases without requiring
re-processing or re-shaping of the underlying data on-disk.
To achieve this, we created our own data format that stores data *temporally*
and provides APIs optimized for *lazily loading* time-slices.

This guide walks through the core data objects, how data is stored and loaded
from disk, what lazy-loading means, and more.

.. toctree::
   :maxdepth: 1

   data_objects_intro
   io
   domain
   gappy_regular_ts
   interval_ops/index

