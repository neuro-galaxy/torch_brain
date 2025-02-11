.. currentmodule:: torch_brain.data

torch_brain.data
================


Dataset
-------

.. currentmodule:: torch_brain.data.dataset

.. autoclass:: Dataset
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: DatasetIndex
    :members:
    :show-inheritance:
    :undoc-members:

Collate
-------
.. currentmodule:: torch_brain.data.collate

.. list-table::
   :widths: 25 125

   * - :py:class:`collate`
     - An extended collate function that handles padding and chaining.
   * - :py:class:`pad`
     - A wrapper to call when padding.
   * - :py:class:`track_mask`
     - A wrapper to call to track the padding mask during padding.
   * - :py:class:`pad8`
     - A wrapper to call when padding, but length is rounded up to the nearest multiple of 8. 
   * - :py:class:`track_mask8`
     - A wrapper to call to track the padding mask during padding with :py:class:`pad8`.
   * - :py:class:`chain`
     - A wrapper to call when chaining.
   * - :py:class:`track_batch`
     - A wrapper to call to track the batch index during chaining.


.. autofunction:: collate

.. autofunction:: pad

.. autofunction:: track_mask

.. autofunction:: pad8

.. autofunction:: trach_mask8

.. autofunction:: chain

.. autofunction:: track_batch

Samplers
--------
.. currentmodule:: torch_brain.data.sampler

.. list-table::
   :widths: 25 125

   * - :py:class:`SequentialFixedWindowSampler`
     - A Sequential sampler, that samples a fixed-length window from data.
   * - :py:class:`RandomFixedWindowSampler`
     - A Random sampler, that samples a fixed-length window from data.
   * - :py:class:`TrialSampler`
     - A sampler that randomly samples a single trial interval from given intervals.
   * - :py:class:`DistributedSamplerWrapper`
     - A wrapper sampler for distributed training that assigns samples to processes.
   * - :py:class:`DistributedStitchingFixedWindowSampler`
     - A distributed sampler for evaluation that enables sliding window inference with prediction stitching.


.. autoclass:: SequentialFixedWindowSampler
  :members:
  :show-inheritance:
  :undoc-members:

.. autoclass:: RandomFixedWindowSampler
  :members:
  :show-inheritance:
  :undoc-members:

.. autoclass:: TrialSampler
  :members:
  :show-inheritance:
  :undoc-members:

.. autoclass:: DistributedSamplerWrapper
  :members:
  :show-inheritance:
  :undoc-members:

.. autoclass:: DistributedStitchingFixedWindowSampler
  :members:
  :show-inheritance:
  :undoc-members: