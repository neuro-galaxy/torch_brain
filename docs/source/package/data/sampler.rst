
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