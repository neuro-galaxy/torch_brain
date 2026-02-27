.. currentmodule:: torch_brain.transforms

torch_brain.transforms
======================

.. list-table::
   :widths: 25 125

   * - :py:class:`Compose`
     - Compose several transforms together.
   * - :py:class:`RandomChoice`
     - Apply a single transformation randomly picked from a list.
   * - :py:class:`ConditionalChoice` 
     - Conditionally apply a single transformation based on whether a condition is met.
   * - :py:class:`UnitDropout`
     - Randomly drop units from the `data.units` and `data.spikes`.
   * - :py:class:`RandomTimeScaling`
     - Randomly scales the time axis.
   * - :py:class:`RandomOutputSampler`
     - Randomly drops output samples.
   * - :py:class:`BinningTransform`
     - Bin spike events into fixed-width time bins.


.. autoclass:: Compose
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: RandomChoice
    :members:
    :show-inheritance:
    :undoc-members:


.. autoclass:: ConditionalChoice
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: UnitDropout
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: TriangleDistribution
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: RandomTimeScaling
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: RandomOutputSampler
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: BinningTransform
    :members:
    :show-inheritance:
    :undoc-members:
