.. currentmodule:: torch_brain.utils.stitcher

Stitcher
--------

The stitcher module provides utilities for combining predictions from overlapping 
windows during evaluation. This is essential when using sliding window inference.

Functions
~~~~~~~~~

.. list-table::
   :widths: 25 125

   * - :py:func:`stitch`
     - Pool predictions by timestamp using mean (continuous) or mode (categorical).

.. autofunction:: stitch

Evaluation Callbacks
~~~~~~~~~~~~~~~~~~~~

These Lightning callbacks handle stitching and metric computation during validation/test.

.. list-table::
   :widths: 25 125

   * - :py:class:`DecodingStitchEvaluator`
     - Single-task evaluation callback with stitching.
   * - :py:class:`MultiTaskDecodingStitchEvaluator`
     - Multi-task evaluation callback with stitching.

.. autoclass:: DecodingStitchEvaluator
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: MultiTaskDecodingStitchEvaluator
    :members:
    :show-inheritance:
    :undoc-members:

Data Containers
~~~~~~~~~~~~~~~

Dataclasses for passing data to evaluation callbacks.

.. list-table::
   :widths: 25 125

   * - :py:class:`DataForDecodingStitchEvaluator`
     - Data container for single-task evaluation.
   * - :py:class:`DataForMultiTaskDecodingStitchEvaluator`
     - Data container for multi-task evaluation.

.. autoclass:: DataForDecodingStitchEvaluator
    :members:
    :undoc-members:

.. autoclass:: DataForMultiTaskDecodingStitchEvaluator
    :members:
    :undoc-members:

