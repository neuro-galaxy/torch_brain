.. currentmodule:: torch_brain.data

Collate
-------
.. currentmodule:: torch_brain.data.collate

.. list-table::
   :widths: 25 125

   * - :py:func:`collate`
     - An extended collate function that handles padding and chaining.
   * - :py:func:`pad`
     - A wrapper to call when padding.
   * - :py:func:`track_mask`
     - A wrapper to call to track the padding mask during padding.
   * - :py:func:`pad8`
     - A wrapper to call when padding, but length is rounded up to the nearest multiple of 8. 
   * - :py:func:`track_mask8`
     - A wrapper to call to track the padding mask during padding with :py:func:`pad8`.
   * - :py:func:`pad2d`
     - A wrapper to call when padding 2D tensors.
   * - :py:func:`chain`
     - A wrapper to call when chaining.
   * - :py:func:`track_batch`
     - A wrapper to call to track the batch index during chaining.


.. autofunction:: collate

.. autofunction:: pad

.. autofunction:: track_mask

.. autofunction:: pad8

.. autofunction:: track_mask8

.. autofunction:: pad2d

.. autofunction:: chain

.. autofunction:: track_batch
