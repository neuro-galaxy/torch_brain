.. currentmodule:: torch_brain.data

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

.. autofunction:: track_mask8

.. autofunction:: chain

.. autofunction:: track_batch
