.. currentmodule:: torch_brain.nn

torch_brain.nn
==============


Embeddings Layers
-----------------

.. list-table::
   :widths: 25 125

   * - :py:class:`Embedding`
     - A simple embedding layer.
   * - :py:class:`InfiniteVocabEmbedding`
     - An extendable embedding layer + tokenizer.
   * - :py:class:`RotaryEmbedding`
     - Rotary embedding layer.
  
.. autoclass:: Embedding
    :members:
    :undoc-members:

.. autoclass:: InfiniteVocabEmbedding
    :members:
    :undoc-members:
    :exclude-members: forward, extra_repr, initialize_parameters

.. autoclass:: RotaryEmbedding
    :members:
    :undoc-members:

.. autofunction:: apply_rotary_pos_emb

Transformer modules
-------------------

.. list-table::
   :widths: 25 125
   * - :py:class:`FeedForward`
     - Feed-forward network with GEGLU activation.
   * - :py:class:`RotaryCrossAttention`
     - Rotary cross-attention layer.
   * - :py:class:`RotarySelfAttention`
     - Rotary self-attention layer.


.. autoclass:: FeedForward
    :members:
    :undoc-members:

.. autoclass:: RotaryCrossAttention
    :members:
    :undoc-members: 

.. autoclass:: RotarySelfAttention
    :members:
    :undoc-members:


Readout Layers
--------------

.. list-table::
   :widths: 25 125
   * - :py:class:`MultitaskReadout`
     - A multi-task readout module.
   * - :py:func:`prepare_for_multitask_readout`
     - Tokenizer function for :py:class:`MultitaskReadout`.

.. autoclass:: MultitaskReadout
    :members:
    :show-inheritance:
    :undoc-members:

.. autofunction:: prepare_for_multitask_readout
