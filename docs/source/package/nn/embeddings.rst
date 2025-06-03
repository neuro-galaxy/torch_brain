.. currentmodule:: torch_brain.nn


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
