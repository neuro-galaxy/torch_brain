.. currentmodule:: torch_brain.registry

torch_brain.registry
====================

.. list-table::
   :widths: 25 125
   
   * - :py:class:`DataType`
     - Enum defining possible data types for modalities.
   * - :py:class:`ModalitySpec`
     - Specification class for defining modalities.
   * - :py:func:`register_modality`
     - Register a new modality specification in the global registry.
   * - :py:func:`get_modality_by_id`
     - Get a modality specification by its ID.

.. autoclass:: DataType
    :members:
    :undoc-members:

.. autoclass:: ModalitySpec
    :members:
    :undoc-members:

.. autofunction:: register_modality

.. autofunction:: get_modality_by_id
