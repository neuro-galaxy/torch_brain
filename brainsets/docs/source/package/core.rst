.. currentmodule:: brainsets

brainsets
---------

Base Classes
~~~~~~~~~~~~

.. list-table::
   :widths: 25 125
   :align: left

   * - :obj:`StringIntEnum <brainsets.core.StringIntEnum>`
     - Base class for string-integer enums
   * - :obj:`Dictable <brainsets.core.Dictable>`
     - Base class for dataclasses that can be converted to dictionaries

.. autoclass:: brainsets.core.StringIntEnum
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: brainsets.core.Dictable
    :inherited-members:
    :show-inheritance:
    :undoc-members:


Serialization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: brainsets.core.string_int_enum_serialize_fn

.. autofunction:: brainsets.core.datetime_serialize_fn

.. autodata:: brainsets.core.serialize_fn_map
