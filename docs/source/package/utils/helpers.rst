.. currentmodule:: torch_brain.utils

Helper Functions
----------------

Reproducibility
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 125

   * - :py:func:`seed_everything`
     - Set random seeds for reproducibility across all libraries.

.. autofunction:: seed_everything

Encoding
~~~~~~~~

.. list-table::
   :widths: 25 125

   * - :py:func:`get_sinusoidal_encoding`
     - Generate 2D sinusoidal position encodings.

.. autofunction:: get_sinusoidal_encoding

Interval Utilities
~~~~~~~~~~~~~~~~~~

.. currentmodule:: torch_brain.utils.weights

.. list-table::
   :widths: 25 125

   * - :py:func:`isin_interval`
     - Check if timestamps fall within intervals.
   * - :py:func:`resolve_weights_based_on_interval_membership`
     - Compute sample weights based on interval membership.

.. autofunction:: isin_interval

.. autofunction:: resolve_weights_based_on_interval_membership

Readout Utilities
~~~~~~~~~~~~~~~~~

.. currentmodule:: torch_brain.utils.readout

.. list-table::
   :widths: 25 125

   * - :py:func:`prepare_for_readout`
     - Prepare data for single-task readout.

.. autofunction:: prepare_for_readout

