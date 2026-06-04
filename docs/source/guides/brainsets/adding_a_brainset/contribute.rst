Run and Contribute
==================

Run the pipeline
----------------

.. code-block:: console

   $ brainsets prepare my_brainset --cores 8

For local development, point to any pipeline directory with ``--local``:

.. code-block:: console

   $ brainsets prepare /path/to/my_brainset --local --cores 8

Development tips:

* Use ``--single <manifest_index>`` to process one Session while debugging.
* Use ``--use-active-env`` to skip the temporary virtual environment.

Contribute to ``torch_brain``
-----------------------------

When your pipeline is ready:

1. Add it under ``brainsets_pipelines/<brainset_id>/`` in the ``torch_brain``
   repository.
2. Follow the ``{author}_{label}_{year}`` naming convention for ``brainset_id``.
3. Add a corresponding **Dataset** class in :mod:`torch_brain.datasets`.
4. Open a pull request.

After merge, anyone can prepare the brainset with ``brainsets prepare``; see
:doc:`../getting_started/index`.
