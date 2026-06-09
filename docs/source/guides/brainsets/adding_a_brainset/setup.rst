Pipeline Setup
==============

Choose a ``brainset_id``
------------------------

Pick a unique identifier, typically ``{first_author}_{label}_{year}``. For
example, Perich et al. 2018 becomes ``perich_miller_population_2018``.

Create the pipeline directory
-----------------------------

A pipeline lives in a directory with at least a ``pipeline.py`` file::

   brainsets_pipelines/my_brainset/
   ├── pipeline.py
   └── ...              # optional supporting files

``pipeline.py`` contains your pipeline class and an inline metadata block
(Python version, dependencies) at the top. Add helper modules or session lists
alongside as needed.

``brainsets prepare`` reads the metadata, creates an isolated environment with
`uv <https://github.com/astral-sh/uv>`_, and runs the pipeline through
:mod:`torch_brain.pipeline.runner`.

Declare dependencies
--------------------

Pipelines declare dependencies using a
`PEP 723 <https://peps.python.org/pep-0723/>`__ inline metadata block at the
top of ``pipeline.py``:

.. code-block:: python

   # /// brainset-pipeline
   # python-version = "3.11"
   # dependencies = [
   #     "dandi==0.61.2",
   #     "scikit-learn==1.5.1",
   # ]
   # ///

The block must start with ``# /// brainset-pipeline`` and end with ``# ///``,
using TOML syntax (each line prefixed with ``#``).

Supported keys:

``python-version``
    Exact Python version string (e.g. ``"3.11"``). Ranges are not supported.

``dependencies``
    List of pip-installable package specifiers. Pin versions for reproducibility.

.. note::

   ``torch_brain`` is added to the environment automatically when not listed in
   ``dependencies``. Listing it explicitly is optional.

Next, see :doc:`pipeline` to implement the pipeline class.
