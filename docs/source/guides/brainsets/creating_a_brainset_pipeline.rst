Creating a Brainset Pipeline
============================

.. py:currentmodule:: torch_brain.pipeline
.. |BrainsetPipeline| replace:: :class:`BrainsetPipeline`
.. |get_manifest| replace:: :meth:`get_manifest <BrainsetPipeline.get_manifest>`
.. |download| replace:: :meth:`download <BrainsetPipeline.download>`
.. |process| replace:: :meth:`process <BrainsetPipeline.process>`

This tutorial explains how to implement a |BrainsetPipeline|: the code that
enumerates Sessions (the manifest), downloads each Session, and converts raw
data into standardized Session H5 files. Run the result with
``brainsets prepare``; the CLI and :mod:`torch_brain.pipeline.runner` handle
isolated environments, parallelism, and progress reporting.

Before starting, see :doc:`getting_started` for storage configuration and
``brainsets prepare`` usage.


Build a Brainset Pipeline
-------------------------


Step 1 — Create a pipeline directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Step 2 — Subclass :class:`BrainsetPipeline`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a unique ``brainset_id`` and implement |get_manifest|, |download|, and
|process|. Attach an :class:`argparse.ArgumentParser` to ``parser`` for
custom CLI flags.

.. code-block:: python

   # /// brainset-pipeline
   # python-version = "3.11"
   # dependencies = [
   #     "dandi==0.61.2",
   #     "scikit-learn==1.5.1",
   # ]
   # ///

   from argparse import ArgumentParser
   import pandas as pd

   from torch_brain.pipeline import BrainsetPipeline


   parser = ArgumentParser()
   parser.add_argument("--redownload", action="store_true")
   parser.add_argument("--reprocess", action="store_true")


   class Pipeline(BrainsetPipeline):
       brainset_id = "my_brainset"
       parser = parser

       @classmethod
       def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
           ...

       def download(self, manifest_item):
           ...

       def process(self, download_output):
           ...

See :ref:`declaring-dependencies` for the metadata block format.

The `Perich & Miller (2018) pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
is a complete reference implementation.


Step 3 — Build the manifest with :meth:`get_manifest`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|get_manifest| returns a :class:`pandas.DataFrame` with one row per Session to
process. The index should be a unique session identifier; columns hold whatever
metadata |download| and |process| need.

.. code-block:: python

   @classmethod
   def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
       from torch_brain.utils.dandi import get_nwb_asset_list

       asset_list = get_nwb_asset_list(cls.dandiset_id)
       manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]

       for m in manifest_list:
           m["session_id"] = ...

       return pd.DataFrame(manifest_list).set_index("session_id")

Tips:

* Store enough information to make output filenames deterministic.
* Cache manifest results under ``raw_dir`` if the source API is slow.
* Respect CLI arguments (e.g. filtering Sessions).


Step 4 — Download each Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|download| receives one manifest row, fetches raw data, and returns whatever
|process| needs (commonly a file path).

.. code-block:: python

   def download(self, manifest_item):
       self.update_status("DOWNLOADING")

       file_path = self.raw_dir / manifest_item.path
       if file_path.exists() and not self.args.redownload:
           return file_path

       # download from manifest_item.url
       ...

       return file_path

Use ``self.raw_dir`` for raw files and ``self.args`` for pipeline-specific
flags. Call ``self.update_status(...)`` for CLI progress output.


Step 5 — Process into Session files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|process| converts downloaded data into a Session :obj:`~torch_brain.data.Data`
object and writes H5 under ``self.processed_dir``.

.. code-block:: python

   def process(self, fpath):
       self.update_status("Loading file")
       ...

       output_file_path = self.processed_dir / f"{session_id}.h5"
       if output_file_path.exists() and not self.args.reprocess:
           return

       self.update_status("Extracting neural activity")
       ...

       data = Data(...)

       self.update_status("Storing")
       with h5py.File(output_file_path, "w") as file:
           data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

See :doc:`building_sessions` for populating the ``Data`` object.


Step 6 — Run and contribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ brainsets prepare my_brainset --cores 8

For local development, point to any pipeline directory with ``--local``:

.. code-block:: console

   $ brainsets prepare /path/to/my_brainset --local --cores 8

Development tips:

* Use ``--single <manifest_index>`` to process one Session while debugging.
* Use ``--use-active-env`` to skip the temporary virtual environment.

When your pipeline is ready, add it under ``brainsets_pipelines/<brainset_id>/``
and open a pull request to the ``torch_brain`` repository. Follow the
``{author}_{label}_{year}`` naming convention for ``brainset_id`` and include a
corresponding **Dataset** class in :mod:`torch_brain.datasets`.


.. _declaring-dependencies:

Declaring dependencies and Python version
-----------------------------------------

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


Specialized pipeline base classes
---------------------------------

For common data sources, ``torch_brain`` provides specialized base classes.
See :doc:`specialized_pipelines/index` for available options.
