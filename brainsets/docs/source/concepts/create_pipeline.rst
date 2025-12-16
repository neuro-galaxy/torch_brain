Creating a :obj:`BrainsetPipeline`
====================================

.. py:currentmodule:: brainsets.pipeline
.. |BrainsetPipeline| replace:: :class:`BrainsetPipeline`
.. |get_manifest| replace:: :meth:`get_manifest <BrainsetPipeline.get_manifest>`
.. |download| replace:: :meth:`download <BrainsetPipeline.download>`
.. |process| replace:: :meth:`process <BrainsetPipeline.process>`

Neural datasets typically contain many recording sessions that can be downloaded and 
processed independently.
A |BrainsetPipeline| encodes the repeatable steps required to download raw
recordings and transform them into a processed brainset. Once you write a pipeline,
it becomes runnable through the ``brainsets`` CLI.

A pipeline implementation simply has to explain

* how to enumerate the available sessions (aka the "manifest"),
* how each session should be downloaded, and
* how to convert a downloaded session into processed artifacts.

Once the pipeline has been implemented, the rest (creating an isolated environment, parallel execution, and progress reporting)
is handled by the CLI and the pipeline runner.


Tutorial: build a pipeline from scratch
---------------------------------------

This tutorial walks through the full lifecycle of building a pipeline.
If you are brand new to the brainsets CLI, start with
`Using the brainsets CLI <using_existing_data.html>`__ to learn how to run existing pipelines.


Step 1 – Create a pipeline directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pipeline lives in a directory containing at least a ``pipeline.py`` file::

   my_brainset/
   ├── pipeline.py
   └── ...              # optional supporting files

``pipeline.py`` contains your pipeline class along with any metadata (Python version,
dependencies) declared inline at the top of the file. You may include additional
supporting files (helper modules, session lists, etc.) in the same directory.

The ``brainsets prepare`` command reads the metadata, creates an isolated environment
with the specified dependencies (using `uv <https://github.com/astral-sh/uv>`_), and
runs ``pipeline.py`` through :mod:`brainsets.runner`.


Step 2 – Subclass :class:`BrainsetPipeline`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside ``pipeline.py`` define your pipeline class. At minimum you must set a
unique ``brainset_id`` and implement |get_manifest|, |download|, and |process|.
Pipelines can also expose custom CLI arguments by attaching an
:class:`argparse.ArgumentParser` to the ``parser`` attribute.

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

    from brainsets.pipeline import BrainsetPipeline


    parser = ArgumentParser()
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    # add any custom arguments that seem fit for your pipeline


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

If your pipeline requires specific dependencies or a specific Python version,
add an inline metadata block at the very top of ``pipeline.py`` (see
:ref:`declaring-dependencies` below).

The `Perich & Miller (2018) pipeline <https://github.com/neuro-galaxy/brainsets/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
is a complete example that uses all of these hooks.


Step 3 – Gather and enumerate metadata with :meth:`get_manifest`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|get_manifest| constructs the manifest—a table describing every session your pipeline
needs to process. Each row provides the metadata needed to fetch and handle a single
recording.
The pipeline runner will iterate over the rows in this table,
downloading the processing each asset it describes.

|get_manifest| receives the path to the directory where raw data should be downloaded
and any arguments parsed from the CLI. 
It should return a :class:`pandas.DataFrame` indexed by a unique identifier, with one 
row per downloadable item. 
Columns can contain any metadata you find useful during download or processing.
When doing single-asset processing (``brainsets prepare my_brainset -s <manifest_item_index>``),
the CLI uses the index of the manifest directly, so keep it easy to understand.

Manifest rows are passed into |download| and (indirectly) |process|. A
minimal manifest might only contain an index and a ``url`` column.


The Perich & Miller manifest calculation demonstrates how to build this table
using the Dandi API:

.. code-block:: python

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        from dandi_utils import get_nwb_asset_list

        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]
        
        # Create a simple identifier for each item
        for m in manifest_list:
            m["session_id"] = ... 

        # Create a dataframe, set its index, and return
        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

Tips:

* Store enough information to make filenames deterministic.
* Use the ``raw_dir`` to cache manifest results if the source API is slow.
* Respect user arguments (for example, allowing the CLI to filter sessions).


Step 4 – Download each session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The |download| method receives one row of the manifest at a time. 
It should fetch raw data for that session and
return whatever object |process| needs to process this item. 

As an example, here we download the file referred in a given manifest row, and 
return the path to that file.

.. code-block:: python

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        file_path = self.raw_dir / manifest_item.path
        if file_path.exists() and not self.args.redownload:
            return file_path
        
        # code to download from url
        ...

        return file_path

Key things to remember:

* Use ``self.raw_dir`` to stash raw files; the directory already respects the
  user's CLI configuration.
* ``self.args`` exposes the pipeline-specific CLI flags (like ``--redownload`` above).
* Call ``self.update_status(...)`` to emit status updates visible in the CLI log.


Step 5 – Process into :obj:`Data` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|process| receives the object returned by |download|, converts
it into processed :obj:`temporaldata.Data` object(s), and stores these inside ``self.processed_dir``.

.. code-block:: python

    def process(self, fpath):
        self.update_status("Loading file")
        ...

        output_file_path = self.processed_dir / f"{session_id}.h5" 
        if output_file_path.exists() and not self.args.reprocess:
            return

        self.update_status("Extracting Neural Activity")
        ...

        self.update_status("Extracting Stimulus")
        ...

        # create data object
        data = Data(...)

        # split data into train, validation and test
        self.update_status("Creating splits")
        data.set_train_domain(...)
        data.set_valid_domain(...)
        data.set_test_domain(...)

        # save data to disk
        self.update_status("Storing")
        with h5py.File(output_file_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


Most of the logic for implementing the |process| method will follow the tutorial 
`Preparing a new Dataset <prepare_data.html>`_.

Best practices:

* Use ``self.processed_dir`` for writing data in the configured space.
* Gate reprocessing with CLI flags like ``--reprocess``.
* Call ``self.update_status(...)`` to emit status updates visible in the CLI log.


Step 6 – Run the pipeline with the CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once your class is in place you can run it through the CLI.

.. code-block:: console

   $ brainsets prepare my_brainset --cores 8
   Preparing my_brainset...
   Raw data directory: /path/to/raw
   Processed data directory: /path/to/processed
   Building temporary virtual environment for .../pipeline.py
   ...


For local development outside the ``brainsets`` repository, you can point the CLI to any pipeline directory by adding
``--local``. 

.. code-block:: console

   $ brainsets prepare /path/to/my_brainset --local --cores 8



While developing:

* Use ``-s <manifest_index>`` while development for quick debugging on a single pipeline run.
* To avoid creating a temporary environment during early development, use the ``--use-active-env`` to stay within your current Python environment.


Once your pipeline is reliable, commit the new directory inside ``brainsets_pipelines``, and submit a Pull Request.


.. _declaring-dependencies:

Declaring dependencies and Python version
-----------------------------------------

Pipelines declare their dependencies and Python version using a
`PEP 723 <https://peps.python.org/pep-0723/>`__-style inline metadata block
at the top of ``pipeline.py``. This block is parsed by the CLI to create an
isolated environment before running your pipeline.

.. code-block:: python

    # /// brainset-pipeline
    # python-version = "3.11"
    # dependencies = [
    #     "dandi==0.61.2",
    #     "scikit-learn==1.5.1",
    # ]
    # ///

    from argparse import ArgumentParser
    ...

The metadata block must:

* Start with ``# /// brainset-pipeline`` and end with ``# ///``
* Use TOML syntax for the content (each line prefixed with ``#`` or ``# ``)

Supported keys:

``python-version``
    A single Python version string (e.g., ``"3.10"``, ``"3.11"``). Version ranges
    are not supported—specify the exact version your pipeline needs.

``dependencies``
    A list of pip-installable package specifiers. Pin versions for reproducibility.

.. note::

    The ``brainsets`` package itself is automatically added to the environment
    if not explicitly listed in ``dependencies``. Not adding ``brainsets`` to the 
    dependencies list is the recommended practice.
