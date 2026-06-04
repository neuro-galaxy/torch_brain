Implement the Pipeline
======================

.. py:currentmodule:: torch_brain.pipeline
.. |BrainsetPipeline| replace:: :class:`BrainsetPipeline`
.. |get_manifest| replace:: :meth:`get_manifest <BrainsetPipeline.get_manifest>`
.. |download| replace:: :meth:`download <BrainsetPipeline.download>`
.. |process| replace:: :meth:`process <BrainsetPipeline.process>`

A |BrainsetPipeline| defines three steps: enumerate Sessions (the manifest),
download each Session, and convert raw data into Session H5 files in |process|.
The CLI and :mod:`torch_brain.pipeline.runner` handle isolated environments,
parallelism, and progress reporting.

Subclass :class:`BrainsetPipeline`
----------------------------------

Set a unique ``brainset_id`` and implement |get_manifest|, |download|, and
|process|. Attach an :class:`argparse.ArgumentParser` to ``parser`` for custom
CLI flags.

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

The `Perich & Miller (2018) pipeline <https://github.com/neuro-galaxy/torch_brain/blob/main/brainsets_pipelines/perich_miller_population_2018/pipeline.py>`__
is a complete reference implementation.

Build the manifest
------------------

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

* Store enough information to make output filenames deterministic.
* Cache manifest results under ``raw_dir`` if the source API is slow.
* Respect CLI arguments (e.g. filtering Sessions).

Download each Session
---------------------

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

In |process|, build a :class:`~torch_brain.data.Data` object and write H5; see
:doc:`build_session` and :doc:`save_to_disk`.
