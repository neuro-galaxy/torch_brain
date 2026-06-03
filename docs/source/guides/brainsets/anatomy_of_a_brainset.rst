Anatomy of a Brainset
=====================

Brainset
--------

A **brainset** is a standardized collection of Sessions from one publication
or data source. On disk, a prepared brainset is a directory of Session H5
files under your configured processed directory, typically
``<processed_dir>/<brainset_id>/``.

Session
-------

Each **Session** is one H5 file representing a single data collection period.
In memory it is a :class:`~torch_brain.data.Data` object holding neural
signals, behavior, metadata, and a temporal **Domain**.

Brainset Pipeline
-----------------

A **Brainset Pipeline** is the code that downloads raw source data and
transforms each session into a standardized H5 file. Pipelines subclass
:class:`~torch_brain.pipeline.BrainsetPipeline` and are run via
``brainsets prepare``.

Dataset
-------

Each brainset provides a **Dataset** class in :mod:`torch_brain.datasets` —
a PyTorch-compatible loader that reads Session files and exposes
**Sampling Intervals** for use with **Samplers**. This is the interface you
use in training scripts.

Multiple brainsets can be combined with
:class:`~torch_brain.datasets.NestedDataset`, which namespaces recording IDs
as ``"<DatasetClassName>/<session_id>"``.

Metadata
--------

Session files carry structured metadata:

* :class:`~torch_brain.data.BrainsetDescription` — brainset-level provenance
  (``id``, ``origin_version``, ``derived_version``, ``source``, ``description``)
* :class:`~torch_brain.data.SubjectDescription` — subject identifiers and species
* :class:`~torch_brain.data.SessionDescription` — session identifiers and dates
* :class:`~torch_brain.data.DeviceDescription` — recording hardware

See :doc:`building_sessions` for how pipeline authors populate these fields.

Directory layout
----------------

After ``brainsets prepare my_brainset``:

.. code-block:: text

   <raw_dir>/my_brainset/          # downloaded source files
   <processed_dir>/my_brainset/    # Session H5 files
       session_1.h5
       session_2.h5
       ...

Pipeline source code lives in the repository at
``brainsets_pipelines/my_brainset/pipeline.py``.
