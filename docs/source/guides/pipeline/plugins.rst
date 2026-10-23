.. _brainsets_plugins:

Writing a brainsets plugin
==========================

A *brainsets plugin* is an installable Python package that adds brainsets to the
``brainsets`` CLI. Use one to publish pipelines (and their Dataset classes) that you
want to maintain yourself, outside of torch_brain.

A plugin does not need to be on PyPI: installing it from git or from a local path
works the same way.

Once installed, a plugin's brainsets show up in ``brainsets list`` and are prepared
with their namespace:

.. code-block:: console

    $ uv add "my_brainsets @ git+https://github.com/me/my_brainsets"
    $ brainsets list
    $ brainsets prepare my_brainsets:my_dataset_2026

Plugins listed in the :ref:`community brainsets registry <community_brainsets>` are
also visible before they are installed (``brainsets list --all``,
``brainsets info <namespace>``).


Requirements
------------

``brainsets check <namespace>`` checks the rules marked *(checked)* without
downloading anything.

1. **Installable package.** It has a ``pyproject.toml`` and installs with ``pip`` or
   ``uv`` from any source (PyPI, ``git+https://...``, a local path).

2. **Entry point** *(checked)*. It declares exactly one entry point in the
   ``brainsets.pipelines`` group. The entry point name is the plugin's *namespace*:
   lowercase letters, digits and underscores, unique among installed plugins, and not
   ``torch_brain``. The value is an importable package (not an object) that holds the
   pipelines.

   .. code-block:: toml

       [project.entry-points."brainsets.pipelines"]
       my_brainsets = "my_brainsets.pipelines"

3. **Layout** *(checked)*. That package has one folder per brainset,
   ``<brainset_id>/pipeline.py``. Folders starting with ``_`` or ``.`` are ignored. The
   folders must be installed with the package, so ship them as package data:

   .. code-block:: toml

       [tool.setuptools.package-data]
       "my_brainsets.pipelines" = ["*/pipeline.py"]

4. **Pipeline class** *(checked)*. Each ``pipeline.py`` defines
   ``class Pipeline(torch_brain.pipeline.BrainsetPipeline)`` whose ``brainset_id``
   equals the folder name, and implements ``get_manifest``, ``download`` and
   ``process`` (see :doc:`create_pipeline`).

5. **Inline metadata** *(checked)*. The optional ``# /// brainset-pipeline`` block may
   only set ``python-version`` (a single version, e.g. ``"3.11"``) and
   ``dependencies`` (see :ref:`declaring-dependencies`). Do not list torch_brain or
   your plugin there: the CLI adds the versions the user has installed, from the same
   source (PyPI, git commit or local path).

6. **Output.** ``process`` writes one ``<recording_id>.h5`` per recording into
   ``self.processed_dir``, which is ``<processed_dir>/<brainset_id>``, using
   ``Data.to_hdf5``. Each file has at least ``brainset``, ``subject``, ``session`` and
   ``domain``.

7. **Split names.** If your Dataset classes expose splits, use ``"train"``,
   ``"valid"`` and ``"test"``, so they work in a
   :class:`~torch_brain.datasets.NestedDataset` next to torch_brain's datasets.


Names and output folders
------------------------

Plugin brainsets are always addressed as ``<namespace>:<brainset_id>`` on the command
line. Bare names (and ``torch_brain:<brainset_id>``) refer to torch_brain's built-in
brainsets.

Output folders are not namespaced: a brainset is written to
``<processed_dir>/<brainset_id>``, as built-in brainsets are. ``prepare`` writes a
small ``.brainsets_source`` file in that folder naming the package that made it, and
refuses to write into a folder made by a different package. Folders without this file,
or made by torch_brain, are accepted. Pick a ``brainset_id`` that is unlikely to clash
with others, e.g. ``<author>_<topic>_<year>``.


Contract with torch_brain
-------------------------

- Declare a torch_brain version range in your package's dependencies, e.g.
  ``torch_brain>=0.2.1,<0.3``. The pipeline's isolated environment resolves it with
  your plugin.
- Only import torch_brain's public API: :mod:`torch_brain.data`,
  :mod:`torch_brain.datasets`, :class:`torch_brain.pipeline.BrainsetPipeline` and the
  public functions of :mod:`torch_brain.utils` (for example
  :mod:`~torch_brain.utils.mne`, :mod:`~torch_brain.utils.s3`,
  :mod:`~torch_brain.utils.bids` and :mod:`~torch_brain.utils.split`). Names starting
  with ``_`` can change without notice.


Recommended
-----------

- Ship a Dataset class per brainset (or one per family) that subclasses
  :class:`torch_brain.datasets.Dataset`, reads ``<root>/<brainset_id>/`` and is
  importable from your package's top level.
- Make pipelines idempotent: skip what is already downloaded or processed, unless the
  user passes flags such as ``--redownload`` or ``--reprocess``.
- While developing, install your package in editable mode (``uv pip install -e .``)
  and run ``brainsets prepare <namespace>:<brainset_id> --single <item> --use-active-env``.
- Run ``brainsets check <namespace>`` in your CI.


Minimal example
---------------

.. code-block:: text

    my_brainsets/
        pyproject.toml
        src/my_brainsets/
            __init__.py                  # exports MyDataset2026
            dataset.py
            pipelines/
                __init__.py
                my_dataset_2026/
                    pipeline.py          # class Pipeline(BrainsetPipeline)

.. code-block:: toml

    [project]
    name = "my_brainsets"
    version = "0.1.0"
    dependencies = ["torch_brain>=0.2.1,<0.3"]

    [project.entry-points."brainsets.pipelines"]
    my_brainsets = "my_brainsets.pipelines"

    [tool.setuptools.package-data]
    "my_brainsets.pipelines" = ["*/pipeline.py"]

Users then run:

.. code-block:: console

    $ uv add "my_brainsets @ git+https://github.com/me/my_brainsets"
    $ brainsets prepare my_brainsets:my_dataset_2026

.. code-block:: python

    from my_brainsets import MyDataset2026

    dataset = MyDataset2026(root="/data/processed")

`openneuro_brainsets <https://github.com/milosobral/openneuro_brainsets>`__ is a
complete example.


.. _brainsets_plugins_listing:

Getting listed
--------------

Listing a plugin in torch_brain's registry
(`torch_brain/pipeline/plugins.toml <https://github.com/neuro-galaxy/torch_brain/blob/main/torch_brain/pipeline/plugins.toml>`__)
makes it visible on the :ref:`community brainsets <community_brainsets>` page and in
``brainsets list --all`` / ``brainsets info``, before users install it. The registry
also reserves your namespace.

1. Tag a release of your plugin.
2. Open a pull request that adds a table for your plugin to ``plugins.toml``, following
   the schema at the top of that file, with the install spec pinned to that tag (or a
   PyPI version). Use the
   `community plugin PR template <https://github.com/neuro-galaxy/torch_brain/compare/main...main?template=community_plugin.md>`__.
3. CI installs your plugin at the pinned spec and runs ``brainsets check``: the entry
   point must resolve to your namespace, and the brainsets you list must be exactly
   the ones your plugin ships.
4. A maintainer does a light content review: an OSI-approved licence and a README,
   publicly available data, cited sources and accurate descriptions. Your code is not
   reviewed: listed plugins are community plugins, maintained by their authors.

To update your listing, open a PR that bumps the pinned version. The registry check
also runs weekly; a plugin that keeps failing it for 4 weeks is delisted.

The registry ships inside torch_brain, so the CLI works offline and shows new listings
from the next torch_brain release; the docs page shows them as soon as they are
merged.
