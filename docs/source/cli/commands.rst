CLI Reference
=============


brainsets
---------

Command line interface for downloading and processing neural data with brainsets.

**Usage**

.. code:: shell

    brainsets [OPTIONS] [COMMAND] [ARGS]

**Commands**

.. rst-class:: cli-option

`brainsets check`_

    Check that an installed plugin follows the plugin rules.

.. rst-class:: cli-option

`brainsets config`_

    Manage brainsets configuration.

.. rst-class:: cli-option

`brainsets info`_

    Show details about a plugin or a brainset.

.. rst-class:: cli-option

`brainsets list`_

    List available brainsets.

.. rst-class:: cli-option

`brainsets prepare`_

    Download and process a single brainset.

**Options**

.. rst-class:: cli-option

``--install-completion``

    Install shell tab-completion for the ``brainsets`` command.


brainsets check
---------------

Check that an installed :ref:`brainsets plugin <brainsets_plugins>` follows the plugin
rules: its entry point resolves, every ``<brainset_id>/pipeline.py`` defines a valid
``Pipeline`` whose ``brainset_id`` matches its folder, the inline metadata is valid,
and, if the plugin is in the registry, the registered brainsets match the shipped ones.
Each ``pipeline.py`` is imported (or read, if its dependencies are not installed), but
no data is downloaded.

**Usage**

.. code:: shell

    brainsets check [OPTIONS] [NAMESPACE]

**Arguments**

.. rst-class:: cli-option

``NAMESPACE``

    The plugin's namespace, or ``torch_brain`` for the built-in pipelines.

**Options**

.. rst-class:: cli-option

``--registry`` *[path]*

    Validate the plugin registry (by default the one bundled with torch_brain) instead
    of a plugin.

.. rst-class:: cli-option

``--strict``

    Treat warnings as errors.


brainsets config
----------------

Manage brainsets configuration.

**Usage**

.. code:: shell

    brainsets config [OPTIONS] COMMAND [ARGS]

**Commands**

.. rst-class:: cli-option

``brainsets config set [OPTIONS]``

    Set raw and processed data directories.

.. rst-class:: cli-option

``brainsets config show``

    Show previously set raw and processed data directories.

**Options**

.. rst-class:: cli-option

``--raw-dir`` *path*

    Path for storing raw data.

.. rst-class:: cli-option

``--processed-dir`` *path*

    Path for storing processed brainsets.

If no options are provided, you will be prompted to enter the paths interactively.



brainsets info
--------------

Show details about a plugin or a brainset: description, source, installed version,
pipeline location and dependencies. Plugins from the
:ref:`registry <community_brainsets>` are shown even when they are not installed,
with the command to install them.

**Usage**

.. code:: shell

    brainsets info NAME

**Arguments**

.. rst-class:: cli-option

``NAME``

    A plugin namespace (e.g. ``openneuro``), a plugin brainset
    (``<namespace>:<brainset_id>``) or a built-in brainset.


brainsets list
--------------

List the built-in brainsets and the brainsets of installed plugins, grouped by source.
Plugin brainsets are listed with their namespace (``<namespace>:<brainset_id>``).

**Usage**

.. code:: shell

    brainsets list [--all]

**Options**

.. rst-class:: cli-option

``--all``

    Also list community plugins from the registry that are not installed.



brainsets prepare
-----------------

Download and process a single brainset.

.. code:: shell

    brainsets prepare [OPTIONS] BRAINSET

**Arguments**

.. rst-class:: cli-option

``BRAINSET``

    Name of the brainset to prepare (see ``brainsets list``). Built-in brainsets use
    their bare name (``torch_brain:<brainset_id>`` also works), plugin brainsets use
    ``<namespace>:<brainset_id>``. When used with ``--local``, this should be the path
    to a local pipeline directory.

    ``prepare`` refuses to write into a ``<processed_dir>/<brainset_id>`` folder made
    by a different package, and marks the folders it writes with a
    ``.brainsets_source`` file.

**Options**

.. rst-class:: cli-option

``-c``, ``--cores`` *number*

    Number of parallel jobs. Default: 4.

.. rst-class:: cli-option

``--raw-dir`` *path*

    Path for storing raw data. Overrides config.

.. rst-class:: cli-option

``--processed-dir`` *path*

    Path for storing processed brainset. Overrides config.

.. rst-class:: cli-option

``--list``

    Print the manifest for the brainset and exit.

.. rst-class:: cli-option

``-s``, ``--single`` *manifest index id*

    Prepare a single item from the manifest if provided.

.. rst-class:: cli-option

``--local``

    Prepare a brainset from a local pipeline directory instead of the built-in registry.

.. rst-class:: cli-option

``--download-only``

    Only download raw data, skip processing.

.. rst-class:: cli-option

``--use-active-env``

    Developer flag. Do not create an isolated environment for the pipeline.

.. rst-class:: cli-option

``-v``, ``--verbose``

    Print debugging information.

Any additional arguments are passed through to the pipeline.
