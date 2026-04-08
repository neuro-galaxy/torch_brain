CLI Reference
=============


brainsets
---------

Command line interface for downloading and processing neural data with brainsets.

**Usage**

.. code-block::

    brainsets [OPTIONS] [COMMAND] [ARGS]

**Commands**

.. rst-class:: cli-option

`brainsets config`_

    Set raw and processed data directories.

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


brainsets config
----------------

Set raw and processed data directories.

**Usage**

.. code-block::

    brainsets config [OPTIONS]

**Options**

.. rst-class:: cli-option

``--raw-dir`` *path*

    Path for storing raw data.

.. rst-class:: cli-option

``--processed-dir`` *path*

    Path for storing processed brainsets.

If no options are provided, you will be prompted to enter the paths interactively.



brainsets list
--------------

List all pipelines provided by the installed package.

**Usage**

.. code-block::

    brainsets list



brainsets prepare
-----------------

Download and process a single brainset.

.. code-block::

    brainsets prepare [OPTIONS] BRAINSET

**Arguments**

.. rst-class:: cli-option

``BRAINSET``

    Name of the brainset to prepare (see ``brainsets list``). When used with ``--local``,
    this should be the path to a local pipeline directory.

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
