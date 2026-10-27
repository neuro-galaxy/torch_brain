Using the brainsets CLI
=======================

The brainsets CLI provides access to existing datasets, supported by the brainsets package.

Enabling tab completion
-----------------------
First, install shell tab-completion to autocomplete brainsets commands, options and dataset names.

.. code:: shell

    brainsets --install-completion

Configuring data directories
----------------------------
Next, configure the directories where brainsets will store raw and processed data.

.. code:: shell

    brainsets config set

You will be prompted to enter the paths to the raw and processed data directories.

.. code-block:: console

    $ brainsets config set
    Enter raw data directory: ./data/raw
    Enter processed data directory: ./data/processed

You can update the configuration at any time by running the ``config set`` command again.

To view the current configuration:

.. code:: shell

    brainsets config show

Listing available datasets
--------------------------
You can list the available datasets by running the ``list`` command.

.. code:: shell

    brainsets list

Preparing data
--------------
You can prepare a dataset by running the ``prepare`` command.

.. code:: shell

    brainsets prepare <brainset>

Data preparation involves downloading the raw data from the source then processing it,
following a set of rules defined in ``pipelines/<brainset>/``.
For example, to prepare the Perich & Miller (2018) dataset, you can run:

.. code:: shell

    brainsets prepare perich_miller_population_2018 --cores 8
