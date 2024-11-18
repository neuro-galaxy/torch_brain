Using the brainsets CLI
=======================

The brainsets CLI provides access to existing datasets, supported by the brainsets package.

Configuring data directories
----------------------------
First, configure the directories where brainsets will store raw and processed data::

    brainsets config

You will be prompted to enter the paths to the raw and processed data directories::

    $> brainsets config
    Enter raw data directory: ./data/raw
    Enter processed data directory: ./data/processed

You can update the configuration at any time by running the ``config`` command again.

Listing available datasets
-------------------------
You can list the available datasets by running the ``list`` command::

    brainsets list

Preparing data
-------------
You can prepare a dataset by running the ``prepare`` command::

    brainsets prepare <brainset>

Data preparation involves downloading the raw data from the source then processing it, 
following a set of rules defined in ``pipelines/<brainset>/``.
For example, to prepare the Perich & Miller (2018) dataset, you can run::

    brainsets prepare perich_miller_population_2018 --cores 8
