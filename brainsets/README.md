# 🧠 brainsets

**brainsets** is a Python package for processing neural data into a standardized format.

## Installation
brainsets is available for Python 3.9 to Python 3.11

To install the package, run the following command:
```bash
pip install -e ".[dandi]"
```

## Using the brainsets CLI

### Configuring data directories
First, configure the directories where brainsets will store raw and processed data:
```bash
brainsets config
```

You will be prompted to enter the paths to the raw and processed data directories.
```bash
$> brainsets config
Enter raw data directory: ./data/raw
Enter processed data directory: ./data/processed
```

You can update the configuration at any time by running the `config` command again.

### Listing available datasets
You can list the available datasets by running the `list` command:
```bash
brainsets list
```

### Preparing data
You can prepare a dataset by running the `prepare` command:
```bash
brainsets prepare <brainset>
```

Data preparation involves downloading the raw data from the source then processing it, 
following a set of rules defined in `pipelines/<brainset>/`.

For example, to prepare the Perich & Miller (2018) dataset, you can run:
```bash
brainsets prepare perich_miller_population_2018 --cores 8
```

## Contributing
If you are planning to contribute to the package, you can install the package in
development mode by running the following command:
```bash
pip install -e ".[dev]"
```

Install pre-commit hooks:
```bash
pre-commit install
```

Unit tests are located under test/. Run the entire test suite with
```bash
pytest
```
or test individual files via, e.g., `pytest test/test_enum_unique.py`
