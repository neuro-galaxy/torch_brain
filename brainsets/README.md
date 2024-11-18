# brainsets

[Documentation](https://brainsets.readthedocs.io/en/latest/) | [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)

[![PyPI version](https://badge.fury.io/py/brainsets.svg)](https://badge.fury.io/py/brainsets)
[![Documentation Status](https://readthedocs.org/projects/brainsets/badge/?version=latest)](https://brainsets.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/brainsets/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/brainsets/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/brainsets/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/brainsets/actions/workflows/linting.yml)


**brainsets** is a Python package for processing neural data into a standardized format.

## Installation
brainsets is available for Python 3.8 to Python 3.11

To install the package, run the following command:
```bash
pip install brainsets
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


## Cite

Please cite [our paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html) if you use this code in your own work:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```