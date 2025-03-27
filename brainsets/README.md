<p align="left">
    <img height="250" src="https://brainsets.readthedocs.io/en/latest/_static/brainsets_logo.png" />
</p>

[Documentation](https://brainsets.readthedocs.io/en/latest/) | [Join our Discord community](https://discord.gg/kQNKA6B8ZC)

[![PyPI version](https://badge.fury.io/py/brainsets.svg)](https://badge.fury.io/py/brainsets)
[![Documentation Status](https://readthedocs.org/projects/brainsets/badge/?version=latest)](https://brainsets.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/brainsets/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/brainsets/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/brainsets/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/brainsets/actions/workflows/linting.yml)
[![Discord](https://img.shields.io/discord/1338561153089146962?label=Discord&logo=discord)](https://discord.gg/kQNKA6B8ZC)


**brainsets** is a Python package for processing neural data into a standardized format.

## Installation
brainsets is available for Python 3.8 to Python 3.11

To install the package, run the following command:
```bash
pip install brainsets
```

## List of available brainsets

| brainset_id | Brainset Card | Raw Data Size | Processed Data Size |
|-------------|---------------|---------------|--------------------|
| churchland_shenoy_neural_2012 | [Link](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#churchland-shenoy-neural-2012) | 46 GB | 25 GB |
| flint_slutzky_accurate_2012 | [Link](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#flint-slutzky-accurate-2012) | 3.2 GB | 151 MB |
| odoherty_sabes_nonhuman_2017 | [Link](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#odoherty-sabes-nonhuman-2017) | 22 GB | 26 GB |
| pei_pandarinath_nlb_2021  | [Link](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#pei-pandarinath-nlb-2021) | 688 KB | 22 MB |
| perich_miller_population_2018 | [Link](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#perich-miller-population-2018) | 13 GB | 2.9 GB |


## Acknowledgements

This work is only made possible thanks to the public release of these valuable datasets by the original researchers. If you use any of the datasets processed by brainsets in your research, please make sure to cite the appropriate original papers and follow any usage guidelines specified by the dataset creators. Proper attribution not only gives credit to the researchers who collected and shared the data but also helps promote open science practices in the neuroscience community. You can find the original papers and usage guidelines for each dataset in the [brainsets documentation](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html).


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