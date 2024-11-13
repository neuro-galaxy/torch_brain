# ⏳ temporaldata

[Documentation](https://temporaldata.readthedocs.io/en/latest/) | [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)

[![PyPI version](https://badge.fury.io/py/temporaldata.svg)](https://badge.fury.io/py/temporaldata)
[![Documentation Status](https://readthedocs.org/projects/temporaldata/badge/?version=latest)](https://temporaldata.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/temporaldata/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/temporaldata/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/temporaldata/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/temporaldata/actions/workflows/linting.yml)


**temporaldata** is a Python package for easily working with temporal data. It provides
advanced data structures and methods to work with multi-modal, multi-resolution 
time series data.

## Installation
temporaldata is available for Python 3.8 to Python 3.11

temporaldata has minimal dependencies, it only requires `numpy`, `pandas`, and `h5py`.

To install the package, run the following command:
```bash
pip install temporaldata
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
or test individual files via, e.g., `pytest test/test_data.py`

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