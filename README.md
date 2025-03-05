# torch_brain


[Documentation](https://torch-brain.readthedocs.io/en/latest/) | [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)

<!-- [![PyPI version](https://badge.fury.io/py/torch_brain.svg)](https://badge.fury.io/py/torch_brain) -->
[![Documentation Status](https://readthedocs.org/projects/torch-brain/badge/?version=latest)](https://torch-brain.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml)


**torch_brain** is a Python library for various deep learning models designed for neuroscience.

## Installation
torch_brain is available for Python 3.9 to Python 3.11

To install the package, run the following command:
```bash
pip install -e .
```

To avoid conflicts between different packages, you can specify the packages that will be
used as follows:
```bash
pip install -e ".[xformers]"
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
or test individual files via, e.g., `pytest test/test_binning.py`


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