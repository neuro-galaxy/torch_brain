<p align="left">
    <img height="250" src="https://torch-brain.readthedocs.io/en/latest/_static/torch_brain_logo.png" />
</p>

[Documentation](https://torch-brain.readthedocs.io/en/latest/) | [Join our Discord community](https://discord.gg/kQNKA6B8ZC)

[![PyPI version](https://badge.fury.io/py/torch_brain.svg)](https://badge.fury.io/py/torch_brain)
[![Documentation Status](https://readthedocs.org/projects/torch-brain/badge/?version=latest)](https://torch-brain.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml)
[![Linting](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml/badge.svg)](https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml)
[![Discord](https://img.shields.io/discord/1338561153089146962?label=Discord&logo=discord)](https://discord.gg/kQNKA6B8ZC)

**torch_brain** is a Python library for various deep learning models designed for neuroscience.

> [!NOTE]
> We have merged `temporaldata` and `brainsets` into `torch_brain`.
> If you are migrating from v0.1.x, please see [this migration guide](howto/MIGRATE_TO_v0.2.md).

### Features
+ Multi-recording training
+ Optimized data loading with with on-demand data access -- only loads data when needed
+ Advanced samplers that enable arbitrary slicing of data on the fly
+ Advanced data collation strategies including chaining and padding
+ Support for arbitrary neural and behavioral modalities
+ Collection of useful nn.Modules like stitchers, multi-output readouts, infinite vocab embeddings, etc.
+ Collection of neural and behavioral transforms and augmentation strategies
+ Implementations of various deep learning models for neuroscience

### List of implemented models

+ [POYO: A Unified, Scalable Framework for Neural Population Decoding (Azabou et al. 2023)](examples/poyo)
+ More coming soon...

## Installation
**torch_brain** is available for Python >= 3.10. To install a stable release, do:
```bash
pip install torch torch_brain
```

If you only wish to work with `torch_brain.data`, and pipeline related modules,
you can skip installing `torch`.

To install the development version:
```bash
pip install git+https://github.com/neuro-galaxy/torch_brain
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

## Building documentation for development

Install requirements:
```bash
pip install -e ".[dev,docs]"
```

Build:
```bash
cd docs
make clean html
```

The documentation is then present in `docs/build/html`

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
