# 🔥 torch_brain

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

## Documentation
> [!WARNING]  
> The docs are hosted publically for convenience, please do not share the link.

You can find the documentation for this package [here](https://chic-dragon-bc9a04.netlify.app/).

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
