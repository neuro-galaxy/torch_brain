# ⏳ temporaldata

**temporaldata** is a Python package for easily working with temporal data. It provides
advanced data structures and methods to work with multi-modal, multi-resolution 
time series data.

## Installation
temporaldata is available for Python 3.8 to Python 3.11

temporaldata has minimal dependencies, it only requires `numpy`, `pandas`, and `h5py`.

To install the package, run the following command:
```bash
pip install -e .
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
