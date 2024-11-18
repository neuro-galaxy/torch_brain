from setuptools import find_packages, setup

setup(
    name="brainsets",
    version="0.1.0",
    author="Mehdi Azabou",
    author_email="mehdiazabou@gmail.com",
    description="A package for processing neural datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "temporaldata",
        "scipy~=1.10.1",
        "pynwb~=2.2.0",
        "setuptools~=60.2.0",
        "numpy~=1.23.5",
        "pandas~=1.5.3",
        "jsonschema~=4.21.1",
        "scikit-image~=0.19.3",
        "tqdm~=4.64.1",
        "rich==13.3.2",
        "msgpack~=1.0.5",
        "snakemake~=7.32.3",
        "pydantic~=2.0",
        "pulp==2.7.0",
        "click~=8.1.3",
        "dandi==0.61.2",
    ],
    extras_require={
        "dev": [
            "pytest~=7.2.1",
            "black==24.2.0",
            "pre-commit>=3.5.0",
            "flake8",
        ],
        "eeg": [
            "mne",
        ],
        "zenodo": [
            "zenodo-get~=1.5.1",
        ],
        "allen": [
            "allensdk==2.16.2",
        ],
        "all": [
            "pytest~=7.2.1",
            "black==24.2.0",
            "pre-commit>=3.5.0",
            "flake8",
            "zenodo-get~=1.5.1",
            "allensdk==2.16.2",
            "mne",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "brainsets=brainsets.cli:cli",
        ],
    },
)
