from setuptools import find_packages, setup

setup(
    name="temporaldata",
    version="0.1.1",
    author="Mehdi Azabou",
    author_email="mehdiazabou@gmail.com",
    description="A utility package for complex temporal data manipulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "setuptools>=60.0.0",
        "numpy>=1.14.0",
        "pandas>=1.0.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest~=7.2.1",
            "black==24.2.0",
            "pre-commit>=3.5.0",
            "flake8",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
