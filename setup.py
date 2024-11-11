from setuptools import find_packages, find_namespace_packages, setup

setup(
    name="torch_brain",
    version="0.1.0",
    author="Mehdi Azabou",
    author_email="mehdiazabou@gmail.com",
    description="A deep learning framework for neural data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages() + find_namespace_packages(include=["hydra_plugins.*"]),
    include_package_data=True,
    install_requires=[
        "temporaldata==0.1.1",
        "torch==2.2.0",
        "einops~=0.6.0",
        "rich==13.3.2",
        "torch-optimizer==0.3.0",
        "tensorboard~=2.13",
        "hydra-core~=1.3.2",
        "lightning==2.3.3",
        "wandb~=0.15",
        "torchtyping~=0.1",
    ],
    extras_require={
        "dev": [
            "pytest~=7.2.1",
            "black==24.2.0",
            "pre-commit>=3.5.0",
            "flake8",
        ],
        "xformers": [
            "xformers==0.0.24",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
