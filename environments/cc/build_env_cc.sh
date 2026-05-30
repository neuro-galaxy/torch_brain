#!/bin/bash
# This should be run on a login node.
# Very important: need to use module load and conda install with the SAME python version.
set -e
ENV_NAME=testenv
PY_VERSION=3.9.6
module load python/$PY_VERSION

# Check if miniconda is already installed
if [ ! -d "$HOME/miniconda3" ] || [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
else
    echo "Miniconda already installed."
    ~/miniconda3/bin/conda init bash
fi

~/miniconda3/bin/conda init bash
conda create -y -n $ENV_NAME python=$PY_VERSION
source ~/.bashrc
conda activate $ENV_NAME

module load postgresql  # For psycopg2 and ultimately allensdk
module load cuda/11.7

# Very hard to install with pip, has a hidden Rust dependency, needed by lightning
conda install -y -c conda-forge pydantic=1.10
conda install -y -c anaconda lz4
# Kill the existing pytorch and replace with the right one.
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Now install regular requirements
pip install appdirs
pip install -r requirements_cc.txt