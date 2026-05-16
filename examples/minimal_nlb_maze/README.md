# Minimal Training Example on NLB Maze

## Setup

Install some packages:
```bash
pip install sklearn
pip install git+https://github.com/neuro-galaxy/brainsets@94fb240
# ^ Needed since the latest brainsets has not been released yet.
# The latest version has some fixes for the NLB dataset which are
# needed for this example to work.
```

Preprocess dataset:
```bash
brainsets prepare pei_pandarinath_nlb_2021 --raw-dir data/raw --processed-dir data/processed
```

Run:
```bash
python train.py
```
