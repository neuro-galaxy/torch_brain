# Minimal Training Example on NLB Maze

> [!TIP]
> Follow along this tutorial in a notebook on Google Collab.
> The notebook includes visualizations of the data and model predictions.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1vbxmqccgHz-6det9Bxk3Ld6xU8RZdk?usp=sharing) |

This example walks through a minimal training script for decoding hand
velocity from motor cortex spiking activity in the `jenkins_maze_train`
recording, originally from the [Neural Latents Benchmark (NLB)](https://neurallatents.github.io/)
MC_Maze dataset.

The training script is short (~150 lines) and annotated with comments explaining
fundamental concepts. **We encourage new users to read through the `train.py`
file end-to-end!**

This example shows how to:

1. Build a custom simple `Dataset` on top of an existing `brainset.dataset`.
2. Create sampling intervals around behavioral events.
3. Bin spike trains and align targets.
4. Set up a minimal train loop.


## Running the example

1. Install some packages:
```bash
pip install scikit-learn
pip install git+https://github.com/neuro-galaxy/brainsets
# ^ Needed since the latest brainsets has not been released yet.
# The latest version has some fixes for the NLB dataset which are
# needed for this example to work.
```

2. Preprocess the dataset (takes ~1 minute and ~50MB of disk):
```bash
brainsets prepare pei_pandarinath_nlb_2021 --raw-dir data/raw --processed-dir data/processed
```

3. Train!
```bash
python train.py --model Linear
python train.py --model GRU
python train.py --model TCN

# To see other configuration options
python train.py --help
```

