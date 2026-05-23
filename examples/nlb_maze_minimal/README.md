# Minimal Training Example on NLB Maze

> [!TIP]
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuro-galaxy/torch_brain/tree/main/docs/source/notebooks/nlb_maze_minimal_example/nlb_maze_minimal_example.ipynb)
> <br/>
> Follow along this tutorial in a notebook on Google Colab.
> The notebook includes visualizations of the data and model predictions.

This example walks through a minimal training script for decoding hand
velocity from motor cortex spiking activity in the `jenkins_maze_train`
recording, originally from the [Neural Latents Benchmark (NLB)](https://neurallatents.github.io/)
MC_Maze dataset.

The training script is short (~150 lines) and annotated with comments explaining
fundamental concepts. **We encourage new users to read through the `train.py`
file end-to-end!**

This example shows how to:

1. Build a custom `Dataset` on top of a `brainsets` recording.
2. Sample fixed-length trials around behavior events using `TrialSampler`
3. Set up a minimal training loop.


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

