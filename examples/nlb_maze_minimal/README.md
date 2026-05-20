# Minimal Training Example on NLB Maze

This example walks through a minimal training script for decoding hand
velocity from motor cortex spiking activity in the `jenkins_maze_train`
recording, originally from the [Neural Latents Benchmark (NLB)](https://neurallatents.github.io/)
MC_Maze dataset.

The training script, `train.py`, is short (~150 lines) and annotated with
comments explaining important concepts. We encourage new users to read through
the file end-to-end!

This example shows how to:

1. Build a custom simple `Dataset` on top of an existing `brainset.dataset`.
2. Create sampling intervals around behavioral events.
3. Transform and shape the data samples.
4. Set up a minimal train loop.


## Running the example

1. Install some packages:
```bash
pip install sklearn
pip install git+https://github.com/neuro-galaxy/brainsets@94fb240
# ^ Needed since the latest brainsets has not been released yet.
# The latest version has some fixes for the NLB dataset which are
# needed for this example to work.
```

2. Preprocess the dataset:
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

