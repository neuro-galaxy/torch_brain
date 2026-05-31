from typing import Literal
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from temporaldata import Interval
from torch_brain.utils import bin_spikes
from torch_brain.samplers import TrialSampler
from torch_brain.datasets import DatasetIndex
from brainsets.datasets import PeiPandarinathNLB2021

import models

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default="GRU", choices=["Linear", "GRU", "TCN"])
parser.add_argument("--data-root", default="data/processed", help="Root data directory")
parser.add_argument("--bin-size", default=0.05, type=float, help="Bin size")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("--batch-size", default=8, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
args = parser.parse_args()


# brainsets ships loaders for a collection of public neuro datasets.
# PeiPandarinathNLB2021 handles file I/O for the NLB Maze dataset.
#
# We subclass this dataset to define two things on top of its file I/O:
#   1. get_sampling_intervals: *which* time windows count as samples
#   2. __getitem__: *how* one window is turned into model-compatible tensors
class SimpleNLBMazeDataset(PeiPandarinathNLB2021):
    sample_length = 0.7
    out_dim = 2
    out_sampling_rate = 1000.0

    def __init__(self, root: str, split: Literal["train", "val"], bin_size: float):
        # recording_ids picks which session(s) inside the dataset to load.
        # We just want to load the maze_train session
        super().__init__(root=root, recording_ids=["jenkins_maze_train"])

        # This recording only specificies train and validation set
        # and the test set is kept hidden for online evaluation
        assert split in ("train", "val")

        # store some attributes that are useful later
        self.split = split
        self.bin_size = bin_size
        self.out_samples = round(self.sample_length * self.out_sampling_rate)
        self.num_bins = round(self.sample_length / self.bin_size)
        # get_unit_ids returns the list of neurons recorded in this session.
        self.num_units = len(self.get_unit_ids())

    # Contract between Datasets and Samplers:
    # get_sampling_intervals() returns {recording_id: Interval} listing
    # the windows the sampler may draw from.
    # Sampler will emit one DatasetIndex per sample.
    def get_sampling_intervals(self, *_args, **_kwargs):
        rid = self.recording_ids[0]  # since we only have 1 recording
        recording = self.get_recording(rid)

        # Taking trials to be relative to the movement onset time
        # from 250ms before onset to 450ms after onset
        # (as stated in the NLB paper Appendix A.5.1).
        move_onset_times = recording.trials.move_onset_time
        trials = Interval(move_onset_times - 0.25, move_onset_times + 0.45)

        # The NLB dataset also provided us a default assignment of
        # training and validation trials. select_by_mask is the standard way
        # to filter an Interval down to a subset based on a boolean mask.
        trial_split_indicator = recording.trials.split_indicator.astype(str)
        train_trials = trials.select_by_mask(trial_split_indicator == "train")
        val_trials = trials.select_by_mask(trial_split_indicator == "val")

        if self.split == "train":
            return {rid: train_trials}
        elif self.split == "val":
            return {rid: val_trials}

    # `index` is a DatasetIndex(recording_id, start, end) produced by the sampler.
    def __getitem__(self, index: DatasetIndex):
        # super().__getitem__ returns a sliced view of the recording, with all
        # modalities (.spikes, .units, .hand.vel, ...) already cropped (lazily).
        data = super().__getitem__(index)

        # In this example, we have designed all models in `model.py` to take in
        # a Tensor of shape (Number of neurons, Bins), and return a Tensor of shape
        # (Number of output timestep, Output dimension).

        # Spikes are an irregular event stream — bin them into a regular grid.
        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()  # shape: (num_bins, num_units)

        # Hand velocity is already a regularly-sampled signal, so we just rescale.
        Y = data.hand.vel / 200.0  # approximate z-score normalization
        Y = torch.from_numpy(Y).float()  # shape: (out_samples, out_dim)
        return X, Y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💡 The torch_brain pattern: Sampler decides *what trials* to load (by emitting
# DatasetIndex objects), Dataset decides *how to load one*, DataLoader batches
# as usual. Note the sampler is passed explicitly — it is not the default
# random/sequential sampler PyTorch picks for an indexable dataset.
train_ds = SimpleNLBMazeDataset(args.data_root, split="train", bin_size=args.bin_size)
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(),
    shuffle=True,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
print(f"Number of units: {train_ds.num_units}")
print(f"Number of training samples: {len(train_sampler)}")

# Validation Dataset, Sampler, and DataLoader
val_ds = SimpleNLBMazeDataset(args.data_root, split="val", bin_size=args.bin_size)
val_sampler = TrialSampler(sampling_intervals=val_ds.get_sampling_intervals())
val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)
print(f"Number of validation samples: {len(val_sampler)}")

# Setup Model and Optimizer
# Feel free to look around in the `models.py` file!
model_class = models.__dict__[args.model]
model = model_class(
    in_units=train_ds.num_units,
    in_bins=train_ds.num_bins,
    out_dim=train_ds.out_dim,
    out_samples=train_ds.out_samples,
)
model = model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {model}")
print(f"Number of parameters {num_parameters:,}")

optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

# Train Loop
for epoch in (epoch_pbar := tqdm(range(args.epochs))):
    # Train epoch
    model.train()
    for X, Y in (step_pbar := tqdm(train_loader, leave=False)):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        step_pbar.set_description(f"Loss: {loss.item():.3f}")

    # Validation epoch
    with torch.no_grad():
        model.eval()
        preds, targets = [], []
        for X, Y in tqdm(val_loader, leave=False, desc="Val"):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            preds.append(pred)
            targets.append(Y)
        pred = torch.cat(preds).flatten(0, 1).cpu()
        target = torch.cat(targets).flatten(0, 1).cpu()
        r2 = r2_score(target, pred)
        epoch_pbar.set_description(f"R2: {r2:.3f}")
