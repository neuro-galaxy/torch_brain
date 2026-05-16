from typing import Literal
import math
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from temporaldata import Interval
from torch_brain.utils import bin_spikes
from torch_brain.data.sampler import TrialSampler
from torch_brain.dataset import Dataset, DatasetIndex
from brainsets.datasets import PeiPandarinathNLB2021

parser = ArgumentParser()
parser.add_argument("--data-root", default="data/processed", type=str)
parser.add_argument("--bin-size", default=0.05, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
args = parser.parse_args()


class SimpleNLBMazeDataset(PeiPandarinathNLB2021):
    sample_length = 0.7
    out_dim = 2
    out_sampling_rate = 1000.0

    def __init__(self, root, split: Literal["train", "val"], bin_size: float):
        super().__init__(root=root, recording_ids=["jenkins_maze_train"])
        self.split = split
        self.bin_size = bin_size

        self.out_samples = round(self.sample_length * self.out_sampling_rate)
        self.num_bins = round(self.sample_length / self.bin_size)
        self.num_units = len(self.get_unit_ids())

    def get_sampling_intervals(self, *_args, **_kwargs):
        rid = self.recording_ids[0]  # since we only have 1 recording
        recording = self.get_recording(rid)

        # Taking trials to be relative to the movement onset time
        # from 250ms before onset to 450ms after onset
        # (as stated in the NLB paper Appendix A.5.1)
        move_onset_times = recording.trials.move_onset_time
        trials = Interval(move_onset_times - 0.25, move_onset_times + 0.45)

        # The NLB dataset also provided us a default assignment of
        # training and validation trials
        trial_split_indicator = recording.trials.split_indicator.astype(str)
        train_trials = trials.select_by_mask(trial_split_indicator == "train")
        val_trials = trials.select_by_mask(trial_split_indicator == "val")

        if self.split == "train":
            return {rid: train_trials}
        elif self.split == "val":
            return {rid: val_trials}

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()  # shape: (num_bins, num_units)

        Y = data.hand.vel / 200.0  # appoximate z-score normalization
        Y = torch.from_numpy(Y).float()  # shape: (out_samples, out_dim)
        return X, Y


class Linear(nn.Module):
    def __init__(self, in_units, in_bins, out_dim, out_samples):
        super().__init__()
        self.out_dim = out_dim
        self.out_samples = out_samples

        input_size = in_units * in_bins
        output_size = out_dim * out_samples
        self.net = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        y = self.net(x.flatten(start_dim=1))
        y = y.view(batch_size, self.out_samples, self.out_dim)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create train dataset
train_ds = SimpleNLBMazeDataset(args.data_root, split="train", bin_size=args.bin_size)
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(),
    shuffle=True,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)

val_ds = SimpleNLBMazeDataset(args.data_root, split="val", bin_size=args.bin_size)
val_sampler = TrialSampler(sampling_intervals=val_ds.get_sampling_intervals())
val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)

print(f"Number of units: {train_ds.num_units}")
print(f"Number of bins/per-sampler: {train_ds.num_bins}")
print(f"Number of training samples: {len(train_sampler)}")
print(f"Number of validation samples: {len(val_sampler)}")

num_units = len(train_ds.get_unit_ids())
model = Linear(
    in_units=num_units,
    in_bins=train_ds.num_bins,
    out_dim=train_ds.out_dim,
    out_samples=train_ds.out_samples,
)
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)


for epoch in (epoch_pbar := tqdm(range(args.epochs))):
    model.train()
    for X, Y in (step_pbar := tqdm(train_loader, leave=False)):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        step_pbar.set_description(f"Loss: {loss.item():.3f}")

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
