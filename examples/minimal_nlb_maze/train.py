from typing import Literal
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
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
args = parser.parse_args()


class SimpleNLBMazeDataset(PeiPandarinathNLB2021):

    def __init__(self, root, split: Literal["train", "val"]):
        super().__init__(root=root, recording_ids=["jenkins_maze_train"])
        self.split = split

        self.sample_length = 0.7
        self.bin_size = 0.02
        self.out_sampling_rate = 1000.0
        self.out_dim = 2
        self.out_samples = int(self.sample_length * self.out_sampling_rate)
        self.num_bins = int(self.sample_length / self.bin_size)

    def get_sampling_intervals(self, *_args, **_kwargs):
        rid = self.recording_ids[0]
        recording = self.get_recording(rid)

        move_onset_times = recording.trials.move_onset_time
        trial_split_indicator = recording.trials.split_indicator.astype(str)
        trials = Interval(move_onset_times - 0.25, move_onset_times + 0.45)

        train_trials = trials.select_by_mask(trial_split_indicator == "train")
        val_trials = trials.select_by_mask(trial_split_indicator == "val")

        if self.split == "train":
            return {rid: train_trials}
        elif self.split == "val":
            return {rid: val_trials}

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()

        Y = data.hand.vel / 100.0
        Y = torch.from_numpy(Y).float()
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

train_ds = SimpleNLBMazeDataset(args.data_root, split="train")
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(),
    shuffle=True,
)
train_loader = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=0)

val_ds = SimpleNLBMazeDataset(args.data_root, split="val")
val_sampler = TrialSampler(sampling_intervals=val_ds.get_sampling_intervals())
val_loader = DataLoader(val_ds, batch_size=8, sampler=val_sampler, num_workers=0)

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
