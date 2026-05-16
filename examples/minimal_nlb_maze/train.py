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
    sample_length = 0.5
    bin_size = 0.02
    out_sampling_rate = 1000.0
    out_dim = 2

    def __init__(self, root, split: Literal["train", "val"]):
        super().__init__(root=root, recording_ids=["jenkins_maze_train"])
        self.split = split

    def get_sampling_intervals(self, *_args, **_kwargs):
        rid = self.recording_ids[0]
        recording = self.get_recording(rid)

        move_onset_times = recording.trials.move_onset_time
        trial_split_indicator = recording.trials.split_indicator.astype(str)
        trials = Interval(move_onset_times - 0.05, move_onset_times + 0.45)

        train_trials = trials.select_by_mask(trial_split_indicator == "train")
        val_trials = trials.select_by_mask(trial_split_indicator == "val")

        if self.split == "train":
            return {rid: train_trials}
        elif self.split == "val":
            return {rid: val_trials}

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).flatten().float()

        Y = data.hand.vel / 100.0
        Y = torch.from_numpy(Y).flatten().float()
        return X, Y


def make_linear(ds: SimpleNLBMazeDataset):
    num_units = len(ds.get_unit_ids())
    num_bins = int(ds.sample_length / ds.bin_size)
    input_size = num_units * num_bins
    output_size = int(ds.sample_length * ds.out_sampling_rate * ds.out_dim)

    model = nn.Linear(input_size, output_size)
    return model


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

model = make_linear(train_ds).to(device)
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
        pred = torch.cat(preds).flatten().cpu()
        target = torch.cat(targets).flatten().cpu()
        r2 = r2_score(target, pred)
        epoch_pbar.set_description(f"R2: {r2:.3f}")
