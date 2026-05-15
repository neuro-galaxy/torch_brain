import math
import numpy as np
from typing import Callable
from torch.optim import AdamW
from torch.utils.data import DataLoader
from brainsets.datasets import PeiPandarinathNLB2021
from temporaldata import Data, Interval
import torchmetrics
from tqdm import tqdm
from torch_brain.data.sampler import TrialSampler
from torch_brain.dataset.dataset import DatasetIndex
from torch_brain.utils.binning import bin_spikes
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CausalLinearDecoder(nn.Module):
    def __init__(
        self,
        num_units: int,
        dim_out: int,
        ctx_window: float,
        bin_size: float,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.bin_size = bin_size
        self.num_units = num_units
        self.ctx_window = ctx_window
        self.num_bins = math.ceil(ctx_window / bin_size)

        dim_in = num_units * self.num_bins
        self.net = nn.Sequential(
            nn.Linear(dim_in, int(math.sqrt(dim_in * dim_out))),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(dim_in * dim_out)), dim_out),
        )

    def forward(self, x: Tensor):
        return self.net(x)

    def tokenize(self, data: Data):
        x = bin_spikes(
            spikes=data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
        ).flatten()

        return x


class CausalDataset(PeiPandarinathNLB2021):
    HAND_VEL_FS = 1000.0  # in Hz
    HAND_VEL_DIM = 2

    def __init__(
        self,
        root: str,
        ctx_window: float,
        transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            root=root,
            dirname="pei_pandarinath_nlb_2021",
            recording_ids=["jenkins_maze_train"],
            transform=transform,
            **kwargs,
        )
        self.ctx_window = ctx_window

    def get_sampling_intervals(self, split, *args, **kwargs):
        ans = super().get_sampling_intervals(split)
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            starts = rec.hand.timestamps - 0.25 / self.HAND_VEL_FS
            ends = rec.hand.timestamps + 0.25 / self.HAND_VEL_FS
            intrvl = Interval(starts, ends)
            ans[rid] = intrvl & ans[rid]
        return ans

    def __getitem__(self, index: DatasetIndex):
        data = self.get_recording(index.recording_id, index._namespace)
        data = data.slice(index.end - self.ctx_window, index.end)
        assert self.transform is not None
        X = self.transform(data)
        y = torch.tensor(data.hand.vel[-1], dtype=torch.float32)
        y = y / 200.0  # normalization
        return X, y


ctx_window = 200e-3  # in seconds
bin_size = 20e-3  # in seconds
batch_size = 512
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ds = CausalDataset(
    root="../brainsets/data/processed/",
    ctx_window=ctx_window,
)

model = CausalLinearDecoder(
    num_units=len(ds.get_unit_ids()),
    dim_out=ds.HAND_VEL_DIM,
    ctx_window=ctx_window,
    bin_size=bin_size,
    dropout_p=0.2,
).to(device)
print(model)
ds.transform = model.tokenize

opt = AdamW(model.parameters())


train_loader = DataLoader(
    dataset=ds,
    sampler=TrialSampler(
        sampling_intervals=ds.get_sampling_intervals("train"),
        shuffle=True,
    ),
    batch_size=batch_size,
    num_workers=16,
)

val_loader = DataLoader(
    dataset=ds,
    sampler=TrialSampler(sampling_intervals=ds.get_sampling_intervals("test")),
    batch_size=batch_size,
    num_workers=16,
)

metric = torchmetrics.R2Score().to(device)

for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    for X, y in (train_pbar := tqdm(train_loader, leave=False)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = F.mse_loss(pred, y)

        train_pbar.set_description(f"Train | Loss {loss.item():.3f}")
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Validation
    metric.reset()
    model.eval()
    for X, y in tqdm(val_loader, leave=False, desc="Val"):
        X, y = X.to(device), y.to(device)
        with torch.inference_mode():
            pred = model(X)
        metric.update(pred, y)

    r2 = metric.compute()
    print(r2.item())
