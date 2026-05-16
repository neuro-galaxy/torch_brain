# Minimal Training Example on NLB Maze

This example walks through a minimal training script for decoding hand
velocity from motor cortex spiking activity in the `jenkins_maze_train`
recording, originally from the [Neural Latents Benchmark (NLB)](https://neurallatents.github.io/)
MC_Maze dataset.

This example is meant for new users of `torch_brain` and `brainsets`,
and shows how to:

1. Build a custom simple `Dataset` on top of an existing `brainset.dataset`.
2. Create sampling intervals around behavioral events.
3. Transform and shape the data samples.
4. Set up a minimal train loop.

We demonstrate this by implementing three simple decoders (Linear, GRU, TCN).

## Running the example

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
python train.py --model Linear
python train.py --model GRU
python train.py --model TCN

# To see other configuration options
python train.py --help
```

## Explanation

### Step 1: A custom dataset
We start out this training script by creating a custom `Dataset`
which bases on top of the `PeiPandarinathNLB2021` dataset provided
in `brainsets`.

First, we list and compute some useful constants in the dataset's constructor.

```python
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
```

Then, we lay out the logic for creating our sampling intervals:
```python
class SimpleNLBMazeDataset(PeiPandarinathNLB2021):
    ...
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
```

Finally, we implement `__getitem__`. We bin the spike times into fixed-width
bins to get a `(num_bins, num_units)` input tensor, and rescale the raw 1 kHz
hand velocity to roughly unit variance for a `(out_samples, out_dim)` target.
```python
    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()  # shape: (num_bins, num_units)

        Y = data.hand.vel / 200.0  # appoximate z-score normalization
        Y = torch.from_numpy(Y).float()  # shape: (out_samples, out_dim)
        return X, Y
```

### Step 2: Samplers and DataLoaders
We use `TrialSampler` to draw one sample per sampling interval (i.e. one per
trial). For training we shuffle; for validation we don't. The dataset's
`__getitem__` is then called by a standard PyTorch `DataLoader`.

```python
train_ds = SimpleNLBMazeDataset(args.data_root, split="train", bin_size=args.bin_size)
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(),
    shuffle=True,
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)

val_ds = SimpleNLBMazeDataset(args.data_root, split="val", bin_size=args.bin_size)
val_sampler = TrialSampler(sampling_intervals=val_ds.get_sampling_intervals())
val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)
```

### Step 3: The model
Three small decoders are defined in `models.py`. All three follow the same
interface: they take a binned-spike tensor of shape `(B, num_bins, num_units)`
and return a velocity prediction of shape `(B, out_samples, out_dim)`.

- **`Linear`** — flattens the input and applies a single `nn.Linear` mapping
  `(num_units * num_bins) -> (out_dim * out_samples)`. Useful as a baseline.
- **`GRU`** — a multi-layer bidirectional GRU over the time axis, followed
  by a per-timestep linear readout and an `AdaptiveAvgPool1d` that upsamples
  from `num_bins` to `out_samples`.
- **`TCN`** — a stack of dilated 1D convolutions (dilation `2**i` at layer `i`)
  with symmetric (non-causal) padding, followed by the same pool + readout
  pattern as the GRU.

The model is selected via `--model` and constructed from the dataset's
precomputed shape constants:
```python
model_class = models.__dict__[args.model]
model = model_class(
    in_units=train_ds.num_units,
    in_bins=train_ds.num_bins,
    out_dim=train_ds.out_dim,
    out_samples=train_ds.out_samples,
).to(device)
```

### Step 4: Training loop
A standard PyTorch loop: MSE loss against the (rescaled) hand velocity, AdamW
optimizer, and an R² score computed on the validation set at the end of each
epoch. Predictions and targets are flattened across the batch and time axes
before scoring so that sklearn's `r2_score` reduces to a single scalar rather
than averaging per-output R² values.

```python
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

for epoch in tqdm(range(args.epochs)):
    model.train()
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        model.eval()
        preds, targets = [], []
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            preds.append(model(X))
            targets.append(Y)
        pred = torch.cat(preds).flatten(0, 1).cpu()
        target = torch.cat(targets).flatten(0, 1).cpu()
        r2 = r2_score(target, pred)
```
