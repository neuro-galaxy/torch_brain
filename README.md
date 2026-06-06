<p align="center">
    <img height="250" src="https://torch-brain.readthedocs.io/en/latest/_static/torch_brain_logo.png" />
</p>

<h1 align="center">torch_brain</h1>

<p align="center">
    <a href="https://torch-brain.readthedocs.io/en/latest/">Documentation</a>
    |
    <a href="https://discord.gg/kQNKA6B8ZC">Join our Discord community</a>
</p>

<p align="center">
    <a href="https://badge.fury.io/py/torch_brain"><img src="https://badge.fury.io/py/torch_brain.svg" alt="PyPI version" /></a>
    <a href="https://torch-brain.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/torch-brain/badge/?version=latest" alt="Documentation Status" /></a>
    <a href="https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml"><img src="https://github.com/neuro-galaxy/torch_brain/actions/workflows/testing.yml/badge.svg" alt="Tests" /></a>
    <a href="https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml"><img src="https://github.com/neuro-galaxy/torch_brain/actions/workflows/linting.yml/badge.svg" alt="Linting" /></a>
    <a href="https://discord.gg/kQNKA6B8ZC"><img src="https://img.shields.io/discord/1338561153089146962?label=Discord&logo=discord" alt="Discord" /></a>
</p>

> [!NOTE]
> We have merged `temporaldata` and `brainsets` into `torch_brain`.
> If you are migrating from v0.1.x, please see [this migration guide](howto/MIGRATE_TO_v0.2.md).

**torch_brain** is an end-to-end framework for building deep learning models
and training pipelines for neuroscience. It pairs a lightweight, time-based data
format (plus tools to preprocess existing neural datasets into it) with
PyTorch-compatible building blocks: datasets, samplers, `nn.Module`s, and
models.

## Features

- Multi-recording training across heterogeneous datasets
- Support for arbitrary neural and behavioral modalities
- Lazy, on-demand data loading that reads only the time-slices and attributes you request
- Advanced samplers for arbitrary on-the-fly slicing of recordings
- Flexible collation strategies, including chaining and padding

## Installation

**torch_brain** requires Python >= 3.10. To install a stable release:

```bash
pip install torch torch_brain
```

> [!TIP]
> If you only need `torch_brain.data` and the data-preparation pipelines, you
> can skip installing `torch`.

<details>
    <summary>Latest development version:</summary>

Install the latest (unstable) development version via the main branch:

```bash
pip install git+https://github.com/neuro-galaxy/torch_brain
```
</details>

## The data format

A recording is a `Data` object holding heterogeneous, time-aware modalities:
regularly-sampled signals (LFP, EEG, ...), irregular event streams (spikes),
interval annotations (trials), and plain arrays.

```python
import numpy as np
from torch_brain.data import Data, IrregularTimeSeries, RegularTimeSeries, Interval

data = Data(
    spikes=IrregularTimeSeries(                       # event stream
        timestamps=[0.1, 0.2, 0.3, 2.1, 2.2, 2.3],
        unit_index=[0, 0, 1, 0, 1, 2],
        domain="auto",
    ),
    lfp=RegularTimeSeries(raw=np.zeros((1000, 3)), sampling_rate=250.0),  # 4s @ 250Hz
    trials=Interval(start=[0, 1, 2], end=[1, 2, 3]),  # annotations
    domain=Interval(0.0, 4.0),
)
```

The point of the format is that **slicing is time-based and lazy**:
Every modality is sliced consistently, regardless of their different
sampling rates, and the data is lazily read from disk so only the
requested window and attributes are loaded.

```python
window = data.slice(1.0, 3.0)
# spikes -> the 3 events in [1, 3)   lfp -> 500 samples   trials -> 2 trials
```

This is why a torch_brain `Dataset` is indexed by time, not by integer (see below).

## Training pipelines

torch_brain leans on the standard PyTorch training loop, and most of its job is
to handle the data side. You define a `Dataset` (built on the time-slicing
above) and a `Sampler` that decides which slices become samples. The
`DataLoader`, model, and loop are ordinary PyTorch.

```python
import torch
from torch.utils.data import DataLoader
from torch_brain.datasets import PeiPandarinathNLB2021, DatasetIndex
from torch_brain.samplers import TrialSampler
from torch_brain.utils import bin_spikes

# torch_brain ships loaders for many public datasets.
# Subclass one to define the two things specific to your task:
class MyDataset(PeiPandarinathNLB2021):
    # 1. WHICH windows count as samples (here, one per behavioral trial).
    def get_sampling_intervals(self):
        sampling_intervals = {}
        for rid in self.recording_ids:
            sampling_intervals[rid] = self.get_recording(rid).trials
        return sampling_intervals

    # 2. HOW one window becomes tensors.
    def __getitem__(self, index: DatasetIndex):
        # `index` is a DatasetIndex(recording_id, start, end) handed in by the sampler;

        data = super().__getitem__(index)
        # super().__getitem__(...) returns that slice with
        # every modality (.spikes, .hand.vel, ...) lazily cropped.

        # Only attributes actually accessed will be loaded into memory from disk.
        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=0.05)
        Y = data.hand.vel
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

dataset = MyDataset(root="data/processed", recording_ids=["jenkins_maze_train"])

# The sampler turns those intervals into per-sample DatasetIndex objects.
sampler = TrialSampler(sampling_intervals=dataset.get_sampling_intervals(), shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=8)

# From here on it's plain PyTorch
for X, Y in loader:
    pred = model(X)
    loss = loss_fn(pred, Y)
    ...
```

The key idea: unlike a standard PyTorch `Dataset` indexed by integers, a torch*brain `Dataset` is indexed by \_time-slices*, and loads data lazily, so only the slice you ask for is read from disk. A `Sampler` decides _what_ to load, the `Dataset` decides _how_, and everything downstream stays vanilla PyTorch.

See [`examples/`](examples/) for simple and readable training implementations.

## Quickstart

Download and preprocess a dataset, then train a model:

```bash
# Prepare a brainset
brainsets prepare perich_miller_population_2018 --raw-dir data/raw --processed-dir data/processed

# Train POYO on the prepared data
cd examples/poyo
python train_simple.py --config-name train_poyo_mp data_root=data/processed
```

## Research powered by `torch_brain`

- [POYO](https://poyo-brain.github.io/) (Azabou et al. NeurIPS 2023)
- [POYO+](https://github.com/nerdslab/poyo_plus) (Azabou and Pan et al. ICLR 2025)
- [POSSM](https://possm-brain.github.io/) (Ryoo and Krishna et al. NeurIPS 2025)
- [NuCLR](https://nerdslab.github.io/nuclr/) (Arora et al. NeurIPS 2025)

## Contributing

Contributions are welcome! Get started with:

```bash
pip install -e ".[dev]"   # editable install with dev dependencies
pre-commit install        # formatting & lint hooks
pytest                    # run the test suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and code-style guidelines.

## Building the documentation

```bash
pip install -e ".[dev,docs]"
cd docs && make clean html
```

The built docs are placed in `docs/build/html`.
