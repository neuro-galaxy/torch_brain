# Migrating from v0.1.x to v0.2.0

The biggest change from v0.1.x to v0.2.0 is the merger of `temporaldata`
and `brainsets` into `torch_brain`.
If you pull `torch_brain` from GitHub, we suggest you start with a fresh clone.

## Installation

See [README.md](../README.md) for updated installation instructions.
Until we release `v0.2.0` on PyPI, you will have to install from GitHub itself.
See [releases page](https://github.com/neuro-galaxy/torch_brain/releases) for updates on releases.

A couple of things to know when coming from v0.1.x:

- Uninstall the old packages first so their stale top-level `temporaldata` /
  `brainsets` modules don't shadow the new ones: `pip uninstall temporaldata brainsets`.
- `torch` is **not** a base dependency anymore, and the `data` and `pipeline`
  modules are designed to work without it. Install `torch` directly from `pip` for other
  modules: `pip install torch git+https://github.com/neuro-galaxy/torch_brain`.

> Renamed imports are not silently aliased to their old location: they raise
> `ImportError`/`AttributeError` pointing at the new path, so every import below
> must be updated.


## Module renames

See below for re-organization of modules in v0.2.0:

### Temporaldata

| Before | After |
|--------|-------|
| `from temporaldata import XYZ` | `from torch_brain.data import XYZ`|
| `from temporaldata.descriptions import XYZ` | `from torch_brain.data import XYZ`|

### Brainsets

| Before | After |
|--------|-------|
| `from brainsets.pipeline import BrainsetPipeline` | `from torch_brain.pipeline import BrainsetPipeline` |
| `from brainsets.utils.openneuro import OpenNeuroPipeline` | `from torch_brain.pipeline.openneuro import OpenNeuroPipeline` |
| `from brainsets.utils.s3_utils import XYZ` | `from torch_brain.utils.s3 import XYZ` |
| `from brainsets.utils.split import XYZ` | `from torch_brain.utils.split import XYZ` |
| `from brainsets.utils.dandi_utils import XYZ` | `from torch_brain.utils.dandi import XYZ` |
| `from brainsets.utils.mne_utils import XYZ` | `from torch_brain.utils.mne import XYZ` |
| `from brainsets.utils.bids_utils import XYZ` | `from torch_brain.utils.bids import XYZ` |
| `from brainsets.processing.signal import XYZ` | `from torch_brain.utils.signal import XYZ` |
| `from brainsets.datasets import XYZ` | `from torch_brain.datasets import XYZ` |

`brainsets` CLI remains the same as before.

### TorchBrain

| Before | After |
|--------|-------|
| `from torch_brain.data.sampler import XYZ` |  `from torch_brain.samplers import XYZ` |
| `from torch_brain.data.collate import XYZ` | `from torch_brain.batching import XYZ` |
| `from torch_brain.dataset import XYZ` | `from torch_brain.datasets import XYZ` |
| `from torch_brain.utils import stitch` | `from torch_brain.utils.stitcher import stitch` |

Old `torch_brain.data.Dataset` has not been completely deprecated. Please switch to `torch_brain.datasets.Dataset`.

These POYO-specific utilities were moved:

| Before | After |
|--------|-------|
| `from torch_brain.utils import create_start_end_unit_tokens` | `from torch_brain.models.poyo import create_start_end_unit_tokens` |
| `from torch_brain.utils import create_linspace_latent_tokens` | `from torch_brain.models.poyo import create_linspace_latent_tokens` |

## Removed APIs

Additionally, a number of modules/functions were removed because they were not
general enough to keep supporting. The table lists a replacement where one exists.

| Removed | Replacement / status |
|---------|----------------------|
| `torch_brain.utils.get_sinusoidal_encoding` | Use `torch_brain.nn.SinusoidalTimeEmbedding` |
| `torch_brain.utils.seed_everything` | Reference implementation in `examples/poyo/utils.py` |
| `torch_brain.optim` | Removed; `SparseLamb` implementation can be found in `examples/poyo` |
| `torch_brain.registry` | Removed, no replacement |
| `torch_brain.nn.loss` | Removed, no replacement |
| `torch_brain.nn.MultitaskReadout` | Removed, no replacement |
| `torch_brain.nn.FeedForwad` | Removed, no replacement |
| `torch_brain.utils.gradient_rescale` | Removed, no replacement |
| `torch_brain.utils.prepare_for_readout` | Removed, no replacement |
| `DecodingStitchEvaluator`, `MultiTaskDecodingStitchEvaluator` (from `torch_brain.utils.callbacks`) | Removed; see `examples/poyo` for an alternate implementation of `DecodingStitchEvaluator` |
| `POYOPlus`, `CalciumPOYOPlus` models | Removed; relied on previously removed modules |

## Data files (HDF5)

The on-disk HDF5 format is unchanged by the merger, so datasets you preprocessed
with `brainsets`/`temporaldata` v0.1.x continue to load with v0.2.0. No
re-processing _should_ be required. Although it might be better to just re-run
`brainsets prepare` if possible.
