# Migrating from v0.1.x to v0.2.0

The biggest change from v0.1.x to v0.2.0 is that we have merged `temporaldata`
and `brainsets` into `torch_brain`

If you pull torch_brain from GitHub, we suggest you start with a fresh clone.


## Module renames

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


