import numpy as np
from typing import Optional
from temporaldata import Data  # type: ignore


# torch_brain/transforms/patch_tokenize.py
import numpy as np
from temporaldata import Data  # assumes temporaldata.Data
from typing import Optional


class PatchTokenize:
    """
    In-place patch tokenization for regularly sampled series stored as
    data.<key>.values (T, C) and data.<key>.timestamps (T,).

    What it does
    ------------
    - Clips T to a multiple of `patch_size`
    - Reshapes values to:
        * (num_patches, patch_size, C) when flatten=False (default)
        * (num_patches*C, patch_size) when flatten=True
    - Replaces timestamps with the **mean timestamp per patch** (shape: (num_patches,))
    - Adds/overwrites `data.num_patches`

    Assumptions
    -----------
    - `getattr(data, key)` is a container with `.values` (ndarray) and
      `.timestamps` (ndarray) attributes typical of RegularTimeSeries-like objects.
    - `.values` is (T, C) or (T,) and `.timestamps` is (T,).

    Parameters
    ----------
    key : str
        Attribute on `data` that holds the series container (e.g., "traces" for zappbench dataset).
    patch_size : int
        Number of consecutive frames per patch.
    flatten : bool
        If True, outputs (num_patches*C, patch_size). Otherwise (num_patches, patch_size, C).
    """

    def __init__(self, key: str = "traces", patch_size: int = 16, flatten: bool = False):
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self.key = key
        self.patch_size = patch_size
        self.flatten = bool(flatten)

    def __call__(self, data: Data) -> Data:
        # --- locate the series container and fetch arrays
        series = getattr(data, self.key, None)
        if series is None:
            raise KeyError(f"Data has no attribute '{self.key}'")

        values = np.asarray(series.values)
        if values.ndim == 1:
            values = values[:, None]  # (T,) -> (T, 1)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D values for data.{self.key}.values, got {values.shape}")

        ts = np.asarray(series.timestamps)
        if ts.ndim != 1 or ts.shape[0] != values.shape[0]:
            raise ValueError(
                f"data.{self.key}.timestamps must be 1D with same length as values' T; "
                f"got timestamps {ts.shape} vs values T {values.shape[0]}"
            )

        # --- make T a multiple of patch_size
        T, C = values.shape
        usable_T = (T // self.patch_size) * self.patch_size
        if usable_T == 0:
            raise ValueError(f"patch_size ({self.patch_size}) is larger than sequence length ({T})")

        values = values[:usable_T]
        ts = ts[:usable_T]
        num_patches = usable_T // self.patch_size

        # --- reshape into patches
        # blocks shape (num_patches, patch_size, C)
        blocks = values.reshape(num_patches, self.patch_size, C)

        # timestamps per patch -> mean  over each patch
        patch_ts = ts.reshape(num_patches, self.patch_size).mean(axis=1)

        # --- write back in place
        if self.flatten:
            # (num_patches, patch_size, C) -> (num_patches*C, patch_size)
            series.values = blocks.transpose(0, 2, 1).reshape(num_patches * C, self.patch_size)
        else:
            series.values = blocks  # (num_patches, patch_size, C)

        series.timestamps = patch_ts  # (num_patches,) shape
        data.num_patches = int(num_patches)

        # In-place modification to the original data complete; return the same object for transform chaining

        return data
