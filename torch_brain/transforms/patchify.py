import numpy as np
from temporaldata import Data, IrregularTimeSeries , RegularTimeSeries # type: ignore
from typing import Optional


class PatchTokenize:
    """
    In-place patchifier for regularly and iregularly sampled series as intance of IrregularTimeSeries , RegularTimeSeries

    After this transform:
    - data.<key> is an IrregularTimeSeries with:
        * timestamps: (N,)
        * values: (N, patch_size, C)
    - or data.<key> is an RegularTimeSeries with:
        * sampling_rate as a float number
        * values: (N, patch_size, C)
        * domain.start to be adjusted to be the center of the first patch

    - data.num_patches is set to num_patches (before channel flattening).

    Assumptions
    -----------
    - getattr(data, key) has `.values` and `.timestamps` attributes for IrregularTimeSeries, or sampling_rate and domain for RegularTimeSeries
    - `.values` is (T, C) or (T,) and `.timestamps` is (T,).
    """

    def __init__(self, key: str = "traces", patch_size: int = 1):
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self.key = key
        self.patch_size = patch_size

    def __call__(self, series: Data) -> Data:

        values = getattr(series, self.key, None) # the value stored in the regular or irregular time series object
        if values is None:
            raise KeyError(f"Data has no attribute '{self.key}'")

        # --- clip T to a multiple of patch_size
        T, C = values.shape
        usable_T = (T // self.patch_size) * self.patch_size
        if usable_T == 0:
            raise ValueError(
                f"patch_size ({self.patch_size}) is larger than sequence length ({T})"
            )

        values = values[:usable_T]
        num_patches = int( usable_T // self.patch_size )

        # --- reshape into patches
        # blocks: (num_patches, patch_size, C)
        patches = values.reshape(num_patches, self.patch_size, C)

        # --- Regular vs. Irregular handling
        if isinstance(series, RegularTimeSeries):
            # Treat as RegularTimeSeries:
            # new effective sampling rate = old * patch_size
            sr = getattr(series, "sampling_rate", None)
            t0 = float(series.domain.start)
            series.sampling_rate = float(sr) * self.patch_size
            series.domain.start= t0 + ( float(sr) * self.patch_size ) / 2
            setattr(series, self.key, patches)

        elif isinstance(series, IrregularTimeSeries):
            ts = np.asarray(series.timestamps)
            ts = ts[:usable_T]
            irg_patch_ts = ts.reshape(num_patches, self.patch_size).mean(axis=1)
            # --- build a new IrregularTimeSeries with only timestamps + values, i.e the values we want to patch
            series = IrregularTimeSeries(
            timestamps=irg_patch_ts,
            domain="auto",
            **{self.key: patches},
            )

        else:
            raise TypeError(f"Unknown series type that is not handled in patchify operation: {type(series)}")

        series.num_patches = num_patches

        return series
