import numpy as np
from temporaldata import IrregularTimeSeries, RegularTimeSeries, Data


class RandomNoise:
    """
    Add Gaussian noise to neural data.

    Behavior:
    ---------
    - RegularTimeSeries (continuous data): Add noise to data array (T × C).
    - IrregularTimeSeries:
         * If amplitudes exist  -> add noise to amplitudes only
         * If no amplitudes     -> no-op (timestamps untouched)

    Args:
        field (str): Path to field, e.g. "lfp.data", "rates.data", "spikes".
        noise_std (float): Standard deviation of Gaussian noise.
        clip (bool): Clip values to min=0 (useful for calcium/rates).
        seed (int): Random seed.
    """

    def __init__(
        self,
        field: str = "rates.amplitudes",
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        clip: bool = False,
        seed: int = None,
    ):
        self.field = field
        self.noise_std = noise_std
        self.clip = clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, data: Data):
        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        if isinstance(obj, RegularTimeSeries):
            assert len(nested) == 2, "Field must be like 'rates.data' or 'lfp.data'"
            arr = getattr(obj, nested[1])

            noise = self.rng.normal(self.noise_mean, self.noise_std, size=arr.shape)
            out = arr + noise

            if self.clip:
                out = np.clip(out, 0, None)

            setattr(obj, nested[1], out)
            return data

        if isinstance(obj, IrregularTimeSeries):
            # If no amplitude information (e.g. spikes), do not modify anything
            if not hasattr(obj, nested[1]):
                raise ValueError(
                    f"Augmentation {type(self)} requires a field with amplitudes to be present in the data."
                )

            amps = getattr(obj, nested[1])
            noise = self.rng.normal(self.noise_mean, self.noise_std, size=amps.shape)
            amps_noisy = amps + noise

            if self.clip:
                amps_noisy = np.clip(amps_noisy, 0, None)

            setattr(obj, nested[1], amps_noisy)
            return data

        raise ValueError(f"Unsupported type for RandomNoise: {type(obj)}")
