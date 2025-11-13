import numpy as np
from temporaldata import IrregularTimeSeries, RegularTimeSeries, Data


class RandomNoise:
    """
    Add Gaussian noise.

    Disclaimer: This transform is only compatible with IrregularTimeSeries.

    Args:
        field (str): Path to field, e.g. "lfp.data", "rates.data", "spikes".
        noise_mean (float): Mean of Gaussian noise.
        noise_std (float): Standard deviation of Gaussian noise.
        clip (bool): Clip values to min=0.
        seed (int): Random seed.
    """

    def __init__(
        self,
        field: str = "cursor.acc",
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        clip: bool = False,
        seed: int = None,
    ):
        self.field = field
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.clip = clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, data: Data):

        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        # TODO Handle RegularTimeSeries
        if isinstance(obj, IrregularTimeSeries):

            amps = getattr(obj, nested[1])
            noise = self.rng.normal(self.noise_mean, self.noise_std, size=amps.shape)
            amps_noisy = amps + noise

            if self.clip:
                amps_noisy = np.clip(amps_noisy, 0, None)

            setattr(obj, nested[1], amps_noisy)
            return data

        raise ValueError(f"Unsupported type for RandomNoise: {type(obj)}")
