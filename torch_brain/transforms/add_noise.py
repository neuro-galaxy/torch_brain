import numpy as np
from temporaldata import IrregularTimeSeries, Data


class RandomNoise:
    """
    Add Gaussian noise.

    Disclaimer: This transform is only compatible with IrregularTimeSeries.

    Args:
        field (str, optional): Path to field, e.g. "lfp.data", "rates.data".
        noise_mean (float): Mean of Gaussian noise.
        noise_std (float): Standard deviation of Gaussian noise.
        distribution (str): Distribution to use for noise. It must be one of 'gaussian',
        'laplace', and 'uniform'. Defaults to 'gaussian'.
        kind (str, optional): How the noise is added to the original time series. It must be either
        'additive' or 'multiplicative'. Defaults to 'additive'.
        clip (bool): Clip values to min=0.
        seed (int): Random seed.
    """

    def __init__(
        self,
        field: str = "cursor.acc",
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        distribution: str = "gaussian",
        kind: str = "additive",
        clip: bool = False,
        seed: int = None,
    ):
        self.field = field
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.distribution = distribution
        self.kind = kind
        self.clip = clip
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: Data):

        nested = self.field.split(".")
        obj = getattr(data, nested[0])

        # TODO Handle RegularTimeSeries
        if isinstance(obj, IrregularTimeSeries):

            values = getattr(obj, nested[1])
            if self.distribution == "gaussian":
                noise = self.rng.normal(
                    loc=self.noise_mean, scale=self.noise_std, size=values.shape
                )
            elif self.distribution == "laplace":
                noise = self.rng.laplace(
                    loc=self.noise_mean, scale=self.noise_std, size=values.shape
                )
            elif self.distribution == "uniform":
                # Convert mean + std → uniform bounds
                a = self.noise_std * np.sqrt(3)  # half-width for matching std
                low = self.noise_mean - a
                high = self.noise_mean + a
                noise = self.rng.uniform(low=low, high=high, size=values.shape)
            else:
                raise ValueError(f"Invalid distribution: {self.distribution}")

            if self.kind == "additive":
                values_noisy = values + noise
            elif self.kind == "multiplicative":
                values_noisy = values * noise
            else:
                raise ValueError(f"Invalid kind: {self.kind}")

            if self.clip:
                values_noisy = np.clip(values_noisy, 0, None)

            setattr(obj, nested[1], values_noisy)
            return data

        raise ValueError(f"Unsupported type for RandomNoise: {type(obj)}")
