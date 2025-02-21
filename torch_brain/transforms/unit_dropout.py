import logging
from typing import Optional

import numpy as np

from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries


class TriangleDistribution:
    r"""Triangular distribution with a peak at mode_units, going from min_units to
    max_units. 

    The unnormalized density function is defined as:
    
    .. math::
        f(x) = 
        \begin{cases} 
        0 & \text{if } x < \text{min_units} \\
        1 + (\text{peak} - 1) \cdot \frac{x - \text{min_units}}{\text{mode_units} - \text{min_units}} & \text{if } \text{min_units} \leq x \leq \text{mode_units} \\
        \text{peak} - (\text{peak} - 1) \cdot \frac{x - \text{mode_units}}{\text{tail_right} - \text{mode_units}} & \text{if } \text{mode_units} \leq x \leq \text{tail_right} \\
        1 & \text{if } \text{tail_right} \leq  x \leq \text{max_units}\\
        0 & \text{otherwise}
        \end{cases}

    Args:
        min_units (int): Minimum number of units to sample. If the population has fewer
            units than this, all units will be kept.
        mode_units (int): Mode of the distribution.
        max_units (int): Maximum number of units to sample. 
        tail_right (int, optional): Right tail of the distribution. If None, it is set to
            `max_units`.
        peak (float, optional): Height of the peak of the distribution.
        M (float, optional): Normalization constant for the proposal distribution.
        max_attempts (int, optional): Maximum number of attempts to sample from the
            distribution.
        seed (int, optional): Seed for the random number generator.

    .. image:: ../_static/img/triangle_distribution.png

    To sample from the distribution, we use rejection sampling. We sample from a uniform
    distribution between `min_units` and `max_units` and accept the sample with
    probability :math:`\frac{f(x)}{M \cdot q(x)}`, where :math:`q(x)` is the proposal
    distribution. 
    """

    def __init__(
        self,
        min_units: int = 20,
        mode_units: int = 100,
        max_units: int = 300,
        tail_right: Optional[int] = None,
        peak: float = 4,
        M: int = 10,
        max_attempts: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.min_units = min_units
        self.mode_units = mode_units
        self.max_units = max_units
        self.tail_right = tail_right if tail_right is not None else max_units
        self.peak = peak
        self.M = M
        self.max_attempts = max_attempts

        # TODO pass a generator?
        self.rng = np.random.default_rng(seed=seed)

    def unnormalized_density_function(self, x):
        if x < self.min_units:
            return 0
        if x <= self.mode_units:
            return 1 + (self.peak - 1) * (x - self.min_units) / (
                self.mode_units - self.min_units
            )
        if x <= self.tail_right:
            return self.peak - (self.peak - 1) * (x - self.mode_units) / (
                self.tail_right - self.mode_units
            )
        return 1

    def proposal_distribution(self, x):
        return self.rng.uniform()

    def sample(self, num_units):
        if num_units < self.min_units:
            return num_units

        # uses rejection sampling
        num_attempts = 0
        while True:
            x = self.min_units + self.rng.uniform() * (
                self.max_units - self.min_units
            )  # Sample from the proposal distribution
            u = self.rng.uniform()
            if u <= self.unnormalized_density_function(x) / (
                self.M * self.proposal_distribution(x)
            ):
                return x
            num_attempts += 1
            if num_attempts > self.max_attempts:
                logging.warning(
                    f"Could not sample from distribution after {num_attempts} attempts,"
                    " using all units."
                )
                return num_units


class UnitDropout:
    r"""Augmentation that randomly drops units from the sample. By default, the number
    of units to keep is sampled from a triangular distribution defined in
    :class:`TriangleDistribution`.

    This transform assumes that the data has a `units` object. It works for both
    :class:`IrregularTimeSeries` and :class:`RegularTimeSeries`. For the former, it will
    drop spikes from the units that are not kept. For the latter, it will drop the
    corresponding columns from the data.

    Args:
        field (str, optional): Field to apply the dropout. Defaults to "spikes".
        *args, **kwargs: Arguments to pass to the :class:`TriangleDistribution` constructor.
    """

    def __init__(self, field: str = "spikes", reset_index=True, *args, **kwargs):
        # TODO allow multiple fields (example: spikes + LFP)
        self.field = field
        self.reset_index = reset_index
        # TODO this currently assumes the type of distribution we use, in the future,
        # the distribution might be passed as an argument.
        self.distribution = TriangleDistribution(*args, **kwargs)

    def __call__(self, data: Data):
        # get units from data
        unit_ids = data.units.id
        num_units = len(unit_ids)

        # sample the number of units to keep from the population
        num_units_to_sample = int(self.distribution.sample(num_units))

        # shuffle units and take the first num_units_to_sample
        keep_indices = np.random.permutation(num_units)[:num_units_to_sample]

        unit_mask = np.zeros_like(unit_ids, dtype=bool)
        unit_mask[keep_indices] = True
        if self.reset_index:
            data.units = data.units.select_by_mask(unit_mask)

        nested_attr = self.field.split(".")
        target_obj = getattr(data, nested_attr[0])
        if isinstance(target_obj, IrregularTimeSeries):
            # make a mask to select spikes that are from the units we want to keep
            spike_mask = np.isin(target_obj.unit_index, keep_indices)

            # using lazy masking, we will apply the mask for all attributes from spikes
            # and units.
            setattr(data, self.field, target_obj.select_by_mask(spike_mask))

            if self.reset_index:
                relabel_map = np.zeros(num_units, dtype=int)
                relabel_map[unit_mask] = np.arange(unit_mask.sum())

                target_obj = getattr(data, self.field)
                target_obj.unit_index = relabel_map[target_obj.unit_index]
        elif isinstance(target_obj, RegularTimeSeries):
            assert len(nested_attr) == 2
            setattr(
                target_obj,
                nested_attr[1],
                getattr(target_obj, nested_attr[1])[:, unit_mask],
            )
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")

        return data
