from typing import Callable, List
import copy
import logging

import numpy as np

import temporaldata


class Compose:
    r"""Compose several transforms together. All transforms will be called sequentially,
    in order, and must accept and return a single :obj:`temporaldata.Data` object, except
    the last transform, which can return any object.

    Args:
        transforms: list of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: temporaldata.Data) -> temporaldata.Data:
        for transform in self.transforms:
            data = transform(data)
        return data


# similar to torchvision.transforms.v2.RandomChoice
class RandomChoice:
    r"""Apply a single transformation randomly picked from a list.

    Args:
        transforms: list of transformations
        p: probability of each transform being picked.
            If :obj:`p` doesn't sum to 1, it is automatically normalized. By default,
            all transforms have the same probability.
    """

    def __init__(
        self, transforms: List[Callable], p: List[float] | None = None
    ) -> None:
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: "
                f"{len(p)} != {len(transforms)}"
            )

        super().__init__()

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]

    def __call__(self, data: temporaldata.Data) -> temporaldata.Data:
        idx = np.random.choice(len(self.transforms), p=self.p)
        transform = self.transforms[idx]
        return transform(data)


# args similar to jax.lax.cond
class ConditionalChoice:
    r"""Conditionally apply a single transformation based on whether a condition is met.

    Args:
        condition: callable that takes a data object and returns a boolean
        true_transform: transformation to apply if the condition is met
        false_transform: transformation to apply if the condition is not met
    """

    def __init__(
        self, condition: Callable, true_transform: Callable, false_transform: Callable
    ) -> None:
        self.condition = condition
        self.true_transform = true_transform
        self.false_transform = false_transform

    def __call__(self, data: temporaldata.Data) -> temporaldata.Data:
        ret = self.condition(data)
        if not isinstance(ret, bool):
            raise ValueError(
                f"Condition must return a boolean, got {type(ret)} instead."
            )

        if ret:
            return self.true_transform(data)
        else:
            return self.false_transform(data)


class SkipOnFailure:
    r"""Safely apply a single transform and skip it on failure.

    If the wrapped transform raises an exception, this container returns the input
    unchanged.

    Example:
        >>> from torch_brain.transforms import SkipOnFailure
        >>> class RequireMinUnits:
        ...     def __init__(self, min_units):
        ...         self.min_units = min_units
        ...     def __call__(self, data):
        ...         if len(data.units) < self.min_units:
        ...             raise ValueError("not enough units")
        ...         return data
        >>> transform = SkipOnFailure(RequireMinUnits(min_units=10))

    Args:
        transform: transformation to attempt to apply.
        backup_copy: whether to make a backup copy of the input data. Set to True if the
            transform risks mutating the input data in-place before raising an exception.
            Defaults to False.
        warn: whether to emit a warning when the transform fails. Defaults to False.
    """

    def __init__(
        self, transform: Callable, backup_copy: bool = False, warn: bool = False
    ) -> None:
        self.transform = transform
        self.backup_copy = backup_copy
        self.warn = warn

    def __call__(self, data: temporaldata.Data) -> temporaldata.Data:
        if self.backup_copy:
            data_backup = copy.deepcopy(data)

        try:
            return self.transform(data)
        except Exception as e:
            if self.backup_copy:
                if self.warn:
                    logging.warning(
                        f"Transform {self.transform} failed. The following exception was raised: {e}\n"
                        f"Restoring pre-transform data."
                    )
                return data_backup

            else:
                if self.warn:
                    logging.warning(
                        f"Transform {self.transform} failed. The following exception was raised: {e}\n"
                        f"Returning original data (may be partially mutated)."
                    )
                return data
