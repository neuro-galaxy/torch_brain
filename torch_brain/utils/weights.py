import numpy as np
from temporaldata import Interval


def isin_interval(timestamps: np.ndarray, interval: Interval) -> np.ndarray:
    r"""Check if timestamps are in any of the intervals in the `Interval` object.

    Args:
        timestamps: Timestamps to check.
        interval: Interval to check against.

    Returns:
        Boolean mask of the same shape as `timestamps`.
    """
    if len(interval) == 0:
        return np.zeros_like(timestamps, dtype=bool)

    timestamps_expanded = timestamps[:, None]
    mask = np.any(
        (timestamps_expanded >= interval.start) & (timestamps_expanded < interval.end),
        axis=1,
    )
    return mask


def resolve_weights_based_on_interval_membership(timestamps, data, config=None):
    """Determine weights for timestamps based on which intervals they fall within.
    The intervals and corresponding weights are specified in the config dictionary.

    The config dictionary maps interval names (nested notation allowed) to weight values.
    For example:
        {
            'movement_periods.random_period': 1.0,
            'movement_periods.hold_period': 0.1,
            'movement_periods.reach_period': 5.0,
            'movement_periods.return_period': 1.0,
            'cursor_outlier_segments': 0.0
        }

    These weights can be used to weight different time periods differently in the loss
    function. In the example above, reach periods are weighted 5x more heavily than
    random periods.

    .. note::
        If intervals overlap, the final weight will be the product of all weights
        from those intervals. For example, if a timestamp falls within both a
        reach_period (weight 5.0) and a cursor_outlier_segments (weight 0.0), its
        final weight will be 5.0 * 0.0 = 0.0. This multiplicative behavior allows for
        complex weighting schemes where other intervals can be combined.

    .. note::
        If a timestamp does not belong to any of the intervals in the config,
        its weight will remain at the default value of 1.0.

    Args:
        timestamps: Array of timestamps
        data: Data object containing intervals
        config: Dictionary mapping interval names to weight values

    Returns:
        Array of weights with same shape as timestamps
    """
    weights = np.ones_like(timestamps, dtype=np.float32)
    if config is not None:
        for weight_key, weight_value in config.items():
            # extract the interval from the weight key
            weight = data.get_nested_attribute(weight_key)
            if not isinstance(weight, Interval):
                raise ValueError(
                    f"Weight {weight_key} is of type {type(weight)}. Expected an "
                    "Interval object."
                )
            weights[isin_interval(timestamps, weight)] *= weight_value
    return weights
