"""Shared model utilities for neuroprobe_eval models."""


def to_int_list(value, default, allow_empty=False):
    """Convert scalar/list-like config value to a list of ints."""
    if value is None:
        return list(default)
    if hasattr(value, "__iter__") and not isinstance(value, str):
        out = [int(v) for v in list(value)]
        if out:
            return out
        return [] if allow_empty else list(default)
    if isinstance(value, (int, float)):
        return [int(value)]
    return list(default)
