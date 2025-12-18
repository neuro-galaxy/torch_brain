from typing import Any
from temporaldata import Data


def set_nested_attribute_in_data(data: Data, path: str, value: Any) -> Data:
    f"""Set a nested attribute in a :class:`temporaldata.Data` object using a dot-separated path.

    Args:
        data: The :class:`temporaldata.Data` object to modify.
        path: The dot-separated path to the nested attribute (e.g., "session.id").
        value: The value to set for the attribute.

    Returns:
        The modified data object (same instance, modified in-place).

    Raises:
        AttributeError: If any component of the path cannot be resolved.
    """
    # Split key by dots, resolve using getattr
    components = path.split(".")
    obj = data
    for c in components[:-1]:
        try:
            obj = getattr(obj, c)
        except AttributeError:
            raise AttributeError(
                f"Could not resolve {path} in data (specifically, at level {c}))"
            )

    setattr(obj, components[-1], value)
    return data
