import numpy as np


def numpy_string_prefix(prefix: str, array: np.ndarray) -> np.ndarray:
    """
    Adds a string prefix to each element of a numpy string array.

    Args:
        prefix (str): The string to prepend to each element.
        array (np.ndarray): An array of strings or string-like objects.

    Returns:
        np.ndarray: New array with the prefix added to each element.
    """
    if np.__version__ >= "2.0":
        return np.strings.add(prefix, array)
    else:
        return np.core.defchararray.add(prefix, array)
