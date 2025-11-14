import copy
import re
import numpy as np
from temporaldata import Data

def bipolar(data: Data):
    r"""Bipolar rereferencing based on channel names.

    Args:
        data (Data): The data to rereference.
    """
    combined_ieeg = data.ieeg.data
    ch_names = copy.deepcopy(data.units.id)
    n_channels = np.shape(combined_ieeg)[1]

    # Rereference the neural activity
    for i in range(n_channels):
        ch_names[i] = re.sub(r'\d+$', '', ch_names[i])
    ch_contiguity_bool = [(ch_names[i] == ch_names[i+1]) for i in range(n_channels - 1)]
    ch_contiguity = np.where(ch_contiguity_bool)[0]

    combined_ieeg = combined_ieeg[:, 1:] - combined_ieeg[:, :-1]
    combined_ieeg = combined_ieeg[:, ch_contiguity]

    # Update the channel names
    ch_contiguity_bool.append(False)
    ch_contiguity_bool = np.array(ch_contiguity_bool)

    units = data.units.select_by_mask(ch_contiguity_bool)

    # Put the rereferenced data back into the data object
    data.ieeg.data = combined_ieeg
    data.units = units

    return data

def common_average(data: Data):
    r"""Common average rereferencing.

    Args:
        data (Data): The data to rereference.
    """
    combined_ieeg = data.ieeg.data
    combined_ieeg = combined_ieeg - np.mean(combined_ieeg, axis=1, keepdims=True)

    # Put the rereferenced data back into the data object
    data.ieeg.data = combined_ieeg

    return data

def laplace(data: Data):
    r"""Laplacian rereferencing based on channel names.

    Args:
        data (Data): The data to rereference.
    """
    combined_ieeg = data.ieeg.data
    ch_names = copy.deepcopy(data.units.id)
    n_channels = np.shape(combined_ieeg)[1]

    # Rereference the neural activity
    combined_ieeg_reref = copy.deepcopy(combined_ieeg)
    for i in range(n_channels):
        ch_names[i] = re.sub(r'\d+$', '', ch_names[i])
    
    ch_non_contiguity_bool = [(ch_names[i] != ch_names[i+1]) for i in range(n_channels - 1)]
    ch_non_contiguity = np.where(ch_non_contiguity_bool)[0]

    right_neighbors = np.arange(n_channels) + 1.
    right_neighbors[ch_non_contiguity] = ch_non_contiguity - 1
    right_neighbors[-1] = n_channels - 2

    left_neighbors = np.arange(n_channels) - 1.
    left_neighbors[ch_non_contiguity + 1] = ch_non_contiguity + 2
    left_neighbors[0] = 1

    left_neighbors = left_neighbors.astype(int)
    right_neighbors = right_neighbors.astype(int)

    combined_ieeg = combined_ieeg - (combined_ieeg[:, left_neighbors] + 
                                    combined_ieeg[:, right_neighbors]) / 2.

    # Put the rereferenced data back into the data object
    data.ieeg.data = combined_ieeg

    return data

class Rereferencing:
    def __init__(self, type="bipolar"):
        self.type = type

    def __call__(self, data):
        match self.type:
            case "bipolar":
                return(bipolar(data))
            case "common_average":
                return(common_average(data))
            case "laplace":
                return(laplace(data))
            case _:
                raise ValueError("Invalid rereferencing type.")
