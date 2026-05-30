import numpy as np


def get_sinusoidal_encoding(x, y, dim):
    assert dim % 2 == 0, "Number of dimensions should be even"
    assert len(x) == len(y), "x and y arrays must be of the same length"

    # Creating scale factors that decrease exponentially
    scale_factors = 1 / np.power(10000, (2 * (np.arange(dim // 2)) / dim))

    # Initialize an array to hold the encodings for all positions
    all_encodings = np.zeros((len(x), dim * 2))

    # Apply sinusoidal encoding to each pair of positions
    for i, (pos_x, pos_y) in enumerate(zip(x, y)):
        encoding_x = np.array(
            [np.sin(pos_x * scale_factors), np.cos(pos_x * scale_factors)]
        ).flatten("F")
        encoding_y = np.array(
            [np.sin(pos_y * scale_factors), np.cos(pos_y * scale_factors)]
        ).flatten("F")
        all_encodings[i] = np.concatenate((encoding_x, encoding_y))

    return all_encodings
