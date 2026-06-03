from enum import Enum

import numpy as np


class TokenType(Enum):
    DEFAULT = 0
    START_OF_SEQUENCE = 1
    END_OF_SEQUENCE = 2


def create_start_end_unit_tokens(unit_ids: np.ndarray, start: float, end: float):
    r"""Creates for each unit a start and end token. Each token is defined by the
    unit index, the token type index and the timestamps.

    Args:
        unit_ids: List of unit identifiers.
        start: The start time of the sequence.
        end: The end time of the sequence.
    """
    U = len(unit_ids)

    token_type_index = np.array(
        [TokenType.START_OF_SEQUENCE.value, TokenType.END_OF_SEQUENCE.value],
        dtype=np.int64,
    )
    token_type_index = np.tile(token_type_index, U)  # (2,) -> (U*2,)

    unit_index = np.arange(U)
    unit_index = np.repeat(unit_index, 2)  # (U,) -> (U*2,)

    timestamps = np.array([start, end], dtype=np.float64)
    timestamps = np.tile(timestamps, U)  # (2,) -> (U*2,)
    return token_type_index, unit_index, timestamps


def create_linspace_latent_tokens(
    start: float, end: float, step: float, num_latents_per_step: int
):
    r"""Creates a sequence of latent tokens. Each token is defined by the
    latent index and the timestamps. The sequence is defined by the start and end
    time and the step size. The group of `num_latents_per_step` latents is repeated
    for each step.

    Args:
        start: The start time of the sequence.
        end: The end time of the sequence.
        step: The step size.
        num_latents_per_step: The number of latents per step.
    """
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    T = len(latent_timestamps)
    U = len(latent_index)

    latent_timestamps = np.repeat(latent_timestamps, U)  # (T,) -> (T*U,)
    latent_index = np.tile(latent_index, T)  # (U,) -> (T*U,)
    return latent_index, latent_timestamps
