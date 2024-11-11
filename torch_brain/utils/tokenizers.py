import numpy as np
from einops import repeat
from enum import Enum


class TokenType(Enum):
    DEFAULT = 0
    START_OF_SEQUENCE = 1
    END_OF_SEQUENCE = 2


def create_start_end_unit_tokens(unit_ids, start, end):
    r"""Creates for each unit a start and end token. Each token is defined by the
    unit index, the token type index and the timestamps.

    Args:
        unit_ids (np.ndarray): List of unit identifiers.
        start (float): The start time of the sequence.
        end (float): The end time of the sequence.
    """
    token_type_index = np.array(
        [TokenType.START_OF_SEQUENCE.value, TokenType.END_OF_SEQUENCE.value],
        dtype=np.int64,
    )
    token_type_index = repeat(token_type_index, "u -> (t u)", t=len(unit_ids))

    unit_index = np.arange(len(unit_ids))
    unit_index = repeat(unit_index, "u -> (u t)", t=2)

    timestamps = np.array([start, end], dtype=np.float64)
    timestamps = repeat(timestamps, "u -> (t u)", t=len(unit_ids))
    return token_type_index, unit_index, timestamps


def create_linspace_latent_tokens(start, end, step, num_latents_per_step):
    r"""Creates a sequence of latent tokens. Each token is defined by the
    latent index and the timestamps. The sequence is defined by the start and end
    time and the step size. The group of `num_latents_per_step` latents is repeated
    for each step.

    Args:
        start (float): The start time of the sequence.
        end (float): The end time of the sequence.
        step (float): The step size.
        num_latents_per_step (int): The number of latents per step.
    """
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    num_timestamps = len(latent_timestamps)
    latent_timestamps = repeat(latent_timestamps, "t -> (t u)", u=len(latent_index))

    latent_index = repeat(latent_index, "u -> (t u)", t=num_timestamps)
    return latent_index, latent_timestamps
