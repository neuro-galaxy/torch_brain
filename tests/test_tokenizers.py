import numpy as np

from torch_brain.utils import (
    create_linspace_latent_tokens,
    create_start_end_unit_tokens,
)


def test_create_linspace_latent_tokens():
    start = 0
    end = 1
    step = 0.5
    num_latents_per_step = 2
    latent_index, latent_timestamps = create_linspace_latent_tokens(
        start, end, step, num_latents_per_step
    )
    assert latent_index.shape == (4,)
    assert latent_timestamps.shape == (4,)

    assert np.allclose(latent_index, [0, 1, 0, 1])
    assert np.allclose(latent_timestamps, [0.25, 0.25, 0.75, 0.75])

    start = 0
    end = 10
    step = 1
    num_latents_per_step = 5
    latent_index, latent_timestamps = create_linspace_latent_tokens(
        start, end, step, num_latents_per_step
    )

    assert latent_index.shape == (50,)
    assert latent_timestamps.shape == (50,)


def test_create_start_end_unit_tokens():
    unit_name = np.array(["a", "b", "c"])
    start = 0
    end = 1
    token_type_index, unit_index, timestamps = create_start_end_unit_tokens(
        unit_name, start, end
    )

    assert token_type_index.shape == (6,)
    assert unit_index.shape == (6,)
    assert timestamps.shape == (6,)

    assert np.allclose(token_type_index, [1, 2, 1, 2, 1, 2])
    assert np.allclose(unit_index, [0, 0, 1, 1, 2, 2])
    assert np.allclose(timestamps, [0, 1, 0, 1, 0, 1])
