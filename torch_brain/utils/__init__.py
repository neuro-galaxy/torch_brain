import torch

from .tokenizers import create_linspace_latent_tokens, create_start_end_unit_tokens
from .sinusoidal_encoding import get_sinusoidal_encoding

from .seed_everything import seed_everything


def move_to(data: dict, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
        elif isinstance(data[key], dict):
            data[key] = move_to(data[key], device)
    return data
