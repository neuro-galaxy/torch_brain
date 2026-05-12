import os
import logging
import random
import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Sets random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    if seed is not None:
        log.info(f"Global seed: {seed}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, (str, int, float, bool, type(None), np.ndarray)):
        # Metadata/scalars do not need device transfer.
        return data
    else:
        raise TypeError(f"Unknown data type {type(data)}")
