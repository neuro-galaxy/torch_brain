import os
import random
import logging

import torch
import numpy as np


log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    r"""Set random seeds across all libraries for reproducibility.

    Seeds PyTorch, CUDA, NumPy, and Python's random module. Also sets
    ``PYTHONHASHSEED`` environment variable and configures cuDNN for
    deterministic behavior.

    Args:
        seed: The random seed to use. If None, no seeding is performed.
    """
    if seed is not None:
        log.info("Global seed set to {}.".format(seed))

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
