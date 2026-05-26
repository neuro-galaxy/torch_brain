import os
import logging
import random
import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    if seed is not None:
        log.info(f"Global seed: {seed}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
