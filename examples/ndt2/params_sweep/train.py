import pickle

old_unpickler = pickle.Unpickler  # Unfortunate hack to fix a bug in Lightning.
# https://github.com/Lightning-AI/lightning/issues/18152
# Will likely be fixed by 2.1.0.
pickle.Unpickler = old_unpickler
import hydra
from omegaconf import DictConfig
import sys

sys.path.insert(0, "..")
# so that we pick the `run_training` from the main `train.py` script
from train import run_training

from utils import get_sweep_run_name


# This loads the config file using Hydra, similar to Flags, but composable.
@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # If sweep is enabled, dynamically name the run using the helper
    if cfg.get("sweep", True):
        cfg.wandb.run_name = get_sweep_run_name(cfg)
    # Rest of the training is exactly identical to the original train.py script.
    run_training(cfg)


if __name__ == "__main__":
    main()
