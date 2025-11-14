from typing import List
from pathlib import Path
import argparse

from omegaconf import DictConfig
import hydra

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange
from hydra import compose, initialize

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

from torch_brain.registry import MODALITY_REGISTRY

class FalconTorchBrainWrapper(BCIDecoder):
    r"""
        For the FALCON challenge
    """

    def __init__(
            self,
            model: nn.Module,
            task_config: FalconConfig,
            max_bins: int,
            batch_size: int = 1,
        ):
        super().__init__(task_config=task_config, batch_size=batch_size)
        pl.seed_everything(seed=0)

        self.model = model.to('cuda:0')
        self.model.eval()

        self.observation_buffer = torch.zeros((
            max_bins,
            self.batch_size,
            task_config.n_channels
        ), dtype=torch.uint8, device='cuda:0')

    def set_batch_size(self, batch_size: int):
        super().set_batch_size(batch_size)
        self.observation_buffer = torch.zeros((
            self.observation_buffer.shape[0],
            batch_size,
            self.observation_buffer.shape[2]
        ), dtype=torch.uint8, device='cuda:0')

    def reset(self, dataset_tags: List[Path] = [""]):
        self.set_steps = 0
        self.observation_buffer.zero_()
        self.cur_batch = len(dataset_tags)

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts

            return:
                out: (batch, n_dims)
        """
        self.observe(neural_observations)
        decoder_in = rearrange(self.observation_buffer[-self.set_steps:], 't b c -> b t c')
        batch_in = decoder_in[:self.cur_batch]
        batch_in_tokenized = self.model.tokenize(batch_in)
        decoder_index = torch.tensor([[0]], device=self.model.device)
        out = self.model(x=batch_in_tokenized, output_decoder_index=decoder_index)
        return out.cpu().numpy()

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        if neural_observations.shape[0] < self.batch_size:
            neural_observations = np.pad(neural_observations, ((0, self.batch_size - neural_observations.shape[0]), (0, 0)))
        self.set_steps += 1
        self.observation_buffer = torch.roll(self.observation_buffer, -1, dims=0)
        self.observation_buffer[-1] = torch.as_tensor(neural_observations, dtype=torch.uint8, device='cuda:0')

    def on_done(self, dones: np.ndarray):
        r"""
        """
        if dones.any():
            self.set_steps = 0
        if dones.shape[0] < self.batch_size:
            dones = np.pad(dones, (0, self.batch_size - dones.shape[0]))
        self.observation_buffer[:, dones].zero_()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--root-config-path", type=str, default='train_falcon_m2_rnn',
        help="Path to root torch brain config used in model preparation."
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1
    )

    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
        dataloader_workers=8,
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)

    # History settings matched to https://github.com/snel-repo/falcon-challenge/blob/main/decoder_demos/ndt2_sample.py
    max_bins = 50 if task in [FalconTask.m1, FalconTask.m2] else 200

    try:
        initialize(
            config_path="./configs",
            version_base="1.3",
        )
    except:
        print('Hydra Initialize failed, assuming this is not the first decoder and config is otherwise available..')
    cfg: DictConfig = compose(config_name=args.root_config_path)

    if args.split == 'm1':
        readout = 'arm_emg_16d'
    elif args.split == 'm2':
        readout = 'arm_velocity_2d'
    elif args.split == 'h1':
        readout = 'arm_velocity_7d'
    else:
        raise ValueError(f"Invalid split: {args.split}")
    readout_spec = MODALITY_REGISTRY[readout]
    model = hydra.utils.instantiate(cfg.model, readout_specs={readout: readout_spec})
    if args.model_path != "DUMMY": # temp
        model.load_state_dict(torch.load(args.model_path)['state_dict'])

    decoder = FalconTorchBrainWrapper(
        model=model,
        task_config=config,
        max_bins=max_bins,
        batch_size=args.batch_size,
    )

    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()