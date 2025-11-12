from pathlib import Path
import torch
import torch.nn as nn
from temporaldata import Data
import numpy as np
import logging

from torch_brain.models.base_class import TorchBrainModel
from torch_brain.registry import ModalitySpec
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)


def bin_spikes(spikes, num_units, bin_size, num_bins=None):
    """
    Bins spike timestamps into a 2D array: [num_units x num_bins].
    """
    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)
    return binned_spikes


class MLPNeuralDecoder(TorchBrainModel):
    def __init__(
        self,
        readout_spec: ModalitySpec,
        sequence_length: float,
        num_units: int,
        bin_size: int,
        hidden_dim: int,
    ):
        """Initialize the neural net layers."""
        super().__init__(readout_spec=readout_spec)

        self.sequence_length = sequence_length
        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size
        self.input_dim = num_units * self.num_timesteps
        self.output_dim = readout_spec.dim * self.num_timesteps

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produces predictions from a binned spiketrain.
        This is pure PyTorch code.

        Shape of x: (B, T, N)
        """
        x = x.flatten(1)  # (B, T, N)    -> (B, T*N)
        x = self.net(x)  # (B, T*N)     -> (B, T*D_out)
        x = x.reshape(-1, self.num_timesteps, 2)  # (B, T*D_out) -> (B, T, D_out)
        return x

    def tokenize(self, data: Data) -> dict:
        """tokenizes a data sample, which is a sliced Data object"""
        # Extract and bin neural activity (data.spikes)
        # Final shape of x here is (timestamps, num_neurons)
        spikes = data.spikes
        x = bin_spikes(
            spikes=spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            num_bins=self.num_timesteps,
        ).T

        # Target variable
        y = getattr(data, self.readout_spec.value_key)

        # Output the "tokenized" data in the form of a dictionary
        data_dict = {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }
        return data_dict

    def set_datasets(self, dir_path: str, dataset_config: str | Path | list[dict]):
        super().set_datasets(dir_path, dataset_config)

        # Connect tokenizers to Datasets
        self.train_dataset.transform = self.tokenize
        self.val_dataset.transform = self.tokenize
        self.test_dataset.transform = self.tokenize

    def get_train_data_sampler(self) -> torch.utils.data.Sampler:
        return RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.seed),
            drop_short=True,
        )

    def get_val_data_sampler(self) -> torch.utils.data.Sampler:
        return SequentialFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=None,
            drop_short=False,
        )

    def get_test_data_sampler(self) -> torch.utils.data.Sampler:
        return SequentialFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=None,
            drop_short=False,
        )

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str | Path,
        readout_spec: ModalitySpec,
        skip_readout: bool = False,
    ) -> "MLPNeuralDecoder":
        """Load a pretrained MLPNeuralDecoder model from a checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_args = checkpoint["model_args"]
        model = cls(readout_spec=readout_spec, **model_args)

        # Load model weights
        # If model is pretrained using lightning, model weights are prefixed with "model."
        state_dict = {
            k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
        }

        # Remove readout layer from checkpoint if we're using a new one
        if skip_readout:
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("net.4.")
            }

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            logging.warning(
                f"Missing keys when loading pretrained MLPNeuralDecoder: {missing_keys}"
            )
        if len(unexpected_keys) > 0:
            logging.warning(
                f"Unexpected keys when loading pretrained MLPNeuralDecoder: {unexpected_keys}"
            )
        return model
