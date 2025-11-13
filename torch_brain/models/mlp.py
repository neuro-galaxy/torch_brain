from pathlib import Path
from typing import Callable
import numpy as np
import logging
import torch
import torch.nn as nn
from temporaldata import Data


from torch_brain.models.base_class import TorchBrainModel
from torch_brain.registry import ModalitySpec
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.data import collate
from torch_brain.utils.training import train_model


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
        input_bin_size: int,
        output_bin_size: int,
        hidden_dim: int,
    ):
        """Initialize the neural net layers."""
        super().__init__(readout_spec=readout_spec)

        self.sequence_length = sequence_length

        self.input_bin_size = input_bin_size
        self.input_num_timesteps = int(sequence_length / input_bin_size)
        self.input_dim = num_units * self.input_num_timesteps

        self.output_bin_size = output_bin_size
        self.output_num_timesteps = int(sequence_length / output_bin_size)
        self.output_dim = readout_spec.dim * self.output_num_timesteps

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
        y = self.net(x)  # (B, T*N)     -> (B, T*D_out)
        y = y.reshape(
            -1, self.output_num_timesteps, self.readout_spec.dim
        )  # (B, T*D_out) -> (B, T, D_out)
        return y

    def tokenize(self, data: Data) -> dict:
        """tokenizes a data sample, which is a sliced Data object"""
        # Extract and bin neural activity (data.spikes)
        # Final shape of x here is (timestamps, num_neurons)
        spikes = data.spikes
        x = bin_spikes(
            spikes=spikes,
            num_units=len(data.units),
            bin_size=self.input_bin_size,
            num_bins=self.input_num_timesteps,
        ).T

        # Target variable
        y = data.get_nested_attribute(self.readout_spec.value_key)

        # Handle length mismatches by cropping or padding
        if len(y) > self.output_num_timesteps:
            # Too many samples - drop the last ones
            y = y[: self.output_num_timesteps]
        elif len(y) < self.output_num_timesteps:
            # Too few samples - pad by repeating the last value
            padding_needed = self.output_num_timesteps - len(y)
            y = np.concatenate([y, np.repeat(y[-1], padding_needed)])

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

    def train_model(
        self,
        device: torch.device | None = None,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict | None = None,
        num_epochs: int = 50,
        data_loader_batch_size: int = 16,
        data_loader_collate_fn: Callable | None = collate,
        data_loader_num_workers: int = 0,
        data_loader_pin_memory: bool = False,
        data_loader_persistent_workers: bool = False,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                else:
                    device = torch.device("cpu")
        self.to(device)
        logging.info(f"Training on device: {device}")

        # Optimizer setup
        if optimizer_kwargs is None:
            optimizer_kwargs = dict(
                lr=1e-3,
            )
        optimizer = optimizer_class(
            self.parameters(),
            **optimizer_kwargs,
        )

        # Data loaders
        if device.type == "mps":
            data_loader_num_workers = 0
            data_loader_pin_memory = False
            data_loader_persistent_workers = False

        data_loader_kwargs = dict(
            batch_size=data_loader_batch_size,
            collate_fn=data_loader_collate_fn,
            num_workers=data_loader_num_workers,
            pin_memory=data_loader_pin_memory,
            persistent_workers=data_loader_persistent_workers,
        )

        train_loader = self.get_data_loader(mode="train", **data_loader_kwargs)
        val_loader = self.get_data_loader(mode="valid", **data_loader_kwargs)

        r2_log, loss_log, train_outputs = train_model(
            device=device,
            model=self,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            store_params=None,
        )

        return (r2_log, loss_log, train_outputs)
