from abc import ABC, abstractmethod
from pathlib import Path
import logging
import yaml
import torch
import torch.nn as nn

from torch_brain.registry import ModalitySpec, register_modality, MODALITY_REGISTRY
from torch_brain.schemas.base_class.dataset_schema import BaseDatasetConfig
from torch_brain.data import Dataset
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)


logger = logging.getLogger(__name__)


class TorchBrainModel(nn.Module, ABC):
    """Base class for all TorchBrain models."""

    def __init__(
        self,
        *,
        readout_spec: ModalitySpec,
        seed: int = 42,
        **torch_kwargs,
    ):
        super().__init__(**torch_kwargs)
        self.seed = seed
        self.validate_and_register_modality(readout_spec)

    def validate_and_register_modality(self, readout_spec: ModalitySpec):
        """Validate and register the model's readout modality."""
        modality_name = readout_spec.name
        if modality_name not in MODALITY_REGISTRY:
            logger.info(f"Registering new modality: {modality_name}")
            register_modality(
                name=modality_name,
                dim=readout_spec.dim,
                type=readout_spec.type,
                timestamp_key=readout_spec.timestamp_key,
                value_key=readout_spec.value_key,
                loss_fn=readout_spec.loss_fn,
            )
        logger.info(f"Set modality: {modality_name}")
        self.readout_spec = readout_spec

    @classmethod
    def get_dataset_config_schema(cls) -> type[BaseDatasetConfig]:
        """Return the dataset configuration schema class for this model."""
        return BaseDatasetConfig

    def validate_dataset_config(self, dataset_config: dict):
        """Validate a dataset configuration against the model's schema and readout_spec."""
        schema = self.get_dataset_config_schema()
        schema(**dataset_config)

        # Validate dataset config against model readout_spec used to initialize the model
        readout_id = dataset_config["readout"]["readout_id"]
        readout_value_key = dataset_config["readout"]["value_key"]
        readout_timestamp_key = dataset_config["readout"]["timestamp_key"]
        if readout_id != self.readout_spec.name:
            raise ValueError(
                f"Dataset config readout_id '{readout_id}' does not match model readout_spec '{self.readout_spec.name}'"
            )
        if readout_value_key != self.readout_spec.value_key:
            raise ValueError(
                f"Dataset config readout value_key '{readout_value_key}' does not match model readout_spec '{self.readout_spec.value_key}'"
            )
        if readout_timestamp_key != self.readout_spec.timestamp_key:
            raise ValueError(
                f"Dataset config readout timestamp_key '{readout_timestamp_key}' does not match model readout_spec '{self.readout_spec.timestamp_key}'"
            )

    def set_datasets(
        self,
        brainset_path: str,
        dataset_config: str | Path | dict,
    ):
        """Set datasets (train, valid, test) for this model based on the provided configuration."""
        # Load from yaml file if a path is provided
        if isinstance(dataset_config, (str, Path)):
            with open(dataset_config, "r") as f:
                dataset_config = yaml.safe_load(f)

        self.validate_dataset_config(dataset_config)

        # Training
        self.train_dataset = Dataset(
            root=brainset_path,
            config=dataset_config,
            split="train",
        )

        # Validation
        self.val_dataset = Dataset(
            root=brainset_path,
            config=dataset_config,
            split="valid",
        )

        # Testing
        self.test_dataset = Dataset(
            root=brainset_path,
            config=dataset_config,
            split="test",
        )

    def get_train_data_sampler(self) -> torch.utils.data.Sampler:
        train_sampling_intervals = self.train_dataset.get_sampling_intervals()
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=train_sampling_intervals,
            window_length=1.0,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_sampler

    def get_val_data_sampler(self) -> torch.utils.data.Sampler:
        val_sampling_intervals = self.val_dataset.get_sampling_intervals()
        val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=val_sampling_intervals,
            window_length=1.0,
        )
        return val_sampler

    def get_test_data_sampler(self) -> torch.utils.data.Sampler:
        test_sampling_intervals = self.test_dataset.get_sampling_intervals()
        test_sampler = SequentialFixedWindowSampler(
            sampling_intervals=test_sampling_intervals,
            window_length=1.0,
        )
        return test_sampler

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method")

    @abstractmethod
    @classmethod
    def load_pretrained(cls) -> "TorchBrainModel":
        """Load a pretrained model from a checkpoint."""
        raise NotImplementedError(
            "Subclasses must implement the load_pretrained method"
        )
