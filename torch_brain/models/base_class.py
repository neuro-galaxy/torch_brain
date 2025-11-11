from abc import ABC, abstractmethod
from typing import Callable, Literal
from pathlib import Path
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_brain.registry import ModalitySpec, register_modality, MODALITY_REGISTRY
from torch_brain.schemas.base_class.dataset_schema import BaseDatasetConfig
from torch_brain.data import Dataset, collate
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

    def validate_dataset_config(self, dataset_config: list[dict]):
        """Validate a dataset configuration against the model's schema and readout_spec."""
        schema = self.get_dataset_config_schema()
        schema(root=dataset_config)

        # Validate dataset config against model readout_spec used to initialize the model
        selection_config = dataset_config[0]["config"]
        readout_id = selection_config["readout"]["readout_id"]
        readout_value_key = selection_config["readout"]["value_key"]
        readout_timestamp_key = selection_config["readout"]["timestamp_key"]
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
        dataset_config: str | Path | list[dict],
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

    def get_data_loader(
        self,
        *,
        mode: Literal["train", "valid", "test"],
        batch_size: int,
        collate_fn: Callable | None = collate,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **dataloader_kwargs,
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the given dataset and sampler."""
        if mode == "train":
            dataset = self.train_dataset
            sampler = self.get_train_data_sampler()
        elif mode == "valid":
            dataset = self.val_dataset
            sampler = self.get_val_data_sampler()
        elif mode == "test":
            dataset = self.test_dataset
            sampler = self.get_test_data_sampler()
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **dataloader_kwargs,
        )

    @abstractmethod
    def get_train_data_sampler(self):
        raise NotImplementedError(
            "Subclasses must implement the get_train_data_sampler method"
        )

    @abstractmethod
    def get_val_data_sampler(self):
        raise NotImplementedError(
            "Subclasses must implement the get_val_data_sampler method"
        )
    
    @abstractmethod
    def get_test_data_sampler(self):
        raise NotImplementedError(
            "Subclasses must implement the get_test_data_sampler method"
        )

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method")

    @classmethod
    @abstractmethod
    def load_pretrained(cls) -> "TorchBrainModel":
        """Load a pretrained model from a checkpoint."""
        raise NotImplementedError(
            "Subclasses must implement the load_pretrained method"
        )
