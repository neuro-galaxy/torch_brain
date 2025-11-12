from abc import ABC, abstractmethod
from typing import Callable, Literal
from pathlib import Path
import logging
import yaml
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from temporaldata import Data

from torch_brain.registry import ModalitySpec, register_modality, MODALITY_REGISTRY
from torch_brain.schemas.base_class.dataset_schema import BaseDatasetConfig
from torch_brain.data import Dataset, collate


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
        dir_path: str,
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
            root=dir_path,
            config=dataset_config,
            split="train",
        )

        # Validation
        self.val_dataset = Dataset(
            root=dir_path,
            config=dataset_config,
            split="valid",
        )

        # Testing
        self.test_dataset = Dataset(
            root=dir_path,
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

    @classmethod
    def create_basic_dataset_config(
        cls,
        dir_path: str,
        brainset: str,
        sessions: list[str],
        readout_id: str,
        value_key: str,
        timestamp_key: str,
    ) -> list[dict]:
        """Create a minimal dataset configuration dictionary for the given brainset and sessions."""

        def deep_getattr(obj, attr_path):
            """Get a nested attribute from an object using a dotted path."""
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                raise AttributeError(
                    f"Attribute path '{attr_path}' not found in the object."
                )

        # Estimate z-score statistics from training data
        session = sessions[0]
        with h5py.File(f"{dir_path}/{brainset}/{session}.h5", "r") as f:
            session_data = Data.from_hdf5(f, lazy=True)
            # Validate value_key and timestamp_key exist in the data
            _ = deep_getattr(session_data, value_key)
            _ = deep_getattr(session_data, timestamp_key)
            # Select training data based on the training domain interval
            train_data = session_data.select_by_interval(session_data.train_domain)
            train_vals = deep_getattr(train_data, value_key)
            mean_val = np.mean(train_vals)
            std_val = np.std(train_vals)

        # Dataset dictionary
        dataset_dict = [
            {
                "selection": [
                    {
                        "brainset": brainset,
                        "sessions": sessions,
                    }
                ],
                "config": {
                    "readout": {
                        "readout_id": readout_id,
                        "normalize_mean": mean_val,
                        "normalize_std": std_val,
                        "timestamp_key": timestamp_key,
                        "value_key": value_key,
                        "weights": {},
                        "eval_interval": None,
                        "metrics": [{"metric": {"_target_": "torchmetrics.R2Score"}}],
                    }
                },
            }
        ]

        # Instantiate using model_validate
        return BaseDatasetConfig.model_validate(dataset_dict).model_dump()

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
