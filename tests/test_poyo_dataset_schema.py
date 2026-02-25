"""Tests for POYO dataset configuration schemas."""

import pytest
from pathlib import Path
from omegaconf import OmegaConf
from torch_brain.schemas.poyo_dataset import POYODatasetConfig


# List of all dataset config files to test
DATASET_CONFIG_FILES = [
    "examples/poyo/configs/dataset/pei_pandarinath_nlb_2021.yaml",
    "examples/poyo/configs/dataset/perich_miller_population_2018.yaml",
    "examples/poyo/configs/dataset/poyo_1.yaml",
]


@pytest.mark.parametrize("config_path", DATASET_CONFIG_FILES)
def test_validate_real_dataset_configs(config_path):
    """Test that all real POYO dataset configs validate successfully."""
    # Load config
    cfg = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Validate - should not raise ValidationError
    validated = POYODatasetConfig.model_validate(config_dict)

    # Basic assertions
    assert len(validated) > 0, "Config should have at least one selection block"

    # Check first block has required structure
    first_block = validated[0]
    assert hasattr(first_block, "selection"), "Block should have selection"
    assert hasattr(first_block, "config"), "Block should have config"
    assert hasattr(first_block.config, "readout"), "Config should have readout"

    # Check readout has required field
    readout = first_block.config.readout
    assert hasattr(readout, "readout_id"), "Readout should have readout_id"
    assert readout.readout_id is not None, "Readout ID should not be None"


def test_schema_rejects_invalid_config():
    """Test that schema rejects invalid configurations."""
    invalid_config = [
        {
            "selection": [{"brainset": "test"}],
            "config": {
                "readout": {
                    "readout_id": "test",
                    "normalize_std": 0.0,  # Invalid: std cannot be 0
                }
            },
        }
    ]

    with pytest.raises(ValueError, match="cannot be 0"):
        POYODatasetConfig.model_validate(invalid_config)


def test_schema_rejects_unknown_fields():
    """Test that schema rejects configs with unknown fields."""
    invalid_config = [
        {
            "selection": [{"brainset": "test"}],
            "config": {
                "readout": {
                    "readout_id": "test",
                    "unknown_field": "value",  # Invalid: unknown field
                }
            },
        }
    ]

    with pytest.raises(Exception):  # Pydantic ValidationError
        POYODatasetConfig.model_validate(invalid_config)
