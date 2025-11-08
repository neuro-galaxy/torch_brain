"""
Pydantic schemas for POYO dataset configurations.

This module provides type-safe validation for dataset configurations
used with POYO models. The schemas ensure configs are valid before
being passed to the Dataset and model tokenizers.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator


class ReadoutConfig(BaseModel):
    """Configuration for POYO readout (single-task)."""
    
    # Required fields
    readout_id: Union[str, int] = Field(
        ...,
        description="Modality identifier from MODALITY_REGISTRY"
    )
    
    # Normalization
    normalize_mean: Optional[float] = Field(
        None,
        description="Mean for z-score normalization."
    )
    normalize_std: Optional[float] = Field(
        None,
        description="Standard deviation for z-score normalization. Must be non-zero."
    )
    
    # Data location
    timestamp_key: Optional[str] = Field(
        None,
        description="Timestamp location (e.g., 'hand.timestamps')",
        examples=["cursor.timestamps", "hand.timestamps", "wheel.timestamps"]
    )
    value_key: Optional[str] = Field(
        None,
        description="Value location (e.g., 'cursor.vel')",
        examples=["cursor.vel", "hand.vel", "wheel.vel"]
    )
    
    # Loss weighting
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Mapping of interval paths to weight multipliers for loss computation"
    )
    
    # Evaluation
    eval_interval: Optional[str] = Field(
        None,
        description="Interval path for filtering evaluation periods",
        examples=["movement_phases.reach_period", "movement_phases.random_period"]
    )
    
    metrics: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of metric configurations for evaluation"
    )
    
    model_config = ConfigDict(extra="forbid")
        
    @field_validator('normalize_std')
    @classmethod
    def validate_std_not_zero(cls, v):
        """Ensure normalization std is not zero."""
        if v is not None:
            if v == 0:
                raise ValueError("normalize_std cannot be 0")
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Ensure all weight values are numeric."""
        if v is not None:
            for key, value in v.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Weight value for interval '{key}' must be numeric, got {type(value).__name__}"
                    )
        return v


class SelectionConfig(BaseModel):
    """Configuration for dataset selection criteria."""
    
    brainset: str = Field(
        ...,
        description="Brainset identifier (e.g., 'perich_miller_population_2018')"
    )
    
    sessions: Optional[List[str]] = Field(
        None,
        description="List of sessions to include"
    )
    
    model_config = ConfigDict(extra="forbid")


class DatasetLevelConfig(BaseModel):
    """Configuration options at the dataset level."""
    
    readout: ReadoutConfig = Field(
        ...,
        description="Readout configuration for this dataset selection"
    )
    
    sampling_intervals_modifier: Optional[str] = Field(
        None,
        description="Python code to modify sampling intervals (advanced usage)"
    )
    
    model_config = ConfigDict(extra="allow")


class DatasetSelectionBlock(BaseModel):
    """A single selection block with associated configuration."""
    
    selection: List[SelectionConfig] = Field(
        ...,
        description="List of selection criteria (can select from multiple brainsets)"
    )
    
    config: DatasetLevelConfig = Field(
        ...,
        description="Configuration for this selection"
    )
    
    model_config = ConfigDict(extra="forbid")


class POYODatasetConfig(RootModel):
    """
    Complete POYO Dataset configuration schema.
    
    This represents the full dataset YAML config structure,
    which can contain multiple selection blocks, each with
    their own readout configuration.
    
    Example:
        config = POYODatasetConfig.model_validate(yaml_data)
    """
    
    root: List[DatasetSelectionBlock]
    
    def __iter__(self):
        """Allow iterating over selection blocks."""
        return iter(self.root)
    
    def __getitem__(self, item):
        """Allow indexing selection blocks."""
        return self.root[item]
    
    def __len__(self):
        """Get number of selection blocks."""
        return len(self.root)


# Export main classes
__all__ = [
    'ReadoutConfig',
    'SelectionConfig',
    'DatasetLevelConfig',
    'DatasetSelectionBlock',
    'POYODatasetConfig',
]
