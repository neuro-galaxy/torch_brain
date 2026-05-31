"""Deprecated. Use torch_brain.datasets instead """

from torch_brain.datasets import (
    Dataset,
    DatasetIndex,
    NestedDataset,
    NestedSpikingDataset,
    SpikingDatasetMixin,
    CalciumImagingDatasetMixin,
    MultiChannelDatasetMixin,
)
import warnings

warnings.warn(
    "All components in moduel torch_brain.dataset have been moved to "
    "torch_brain.datasets. torch_brain.dataset is being kept for "
    "backwards compatibility for some time, and will be removed soon. "
    "Please use `torch_brain.datasets`",
    DeprecationWarning,
    stacklevel=2,
)
