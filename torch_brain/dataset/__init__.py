"""
Module Overview
---------------

This module contains base classes to ease creation of PyTorch datasets for your data.

- The :class:`Dataset` class is inherited by all datasets. These handle opening and accessing single datasets.
- The :class:`NestedDataset` class is for opening and accessing multiple datasets through a unified interface.
- :ref:`Mixin <Mixins>` classes are provided to add modality-specific functionalities to the Dataset classes.


Dataset
^^^^^^^

torch_brain's :class:`Dataset` class (and its sub-classes) allow you to sample *time-slices* of your data.
This is a major deviation from the standard :class:`torch.utils.data.Dataset`, which is indexed by integers.
To achieve arbitrary time-slice based access, our Dataset class is indexed by three things:

1. The recording id from which you want the slice,
2. Start time of the slice, and
3. End time of the slice

These are put into a :class:`DatasetIndex` object, which is then used to index the :class:`Dataset`.
Since different machine learning applications require different ways of sampling, we provide a collection of
:doc:`samplers <data>` which are responsible for creating these :class:`DatasetIndex` objects.

NestedDataset
^^^^^^^^^^^^^

The :class:`Dataset` class is designed to operate on a single dataset. However, many modern ML methods perform
training over multiple datasets. For this, we provide :class:`NestedDataset` that allows users to open and index through
multple datasets.
"""

from .dataset import Dataset, DatasetIndex
from .nested import NestedDataset, NestedSpikingDataset
from .mixins import (
    SpikingDatasetMixin,
    CalciumImagingDatasetMixin,
    MultiChannelDatasetMixin,
)

__all__ = [
    "Dataset",
    "DatasetIndex",
    "NestedDataset",
    "NestedSpikingDataset",
    "SpikingDatasetMixin",
    "CalciumImagingDatasetMixin",
    "MultiChannelDatasetMixin",
]
# _dataset_classes = ["Dataset", "DatasetIndex"]
# _nested_classes = ["NestedDataset", "NestedSpikingDataset"]
# _mixin_classes = [
#     "SpikingDatasetMixin",
#     "CalciumImagingDatasetMixin",
#     "MultiChannelDatasetMixin",
# ]
