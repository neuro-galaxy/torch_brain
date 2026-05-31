"""Base classes for creating PyTorch datasets

Module Overview
---------------

This module contains base classes to ease creation of PyTorch datasets for your data.

- The :class:`Dataset` class is inherited by all datasets. These handle opening and accessing single datasets.
- The :class:`NestedDataset` class is for opening and accessing multiple datasets through a unified interface.
- :ref:`Mixin <datasets_ref-mixins>` classes are provided to add modality-specific functionalities to the Dataset classes.


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
:ref:`samplers <samplers_ref>` which are responsible for creating these :class:`DatasetIndex` objects.

NestedDataset
^^^^^^^^^^^^^

The :class:`Dataset` class is designed to operate on a single dataset. However, many modern ML methods perform
training over multiple datasets. For this, we provide :class:`NestedDataset` that allows users to open and index through
multiple datasets.
"""

__all__ = [
    "Dataset",
    "DatasetIndex",
    "NestedDataset",
    "NestedSpikingDataset",
    "SpikingDatasetMixin",
    "CalciumImagingDatasetMixin",
    "MultiChannelDatasetMixin",
    "PerichMillerPopulation2018",
    "PeiPandarinathNLB2021",
    "FlintSlutzkyAccurate2012",
    "ChurchlandShenoyNeural2012",
    "OdohertySabesNonhuman2017",
    "VollanMoserAlternating2025",
    "ShiraziHBNR1DS005505",
    "KlinzingSleepDS005555",
    "AllenVisualCodingOphys2016",
    "Neuroprobe2025",
    "KochiVisualNamingDS006914",
    "KempSleepEDF2013",
    "OpenNeuroDataset",
    "OpenNeuroSplitType",
]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": None,
    "sections": [],
}

from .dataset import Dataset, DatasetIndex
from .nested import NestedDataset, NestedSpikingDataset
from .mixins import (
    SpikingDatasetMixin,
    CalciumImagingDatasetMixin,
    MultiChannelDatasetMixin,
)
from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType

from .PerichMillerPopulation2018 import PerichMillerPopulation2018
from .PeiPandarinathNLB2021 import PeiPandarinathNLB2021
from .FlintSlutzkyAccurate2012 import FlintSlutzkyAccurate2012
from .ChurchlandShenoyNeural2012 import ChurchlandShenoyNeural2012
from .OdohertySabesNonhuman2017 import OdohertySabesNonhuman2017
from .AllenVisualCodingOphys2016 import AllenVisualCodingOphys2016
from .KempSleepEDF2013 import KempSleepEDF2013
from .Neuroprobe2025 import Neuroprobe2025
from .KlinzingSleepDS005555 import KlinzingSleepDS005555
from .KochiVisualNamingDS006914 import KochiVisualNamingDS006914
from .ShiraziHBNR1DS005505 import ShiraziHBNR1DS005505
from .VollanMoserAlternating2025 import VollanMoserAlternating2025

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Base Classes",
            "description": None,
            "autosummary": [
                "Dataset",
                "DatasetIndex",
                "NestedDataset",
                "OpenNeuroDataset",
            ],
        },
        {
            "title": "Mixins",
            "description": None,
            "autosummary": [
                "SpikingDatasetMixin",
                "CalciumImagingDatasetMixin",
                "MultiChannelDatasetMixin",
            ],
        },
        {
            "title": "Electrophysiology Datasets",
            "template": "dataset.rst",
            "autosummary": [
                "PerichMillerPopulation2018",
                "PeiPandarinathNLB2021",
                "FlintSlutzkyAccurate2012",
                "ChurchlandShenoyNeural2012",
                "OdohertySabesNonhuman2017",
                "VollanMoserAlternating2025",
                "ShiraziHBNR1DS005505",
            ],
        },
        {
            "title": "Calcium Imaging Datasets",
            "template": "dataset.rst",
            "autosummary": [
                "AllenVisualCodingOphys2016",
            ],
        },
        {
            "title": "iEEG Datasets",
            "template": "dataset.rst",
            "autosummary": [
                "Neuroprobe2025",
                "KochiVisualNamingDS006914",
            ],
        },
        {
            "title": "EEG Datasets",
            "template": "dataset.rst",
            "autosummary": [
                "KlinzingSleepDS005555",
            ],
        },
        {
            "title": "PSG Datasets",
            "template": "dataset.rst",
            "autosummary": [
                "KempSleepEDF2013",
            ],
        },
    ],
}
