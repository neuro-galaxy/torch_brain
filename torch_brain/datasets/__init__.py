"""Module containing base classes for creating PyTorch-compatible Datasets,
as well as a library of pre-written Dataset classes for a number of brainsets.
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

from .AllenVisualCodingOphys2016 import AllenVisualCodingOphys2016
from .ChurchlandShenoyNeural2012 import ChurchlandShenoyNeural2012
from .dataset import Dataset, DatasetIndex
from .KempSleepEDF2013 import KempSleepEDF2013
from .KlinzingSleepDS005555 import KlinzingSleepDS005555
from .KochiVisualNamingDS006914 import KochiVisualNamingDS006914
from .mixins import (
    CalciumImagingDatasetMixin,
    MultiChannelDatasetMixin,
    SpikingDatasetMixin,
)
from .nested import NestedDataset, NestedSpikingDataset
from .Neuroprobe2025 import Neuroprobe2025
from .OdohertySabesNonhuman2017 import OdohertySabesNonhuman2017
from .OpenNeuroDataset import OpenNeuroDataset, OpenNeuroSplitType
from .PeiPandarinathNLB2021 import PeiPandarinathNLB2021
from .PerichMillerPopulation2018 import PerichMillerPopulation2018
from .ShiraziHBNR1DS005505 import ShiraziHBNR1DS005505
from .VollanMoserAlternating2025 import VollanMoserAlternating2025

_base_classes_doc = """
Base classes to ease creation of PyTorch-compatible Datasets for your data.

- The :class:`Dataset` class is inherited by all datasets. These handle opening
  and accessing single datasets.
- The :class:`NestedDataset` class is for opening and accessing multiple
  datasets through a unified interface.
- :ref:`Mixin <datasets_ref-mixins>` classes are provided to add
  modality-specific functionalities to the Dataset classes.


**Dataset:** torch_brain's :class:`Dataset` class (and its sub-classes) allow
you to sample *time-slices* of your data. This is a major deviation from the
standard :class:`torch.utils.data.Dataset`, which is indexed by integers. To
achieve arbitrary time-slice based access, our Dataset class is indexed by
a :class:`DatasetIndex` containing three attributes:

.. code-block:: python

    DatasetIndex(
        recording_id=...,  # The recording ID from which we want the slice
        start=...,         # Start time of the slice
        end=...,           # End time of the slice
    )

Since different machine learning applications require different ways of
sampling, we provide a collection of :ref:`samplers <samplers_ref>` which are
responsible for creating these :class:`DatasetIndex` objects.

See
:doc:`NLB Maze minimal example </generated/notebooks/nlb_maze_minimal_example>`
for an example of how to create your own Dataset subclasses.

**NestedDataset:** The :class:`Dataset` class is designed to operate on a
single dataset. However, many modern ML methods perform training over multiple
datasets. For this, we provide :class:`NestedDataset` that allows users to open
and index through multiple datasets.
"""

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Base Classes & Mixins",
            "description": _base_classes_doc,
            "autosummary": [
                "Dataset",
                "DatasetIndex",
                "NestedDataset",
                "OpenNeuroDataset",
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
