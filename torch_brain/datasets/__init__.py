__all__ = [
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
            "template": "dataset.rst",
            "autosummary": [
                "OpenNeuroDataset",
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
