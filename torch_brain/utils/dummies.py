"""Dummy data generation for brainsets.

This module provides schema definitions and a generator function for creating
realistic dummy Data objects that match the structure of actual brainsets.
Everything is fully declarative - the generator uses only the schema definitions.
"""

import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)


# =============================================================================
# Schema Dataclasses
# =============================================================================


class TemporalType(Enum):
    """Types of temporal data structures from temporaldata."""

    IRREGULAR_TIME_SERIES = auto()
    REGULAR_TIME_SERIES = auto()
    ARRAY_DICT = auto()
    INTERVAL = auto()


@dataclass
class ArraySpec:
    """Specifies an array attribute's structure.

    Attributes:
        dtype: NumPy dtype string ("float64", "int64", "str", "bool").
        shape: List of dimension names (str) or fixed integers.
        distribution: Distribution for generating values:
            - "normal": params {"mean": float, "std": float}
            - "uniform": params {"low": float, "high": float}
            - "linspace": linearly spaced values scaled to time_span
            - "randint": params {"low": int, "high_dim": str} - high from dimension
            - "choice": params {"values": list}
            - "sequential_str": params {"prefix": str} - "prefix_0", "prefix_1", ...
            - "sequential_int": 0, 1, 2, ...
            - "bool_random": random True/False with bias
            - "interval_bounds": generates start/end for intervals
            - "interval_relative": params {"offset_frac": float} - relative to start
        params: Parameters for the distribution.
        is_interval_start: Mark this as the interval start field.
        is_interval_end: Mark this as the interval end field.
    """

    dtype: str = "float64"
    shape: List[Union[int, str]] = field(default_factory=list)
    distribution: str = "normal"
    params: Dict[str, Any] = field(default_factory=dict)
    is_interval_start: bool = False
    is_interval_end: bool = False


@dataclass
class TemporalObjectSpec:
    """Specifies a temporal data object.

    Attributes:
        type: The temporaldata type.
        attributes: Dict mapping attribute names to ArraySpec.
        length_dimension: Name of the dimension representing object length.
        sampling_rate: For RegularTimeSeries, the sampling rate in Hz.
    """

    type: TemporalType
    attributes: Dict[str, ArraySpec] = field(default_factory=dict)
    length_dimension: Optional[str] = None
    sampling_rate: Optional[float] = None


@dataclass
class BrainsetSpec:
    """Complete specification for a brainset's data structure.

    Attributes:
        brainset_id: Identifier matching the brainset pipeline name.
        description: Human-readable description of the brainset.
        dimensions: Dict of dimension name to default size.
        objects: Dict mapping attribute names to TemporalObjectSpec.
        metadata: Additional brainset-specific configuration.
    """

    brainset_id: str
    description: str = ""
    dimensions: Dict[str, int] = field(default_factory=dict)
    objects: Dict[str, TemporalObjectSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Brainset Specifications
# =============================================================================

BRAINSET_SPECS: Dict[str, BrainsetSpec] = {}

# -----------------------------------------------------------------------------
# O'Doherty-Sabes 2017: Self-paced reaching with Utah arrays
# -----------------------------------------------------------------------------
BRAINSET_SPECS["odoherty_sabes_nonhuman_2017"] = BrainsetSpec(
    brainset_id="odoherty_sabes_nonhuman_2017",
    description="Self-paced reaching task with Utah array recordings from M1 and S1",
    dimensions={
        "n_spikes": 50000,
        "n_units": 96,
        "n_behavior_samples": 2500,
        "waveform_samples": 48,
    },
    objects={
        "spikes": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_spikes",
            sampling_rate=5000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_spikes"], distribution="linspace"
                ),
                "unit_index": ArraySpec(
                    dtype="int64",
                    shape=["n_spikes"],
                    distribution="randint",
                    params={"low": 0, "high_dim": "n_units"},
                ),
                "waveforms": ArraySpec(
                    dtype="float32",
                    shape=["n_spikes", "waveform_samples"],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "unit_"},
                ),
                "type": ArraySpec(
                    dtype="int64",
                    shape=["n_units"],
                    distribution="choice",
                    params={"values": [1, 2]},
                ),
                "area_name": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="choice",
                    params={"values": ["M1", "S1"]},
                ),
            },
        ),
        "cursor": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=250.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
                "vel": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 100.0},
                ),
                "acc": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 500.0},
                ),
            },
        ),
        "finger": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=250.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos_3d": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 3],
                    distribution="normal",
                    params={"mean": 0.0, "std": 5.0},
                ),
                "vel_3d": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 3],
                    distribution="normal",
                    params={"mean": 0.0, "std": 10.0},
                ),
            },
        ),
    },
    metadata={
        "species": "MACACA_MULATTA",
        "task": "REACHING",
        "recording_tech": "UTAH_ARRAY_SPIKES",
    },
)

# -----------------------------------------------------------------------------
# Pei-Pandarinath NLB 2021: Delayed reaching through maze
# -----------------------------------------------------------------------------
BRAINSET_SPECS["pei_pandarinath_nlb_2021"] = BrainsetSpec(
    brainset_id="pei_pandarinath_nlb_2021",
    description="Delayed reaching task through maze with Utah array recordings",
    dimensions={
        "n_spikes": 50000,
        "n_units": 96,
        "n_behavior_samples": 2500,
        "n_trials": 5,
    },
    objects={
        "spikes": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_spikes",
            sampling_rate=5000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_spikes"], distribution="linspace"
                ),
                "unit_index": ArraySpec(
                    dtype="int64",
                    shape=["n_spikes"],
                    distribution="randint",
                    params={"low": 0, "high_dim": "n_units"},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "unit_"},
                ),
            },
        ),
        "hand": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=250.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 100.0},
                ),
                "vel": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 200.0},
                ),
            },
        ),
        "eye": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=250.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 10.0},
                ),
            },
        ),
        "trials": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_trials",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_end=True
                ),
                "go_cue_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.2},
                ),
                "move_onset_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.4},
                ),
            },
        ),
    },
    metadata={
        "species": "MACACA_MULATTA",
        "task": "REACHING",
        "recording_tech": "UTAH_ARRAY_SPIKES",
    },
)

# -----------------------------------------------------------------------------
# Churchland-Shenoy 2012: Center-out reaching
# -----------------------------------------------------------------------------
BRAINSET_SPECS["churchland_shenoy_neural_2012"] = BrainsetSpec(
    brainset_id="churchland_shenoy_neural_2012",
    description="Center-out reaching task with dual Utah array recordings from M1 and PMd",
    dimensions={
        "n_spikes": 50000,
        "n_units": 192,
        "n_behavior_samples": 10000,
        "n_trials": 5,
    },
    objects={
        "spikes": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_spikes",
            sampling_rate=5000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_spikes"], distribution="linspace"
                ),
                "unit_index": ArraySpec(
                    dtype="int64",
                    shape=["n_spikes"],
                    distribution="randint",
                    params={"low": 0, "high_dim": "n_units"},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "unit_"},
                ),
                "unit_number": ArraySpec(
                    dtype="int64", shape=["n_units"], distribution="sequential_int"
                ),
                "count": ArraySpec(
                    dtype="int64",
                    shape=["n_units"],
                    distribution="randint",
                    params={"low": 100, "high": 10000},
                ),
                "type": ArraySpec(
                    dtype="int64",
                    shape=["n_units"],
                    distribution="choice",
                    params={"values": [1]},
                ),
            },
        ),
        "cursor": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=1000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
                "vel": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 100.0},
                ),
                "acc": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 500.0},
                ),
            },
        ),
        "hand": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=1000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos_2d": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
                "vel_2d": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 100.0},
                ),
                "acc_2d": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 500.0},
                ),
            },
        ),
        "eye": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=1000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 10.0},
                ),
            },
        ),
        "trials": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_trials",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_end=True
                ),
                "target_on_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.1},
                ),
                "go_cue_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.2},
                ),
                "move_begins_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.3},
                ),
                "move_ends_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.7},
                ),
                "is_valid": ArraySpec(
                    dtype="bool",
                    shape=["n_trials"],
                    distribution="bool_random",
                    params={"true_prob": 0.9},
                ),
            },
        ),
    },
    metadata={
        "species": "MACACA_MULATTA",
        "task": "REACHING",
        "recording_tech": "UTAH_ARRAY_THRESHOLD_CROSSINGS",
    },
)

# -----------------------------------------------------------------------------
# Perich-Miller 2018: Center-out and random target reaching
# -----------------------------------------------------------------------------
BRAINSET_SPECS["perich_miller_population_2018"] = BrainsetSpec(
    brainset_id="perich_miller_population_2018",
    description="Center-out and random target reaching with M1/PMd recordings",
    dimensions={
        "n_spikes": 50000,
        "n_units": 96,
        "n_behavior_samples": 10000,
        "n_trials": 5,
    },
    objects={
        "spikes": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_spikes",
            sampling_rate=5000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_spikes"], distribution="linspace"
                ),
                "unit_index": ArraySpec(
                    dtype="int64",
                    shape=["n_spikes"],
                    distribution="randint",
                    params={"low": 0, "high_dim": "n_units"},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "unit_"},
                ),
            },
        ),
        "cursor": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=1000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "pos": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 5.0},
                ),
                "vel": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 10.0},
                ),
                "acc": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
            },
        ),
        "trials": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_trials",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_end=True
                ),
                "target_on_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.1},
                ),
                "go_cue_time": ArraySpec(
                    dtype="float64",
                    shape=["n_trials"],
                    distribution="interval_relative",
                    params={"offset_frac": 0.2},
                ),
                "is_valid": ArraySpec(
                    dtype="bool",
                    shape=["n_trials"],
                    distribution="bool_random",
                    params={"true_prob": 0.9},
                ),
            },
        ),
    },
    metadata={
        "species": "MACACA_MULATTA",
        "task": "REACHING",
        "recording_tech": "UTAH_ARRAY_SPIKES",
    },
)

# -----------------------------------------------------------------------------
# Flint-Slutzky 2012: Reaching with threshold crossings
# -----------------------------------------------------------------------------
BRAINSET_SPECS["flint_slutzky_accurate_2012"] = BrainsetSpec(
    brainset_id="flint_slutzky_accurate_2012",
    description="Reaching task with M1/PMd recordings using Cerebus system",
    dimensions={
        "n_spikes": 30000,
        "n_units": 128,
        "n_behavior_samples": 100,
        "n_trials": 5,
    },
    objects={
        "spikes": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_spikes",
            sampling_rate=3000.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_spikes"], distribution="linspace"
                ),
                "unit_index": ArraySpec(
                    dtype="int64",
                    shape=["n_spikes"],
                    distribution="randint",
                    params={"low": 0, "high_dim": "n_units"},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "unit_"},
                ),
                "unit_number": ArraySpec(
                    dtype="int64", shape=["n_units"], distribution="sequential_int"
                ),
            },
        ),
        "hand": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_behavior_samples",
            sampling_rate=10.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_behavior_samples"],
                    distribution="linspace",
                ),
                "vel": ArraySpec(
                    dtype="float32",
                    shape=["n_behavior_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 20.0},
                ),
            },
        ),
        "trials": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_trials",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_trials"], is_interval_end=True
                ),
            },
        ),
    },
    metadata={
        "species": "MACACA_MULATTA",
        "task": "REACHING",
        "recording_tech": "UTAH_ARRAY_SPIKES",
    },
)

# -----------------------------------------------------------------------------
# Allen Visual Coding Ophys 2016: Two-photon calcium imaging
# -----------------------------------------------------------------------------
BRAINSET_SPECS["allen_visual_coding_ophys_2016"] = BrainsetSpec(
    brainset_id="allen_visual_coding_ophys_2016",
    description="Two-photon calcium imaging during visual stimulus presentation",
    dimensions={
        "n_frames": 300,
        "n_units": 50,
        "n_running_samples": 600,
        "n_pupil_samples": 300,
        "n_stimuli": 5,
    },
    objects={
        "calcium_traces": TemporalObjectSpec(
            type=TemporalType.REGULAR_TIME_SERIES,
            length_dimension="n_frames",
            sampling_rate=30.0,
            attributes={
                "df_over_f": ArraySpec(
                    dtype="float32",
                    shape=["n_frames", "n_units"],
                    distribution="normal",
                    params={"mean": 0.0, "std": 0.5},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_units",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_units"],
                    distribution="sequential_str",
                    params={"prefix": "roi_"},
                ),
                "imaging_plane_xy": ArraySpec(
                    dtype="float32",
                    shape=["n_units", 2],
                    distribution="uniform",
                    params={"low": 0.0, "high": 512.0},
                ),
                "imaging_plane_area": ArraySpec(
                    dtype="float32",
                    shape=["n_units"],
                    distribution="uniform",
                    params={"low": 50.0, "high": 500.0},
                ),
                "imaging_plane_width": ArraySpec(
                    dtype="float32",
                    shape=["n_units"],
                    distribution="uniform",
                    params={"low": 5.0, "high": 30.0},
                ),
                "imaging_plane_height": ArraySpec(
                    dtype="float32",
                    shape=["n_units"],
                    distribution="uniform",
                    params={"low": 5.0, "high": 30.0},
                ),
            },
        ),
        "running": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_running_samples",
            sampling_rate=60.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64",
                    shape=["n_running_samples"],
                    distribution="linspace",
                ),
                "running_speed": ArraySpec(
                    dtype="float32",
                    shape=["n_running_samples", 1],
                    distribution="normal",
                    params={"mean": 5.0, "std": 10.0},
                ),
            },
        ),
        "pupil": TemporalObjectSpec(
            type=TemporalType.IRREGULAR_TIME_SERIES,
            length_dimension="n_pupil_samples",
            sampling_rate=30.0,
            attributes={
                "timestamps": ArraySpec(
                    dtype="float64", shape=["n_pupil_samples"], distribution="linspace"
                ),
                "location": ArraySpec(
                    dtype="float32",
                    shape=["n_pupil_samples", 2],
                    distribution="normal",
                    params={"mean": 0.0, "std": 5.0},
                ),
                "size": ArraySpec(
                    dtype="float32",
                    shape=["n_pupil_samples"],
                    distribution="uniform",
                    params={"low": 10.0, "high": 100.0},
                ),
            },
        ),
        "drifting_gratings": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_stimuli",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_stimuli"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_stimuli"], is_interval_end=True
                ),
                "orientation": ArraySpec(
                    dtype="float64",
                    shape=["n_stimuli"],
                    distribution="choice",
                    params={"values": [0, 45, 90, 135, 180, 225, 270, 315]},
                ),
                "temporal_frequency": ArraySpec(
                    dtype="float64",
                    shape=["n_stimuli"],
                    distribution="choice",
                    params={"values": [1, 2, 4, 8, 15]},
                ),
            },
        ),
    },
    metadata={
        "species": "MUS_MUSCULUS",
        "task": None,
        "recording_tech": "TWO_PHOTON_IMAGING",
    },
)

# -----------------------------------------------------------------------------
# Kemp Sleep EDF 2013: Polysomnography sleep recordings
# -----------------------------------------------------------------------------
BRAINSET_SPECS["kemp_sleep_edf_2013"] = BrainsetSpec(
    brainset_id="kemp_sleep_edf_2013",
    description="Whole-night polysomnographic sleep recordings with EEG, EOG, EMG",
    dimensions={
        "n_samples": 3000,
        "n_channels": 4,
        "n_stages": 15,
    },
    objects={
        "eeg": TemporalObjectSpec(
            type=TemporalType.REGULAR_TIME_SERIES,
            length_dimension="n_samples",
            sampling_rate=100.0,
            attributes={
                "signal": ArraySpec(
                    dtype="float32",
                    shape=["n_samples", "n_channels"],
                    distribution="normal",
                    params={"mean": 0.0, "std": 50.0},
                ),
            },
        ),
        "units": TemporalObjectSpec(
            type=TemporalType.ARRAY_DICT,
            length_dimension="n_channels",
            attributes={
                "id": ArraySpec(
                    dtype="str",
                    shape=["n_channels"],
                    distribution="sequential_str",
                    params={"prefix": "ch_"},
                ),
                "modality": ArraySpec(
                    dtype="str",
                    shape=["n_channels"],
                    distribution="choice",
                    params={"values": ["EEG", "EOG", "EMG"]},
                ),
            },
        ),
        "stages": TemporalObjectSpec(
            type=TemporalType.INTERVAL,
            length_dimension="n_stages",
            attributes={
                "start": ArraySpec(
                    dtype="float64", shape=["n_stages"], is_interval_start=True
                ),
                "end": ArraySpec(
                    dtype="float64", shape=["n_stages"], is_interval_end=True
                ),
                "names": ArraySpec(
                    dtype="str",
                    shape=["n_stages"],
                    distribution="choice",
                    params={
                        "values": [
                            "Sleep stage W",
                            "Sleep stage 1",
                            "Sleep stage 2",
                            "Sleep stage 3",
                            "Sleep stage R",
                        ]
                    },
                ),
                "id": ArraySpec(
                    dtype="int64",
                    shape=["n_stages"],
                    distribution="choice",
                    params={"values": [0, 1, 2, 3, 5]},
                ),
            },
        ),
    },
    metadata={
        "species": "HOMO_SAPIENS",
        "task": None,
        "recording_tech": "POLYSOMNOGRAPHY",
    },
)


# =============================================================================
# Generator Function - Fully Generic
# =============================================================================


def generate_dummy_data(
    brainset_id: str,
    time_span: float = 10.0,
    n_units: int = 96,
    seed: Optional[int] = None,
) -> Data:
    """Generate a dummy Data object matching the structure of a brainset.

    Args:
        brainset_id: Identifier of the brainset to generate data for.
        time_span: Duration of the generated data in seconds.
        n_units: Number of units (neurons/channels) to generate.
        seed: Random seed for reproducibility.

    Returns:
        A temporaldata.Data object with realistic dummy data.

    Raises:
        ValueError: If brainset_id is not recognized.
    """
    if brainset_id not in BRAINSET_SPECS:
        available = ", ".join(BRAINSET_SPECS.keys())
        raise ValueError(f"Unknown brainset_id: {brainset_id}. Available: {available}")

    if seed is not None:
        np.random.seed(seed)

    spec = BRAINSET_SPECS[brainset_id]
    metadata = spec.metadata

    # Scale dimensions based on time_span
    resolved_dims = _resolve_dimensions(spec, time_span, n_units)

    brainset_description = Data(
        id=brainset_id,
        origin_version="dummy/1.0.0",
        derived_version="1.0.0",
        source="generated",
        description=spec.description or f"Dummy data generated for {brainset_id}",
    )

    subject = Data(
        id="dummy_subject",
        species=metadata.get("species", "MACACA_MULATTA"),
        sex="UNKNOWN",
        age=0.0,
        genotype="unknown",
    )

    session = Data(
        id="dummy_session",
        recording_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        task=metadata.get("task"),
    )

    device = Data(
        id="dummy_device",
        recording_tech=metadata.get("recording_tech", "UTAH_ARRAY_SPIKES"),
    )

    generated_objects = {}
    for obj_name, obj_spec in spec.objects.items():
        obj = _generate_object(obj_spec, time_span, resolved_dims)
        if obj is not None:
            generated_objects[obj_name] = obj

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session,
        device=device,
        domain=Interval(0.0, time_span),
        **generated_objects,
    )

    train_end = time_span * 0.7
    valid_end = time_span * 0.8
    data.set_train_domain(Interval(0.0, train_end))
    data.set_valid_domain(Interval(train_end, valid_end))
    data.set_test_domain(Interval(valid_end, time_span))

    return data


def _resolve_dimensions(
    spec: BrainsetSpec, time_span: float, n_units: int
) -> Dict[str, int]:
    """Resolve dimension values based on time_span and n_units."""
    resolved = {}
    for dim_name, default_val in spec.dimensions.items():
        # Check if this dimension is used as length_dimension for any object
        for obj_spec in spec.objects.values():
            if obj_spec.length_dimension == dim_name and obj_spec.sampling_rate:
                resolved[dim_name] = int(time_span * obj_spec.sampling_rate)
                break
        else:
            # Use n_units for dimensions containing "unit" or "channel"
            if "unit" in dim_name.lower() or "channel" in dim_name.lower():
                resolved[dim_name] = n_units
            else:
                resolved[dim_name] = default_val
    return resolved


def _generate_object(
    obj_spec: TemporalObjectSpec,
    time_span: float,
    resolved_dims: Dict[str, int],
) -> Optional[Any]:
    """Generate a single temporal data object from its specification."""
    if obj_spec.type == TemporalType.IRREGULAR_TIME_SERIES:
        return _generate_irregular_time_series(obj_spec, time_span, resolved_dims)
    elif obj_spec.type == TemporalType.REGULAR_TIME_SERIES:
        return _generate_regular_time_series(obj_spec, resolved_dims)
    elif obj_spec.type == TemporalType.ARRAY_DICT:
        return _generate_array_dict(obj_spec, resolved_dims)
    elif obj_spec.type == TemporalType.INTERVAL:
        return _generate_interval(obj_spec, time_span, resolved_dims)
    return None


def _generate_irregular_time_series(
    obj_spec: TemporalObjectSpec,
    time_span: float,
    resolved_dims: Dict[str, int],
) -> IrregularTimeSeries:
    """Generate an IrregularTimeSeries object."""
    attrs = {}
    for attr_name, attr_spec in obj_spec.attributes.items():
        shape = _resolve_shape(attr_spec.shape, resolved_dims)
        attrs[attr_name] = _generate_array(attr_spec, shape, resolved_dims, time_span)
    return IrregularTimeSeries(**attrs, domain="auto")


def _generate_regular_time_series(
    obj_spec: TemporalObjectSpec,
    resolved_dims: Dict[str, int],
) -> RegularTimeSeries:
    """Generate a RegularTimeSeries object."""
    attrs = {}
    for attr_name, attr_spec in obj_spec.attributes.items():
        shape = _resolve_shape(attr_spec.shape, resolved_dims)
        attrs[attr_name] = _generate_array(attr_spec, shape, resolved_dims)
    return RegularTimeSeries(
        **attrs, sampling_rate=obj_spec.sampling_rate, domain="auto"
    )


def _generate_array_dict(
    obj_spec: TemporalObjectSpec,
    resolved_dims: Dict[str, int],
) -> ArrayDict:
    """Generate an ArrayDict object."""
    attrs = {}
    for attr_name, attr_spec in obj_spec.attributes.items():
        shape = _resolve_shape(attr_spec.shape, resolved_dims)
        attrs[attr_name] = _generate_array(attr_spec, shape, resolved_dims)
    return ArrayDict(**attrs)


def _generate_interval(
    obj_spec: TemporalObjectSpec,
    time_span: float,
    resolved_dims: Dict[str, int],
) -> Interval:
    """Generate an Interval object."""
    n_intervals = resolved_dims.get(obj_spec.length_dimension, 5)
    interval_duration = time_span / n_intervals

    starts = np.array([i * interval_duration for i in range(n_intervals)])
    ends = starts + interval_duration * 0.9

    attrs = {}
    for attr_name, attr_spec in obj_spec.attributes.items():
        if attr_spec.is_interval_start:
            attrs[attr_name] = starts
        elif attr_spec.is_interval_end:
            attrs[attr_name] = ends
        elif attr_spec.distribution == "interval_relative":
            offset_frac = attr_spec.params.get("offset_frac", 0.5)
            attrs[attr_name] = starts + interval_duration * offset_frac
        else:
            shape = _resolve_shape(attr_spec.shape, resolved_dims)
            attrs[attr_name] = _generate_array(
                attr_spec, shape, resolved_dims, time_span
            )

    return Interval(**attrs)


def _resolve_shape(
    shape_spec: List[Union[int, str]], resolved_dims: Dict[str, int]
) -> Tuple[int, ...]:
    """Resolve shape specification, replacing dimension names with values."""
    return tuple(
        resolved_dims.get(dim, dim) if isinstance(dim, str) else dim
        for dim in shape_spec
    )


def _generate_array(
    attr_spec: ArraySpec,
    shape: Tuple[int, ...],
    resolved_dims: Dict[str, int],
    time_span: float = 1.0,
) -> np.ndarray:
    """Generate an array based on its specification."""
    distribution = attr_spec.distribution
    params = attr_spec.params
    dtype = attr_spec.dtype

    if distribution == "normal":
        arr = np.random.normal(params.get("mean", 0), params.get("std", 1), shape)
    elif distribution == "uniform":
        arr = np.random.uniform(params.get("low", 0), params.get("high", 1), shape)
    elif distribution == "linspace":
        arr = np.linspace(0, time_span, shape[0])
    elif distribution == "randint":
        low = params.get("low", 0)
        high = params.get("high", shape[0])
        if "high_dim" in params:
            high = resolved_dims.get(params["high_dim"], high)
        arr = np.random.randint(low, high, shape)
    elif distribution == "choice":
        arr = np.random.choice(params.get("values", [0, 1]), size=shape)
    elif distribution == "sequential_str":
        prefix = params.get("prefix", "item_")
        arr = np.array([f"{prefix}{i}" for i in range(shape[0])])
    elif distribution == "sequential_int":
        arr = np.arange(shape[0])
    elif distribution == "bool_random":
        arr = np.random.random(shape) < params.get("true_prob", 0.5)
    else:
        arr = np.zeros(shape)

    if dtype == "str":
        return arr.astype(str) if arr.dtype != object else arr
    elif dtype == "bool":
        return arr.astype(bool)
    return arr.astype(dtype)


def get_available_dummy_brainsets() -> list:
    """Return list of available brainset IDs for dummy data generation."""
    return list(BRAINSET_SPECS.keys())
