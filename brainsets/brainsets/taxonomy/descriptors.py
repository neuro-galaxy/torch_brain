import datetime
from typing import Dict, List, Tuple, Optional, Union

from pydantic.dataclasses import dataclass
import temporaldata

import brainsets
from brainsets.taxonomy import *
from brainsets.taxonomy.mice import *


@dataclass
class BrainsetDescription(temporaldata.Data):
    id: str
    origin_version: str
    derived_version: str
    source: str
    description: str
    brainsets_version: str = brainsets.__version__
    temporaldata_version: str = temporaldata.__version__


@dataclass
class SubjectDescription(temporaldata.Data):
    id: str
    species: Species
    age: float = 0.0  # in days
    sex: Sex = Sex.UNKNOWN
    genotype: str = "unknown"  # no idea how many there will be for now.
    cre_line: Optional[Cre_line] = None
    target_area: Optional[Vis_areas] = None
    depth_class: Optional[Depth_classes] = None


@dataclass
class SessionDescription(temporaldata.Data):
    id: str
    recording_date: datetime.datetime
    task: Task


@dataclass
class DeviceDescription(temporaldata.Data):
    id: str
    # units: List[str]
    # areas: Union[List[StringIntEnum], List[Macaque]]
    recording_tech: Union[RecordingTech, List[RecordingTech]] = None
    processing: Optional[str] = None
    chronic: bool = False
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
