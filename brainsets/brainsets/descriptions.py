import datetime
from typing import Dict, List, Tuple, Optional, Union

from pydantic.dataclasses import dataclass
import temporaldata

import brainsets
from brainsets.taxonomy import *
from brainsets.taxonomy.mice import *


@dataclass
class BrainsetDescription(temporaldata.Data):
    r"""A class for describing a brainset.

    Parameters
    ----------
    id : str
        Unique identifier for the brainset
    origin_version : str
        Version identifier for the original data source
    derived_version : str
        Version identifier for the derived/processed data
    source : str
        Original data source (usually a URL, or a short description otherwise)
    description : str
        Text description of the brainset
    brainsets_version : str, optional
        Version of brainsets package used, defaults to current version
    temporaldata_version : str, optional
        Version of temporaldata package used, defaults to current version
    """

    id: str
    origin_version: str
    derived_version: str
    source: str
    description: str
    brainsets_version: str = brainsets.__version__
    temporaldata_version: str = temporaldata.__version__


@dataclass
class SubjectDescription(temporaldata.Data):
    r"""A class for describing a subject.

    Parameters
    ----------
    id : str
        Unique identifier for the subject
    species : Species
        Species of the subject
    age : float, optional
        Age of the subject in days, defaults to 0.0
    sex : Sex, optional
        Sex of the subject, defaults to UNKNOWN
    genotype : str, optional
        Genotype of the subject, defaults to "unknown"
    cre_line : Cre_line, optional
        Cre line of the subject, defaults to None
    """

    id: str
    species: Species
    age: float = 0.0  # in days
    sex: Sex = Sex.UNKNOWN
    genotype: str = "unknown"  # no idea how many there will be for now.
    cre_line: Optional[Cre_line] = None


@dataclass
class SessionDescription(temporaldata.Data):
    r"""A class for describing an experimental session.

    Parameters
    ----------
    id : str
        Unique identifier for the session
    recording_date : datetime.datetime
        Date and time when the recording was made
    task : Task
        Task performed during the session
    """

    id: str
    recording_date: datetime.datetime
    task: Optional[Task] = None


@dataclass
class DeviceDescription(temporaldata.Data):
    r"""A class for describing a recording device.

    Parameters
    ----------
    id : str
        Unique identifier for the device
    recording_tech : RecordingTech or List[RecordingTech], optional
        Recording technology used, defaults to None
    processing : str, optional
        Processing applied to the recording, defaults to None
    chronic : bool, optional
        Whether the device was chronically implanted, defaults to False
    start_date : datetime.datetime, optional
        Date when device was implanted/first used, defaults to None
    end_date : datetime.datetime, optional
        Date when device was removed/last used, defaults to None
    imaging_depth : float, optional
        Depth of imaging in micrometers, defaults to None
    target_area : BrainRegion, optional
        Target brain region for recording, defaults to None
    """

    id: str
    # units: List[str]
    # areas: Union[List[StringIntEnum], List[Macaque]]
    recording_tech: Union[RecordingTech, List[RecordingTech]] = None
    processing: Optional[str] = None
    chronic: bool = False
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
    # Ophys
    imaging_depth: Optional[float] = None  # in um
    target_area: Optional[BrainRegion] = None
