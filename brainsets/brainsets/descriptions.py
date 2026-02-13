import datetime
from typing import Dict, List, Tuple, Optional, Union

from pydantic import field_validator
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

    Fields are automatically normalized during construction:
    - ``species`` accepts a Species enum, string, int, or None (defaults to Species.UNKNOWN)
    - ``age`` accepts a float, int, numeric string, or None (defaults to 0.0)
    - ``sex`` accepts a Sex enum, string, int, or None (defaults to Sex.UNKNOWN)

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
    species: Species = Species.UNKNOWN
    age: float = 0.0  # in days
    sex: Sex = Sex.UNKNOWN
    genotype: str = "unknown"  # no idea how many there will be for now.
    cre_line: Optional[Cre_line] = None

    @field_validator("age", mode="before")
    @classmethod
    def normalize_age(cls, age: Union[float, int, str, None] = None) -> float:
        """Normalize an age value to a float in days.

        Args:
            age: Age of the subject. Can be a float, int, numeric string, or None.

        Returns:
            Normalized age as a float. Defaults to 0.0 if None or unparseable.

        Raises:
            ValueError: If the age is negative.
            TypeError: If the age is not a float, int, numeric string, or None.
        """
        if age is None:
            return 0.0
        elif isinstance(age, (int, float)):
            age_normalized = float(age)
            if age_normalized < 0:
                raise ValueError(f"Age cannot be negative, got {age_normalized}")
            return age_normalized
        elif isinstance(age, str):
            try:
                age_normalized = float(age)
            except (ValueError, TypeError):
                return 0.0
            else:
                if age_normalized < 0:
                    raise ValueError(f"Age cannot be negative, got {age_normalized}")
                return age_normalized
        else:
            raise TypeError(
                f"Age must be a float, int, numeric string, or None, got {type(age).__name__}"
            )

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, sex: Union[str, int, Sex, None] = None) -> Sex:
        """Normalize a sex value to a Sex enum member.

        Args:
            sex: Sex of the subject. Can be a string (e.g. "M", "MALE"), an int
                (0=UNKNOWN, 1=MALE, 2=FEMALE, 3=OTHER), a Sex enum member, or None.

        Returns:
            Normalized Sex enum member. Defaults to Sex.UNKNOWN if None or unrecognized.

        Raises:
            TypeError: If the sex is not a Sex enum, string, int, or None.
        """
        if sex is None:
            return Sex.UNKNOWN
        elif isinstance(sex, Sex):
            return sex
        elif isinstance(sex, bool):
            raise TypeError(f"Sex must be a Sex enum, string, int, or None, got bool")
        elif isinstance(sex, str):
            try:
                return Sex.from_string(sex)
            except ValueError:
                return Sex.UNKNOWN
        elif isinstance(sex, int):
            try:
                return Sex(sex)
            except ValueError:
                return Sex.UNKNOWN
        else:
            raise TypeError(
                f"Sex must be a Sex enum, string, int, or None, got {type(sex).__name__}"
            )

    @field_validator("species", mode="before")
    @classmethod
    def normalize_species(
        cls, species: Union[str, int, Species, None] = None
    ) -> Species:
        """Normalize a species value to a Species enum member.

        Args:
            species: Species of the subject. Can be a string, an int, a Species
                enum member, or None.

        Returns:
            Normalized Species enum member. Defaults to Species.UNKNOWN if None or
            unrecognized.

        Raises:
            TypeError: If the species is not a Species enum, string, int, or None.
        """
        if species is None:
            return Species.UNKNOWN
        elif isinstance(species, Species):
            return species
        elif isinstance(species, str):
            try:
                return Species.from_string(species)
            except ValueError:
                return Species.UNKNOWN
        elif isinstance(species, int):
            try:
                return Species(species)
            except ValueError:
                return Species.UNKNOWN
        else:
            raise TypeError(
                f"Species must be a Species enum, string, int, or None, got {type(species).__name__}"
            )


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
