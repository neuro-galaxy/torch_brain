__all__ = [
    "BrainsetDescription",
    "SubjectDescription",
    "SessionDescription",
    "DeviceDescription",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}

import datetime

import torch_brain
from .data import Data


def _validate_string_type(v, name: str, allow_none: bool = False):
    if v is None:
        if allow_none:
            return v
        else:
            raise ValueError(f"{name} must be a string, got {v!r}")

    if not isinstance(v, str):
        if allow_none:
            raise ValueError(f"{name} must be a string or None, got {v!r}")
        else:
            raise ValueError(f"{name} must be a string, got {v!r}")

    if len(v) == 0:
        raise ValueError(f"{name} cannot be an empty string, got {v!r}")


class BrainsetDescription(Data):
    r"""A container for storing brainset metadata.

    Args:
        id: Unique identifier for the brainset
        origin_version: Version identifier for the original data source
        derived_version: Version identifier for the derived/processed data
        source: Original data source (usually a URL, or a short description otherwise)
        description: Text description of the brainset
        **kwargs: Any additional metadata
    """

    id: str
    origin_version: str
    derived_version: str
    source: str
    description: str
    torch_brain_version: str

    def __init__(
        self,
        id: str,
        origin_version: str,
        derived_version: str,
        source: str,
        description: str,
        **kwargs,
    ):
        _validate_string_type(id, "id")
        _validate_string_type(origin_version, "origin_version")
        _validate_string_type(derived_version, "derived_version")
        _validate_string_type(source, "source")
        _validate_string_type(description, "description")

        # torch_brain_version needs to be set by us
        if "torch_brain_version" in kwargs:
            raise ValueError("Cannot set torch_brain_version manually")

        super().__init__(
            id=id,
            origin_version=origin_version,
            derived_version=derived_version,
            source=source,
            description=description,
            torch_brain_version=torch_brain.__version__,
            **kwargs,
        )


class SubjectDescription(Data):
    r"""A container for storing subject related metadata.

    Args:
        id: Unique identifier for the subject
        species: Species of the subject, defaults to None
        age: Age of the subject (in days).
            It will be converted to float if not None. defaults to None
        sex: Sex of the subject, defaults to None
    """

    id: str
    species: str | None
    age: float | None
    sex: str | None

    def __init__(
        self,
        id: str,
        species: str | None = None,
        age: int | str | float | None = None,
        sex: str | None = None,
        **kwargs,
    ):

        _validate_string_type(id, "id")
        _validate_string_type(species, "species", allow_none=True)
        _validate_string_type(sex, "sex", allow_none=True)
        age = self._normalize_age(age)

        super().__init__(
            id=id,
            species=species,
            age=age,
            sex=sex,
            **kwargs,
        )

    def _normalize_age(self, age) -> float | None:
        """Normalize and validate age value to a float in days."""

        if age is None:
            return None

        if isinstance(age, (int, float)):
            age_normalized = float(age)
            if age_normalized < 0:
                raise ValueError(f"age cannot be negative, got {age_normalized}")
            return age_normalized

        if isinstance(age, str):
            age_normalized = float(age)

            if age_normalized < 0:
                raise ValueError(f"age cannot be negative, got {age_normalized}")
            return age_normalized

        raise TypeError(
            f"age must be a float, int, numeric string, or None, got {type(age).__name__}"
        )


class SessionDescription(Data):
    r"""A container to store experimental session related metadata.

    Args:
        id: Unique identifier for the session
        recording_date: Date and time when the recording was made, defaults to None
        **kwargs: Any additional metadata
    """

    id: str
    recording_date: datetime.datetime | None

    def __init__(
        self,
        id: str,
        recording_date: datetime.datetime | None = None,
        **kwargs,
    ):

        _validate_string_type(id, "id")

        if recording_date is not None:
            if not isinstance(recording_date, datetime.datetime):
                raise ValueError(
                    "recording_date must be None or a datetime.datetime object"
                    f", got {recording_date!r}"
                )

        super().__init__(
            id=id,
            recording_date=recording_date,
            **kwargs,
        )


class DeviceDescription(Data):
    r"""A container for storing recording device metadata.

    Args:
        id: Identifier for the device
        recording_tech: A string description of the device tech, default None
        **kwargs: Any additional metadata
    """

    id: str
    recording_tech: str | None

    def __init__(
        self,
        id: str,
        recording_tech: str | None = None,
        **kwargs,
    ):

        _validate_string_type(id, "id")
        _validate_string_type(recording_tech, "recording_tech", allow_none=True)

        super().__init__(
            id=id,
            recording_tech=recording_tech,
            **kwargs,
        )
