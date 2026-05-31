import datetime

import pytest

import torch_brain
from torch_brain.data import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)


class TestSubjectDescription:

    def test_basic_usage_with_all_parameters(self):
        result = SubjectDescription(
            id="subject_1",
            age=30.5,
            sex="MALE",
            species="HUMAN",
            extra_md="something-extra",
        )

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age == 30.5
        assert result.sex == "MALE"
        assert result.species == "HUMAN"
        assert result.extra_md == "something-extra"

    def test_minimal_usage_with_only_id(self):
        result = SubjectDescription(id="subject_1")

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age is None
        assert result.sex is None
        assert result.species is None

    # Age normalization tests
    def test_age_as_int(self):
        result = SubjectDescription(id="subject_1", age=25)
        assert result.age == 25.0
        assert isinstance(result.age, float)

    def test_age_as_float(self):
        result = SubjectDescription(id="subject_1", age=30.5)
        assert result.age == 30.5

    def test_age_as_string_numeric(self):
        result = SubjectDescription(id="subject_1", age="45.7")
        assert result.age == 45.7

    def test_invalid_age_raises(self):
        with pytest.raises(ValueError):
            SubjectDescription(id="subject_1", age="invalid")

    def test_sex_type_validation(self):
        invalid_options = [True, 0.1, 0, {1, 2}, (1, 2), [1, 2]]
        for sex in invalid_options:
            with pytest.raises(ValueError, match="sex must be a string or None"):
                SubjectDescription(id="subject_1", sex=sex)  # type: ignore

        with pytest.raises(ValueError, match="sex cannot be an empty string"):
            SubjectDescription(id="subject_1", sex="")

    def test_species_type_validation(self):
        invalid_options = [True, 0.1, 0, {1, 2}, (1, 2), [1, 2]]
        for species in invalid_options:
            with pytest.raises(ValueError, match="species must be a string or None"):
                SubjectDescription(id="subject_1", species=species)  # type: ignore

        with pytest.raises(ValueError, match="species cannot be an empty string"):
            SubjectDescription(id="subject_1", species="")

    def test_zero_age(self):
        result = SubjectDescription(id="subject_1", species=None, age=0)
        assert result.age == 0.0

    def test_negative_age(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5)

    def test_negative_age_float(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5.5)

    def test_negative_age_string(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age="-10")

    def test_age_with_unexpected_type(self):
        with pytest.raises(
            Exception, match="age must be a float, int, numeric string, or None"
        ):
            SubjectDescription(id="subject_1", age=[])  # type: ignore


class TestBrainsetDescription:

    def test_basic_usage_with_all_parameters(self):
        result = BrainsetDescription(
            id="brainset_1",
            origin_version="1.0.0",
            derived_version="2.0.0",
            source="https://example.com/data",
            description="A test brainset",
            extra_md="something-extra",
        )

        assert isinstance(result, BrainsetDescription)
        assert result.id == "brainset_1"
        assert result.origin_version == "1.0.0"
        assert result.derived_version == "2.0.0"
        assert result.source == "https://example.com/data"
        assert result.description == "A test brainset"
        assert result.extra_md == "something-extra"

    def test_version_fields_are_populated(self):
        result = BrainsetDescription(
            id="brainset_1",
            origin_version="1.0.0",
            derived_version="2.0.0",
            source="https://example.com/data",
            description="A test brainset",
        )

        assert result.torch_brain_version == torch_brain.__version__

    def test_missing_required_argument_raises(self):
        with pytest.raises(TypeError):
            BrainsetDescription(id="brainset_1")  # type: ignore

    @pytest.mark.parametrize(
        "field",
        ["id", "origin_version", "derived_version", "source", "description"],
    )
    def test_required_field_none_raises(self, field):
        kwargs = {
            "id": "brainset_1",
            "origin_version": "1.0.0",
            "derived_version": "2.0.0",
            "source": "https://example.com/data",
            "description": "A test brainset",
        }
        kwargs[field] = None
        with pytest.raises(ValueError, match=f"{field} must be a string"):
            BrainsetDescription(**kwargs)

    @pytest.mark.parametrize(
        "field",
        ["id", "origin_version", "derived_version", "source", "description"],
    )
    def test_required_field_non_string_raises(self, field):
        kwargs = {
            "id": "brainset_1",
            "origin_version": "1.0.0",
            "derived_version": "2.0.0",
            "source": "https://example.com/data",
            "description": "A test brainset",
        }
        kwargs[field] = 123
        with pytest.raises(ValueError, match=f"{field} must be a string, got"):
            BrainsetDescription(**kwargs)

    @pytest.mark.parametrize(
        "field",
        ["id", "origin_version", "derived_version", "source", "description"],
    )
    def test_required_field_empty_string_raises(self, field):
        kwargs = {
            "id": "brainset_1",
            "origin_version": "1.0.0",
            "derived_version": "2.0.0",
            "source": "https://example.com/data",
            "description": "A test brainset",
        }
        kwargs[field] = ""
        with pytest.raises(ValueError, match=f"{field} cannot be an empty string"):
            BrainsetDescription(**kwargs)

    def test_torch_brain_version_kwarg_raises(self):
        with pytest.raises(ValueError, match="Cannot set torch_brain_version manually"):
            BrainsetDescription(
                id="brainset_1",
                origin_version="1.0.0",
                derived_version="2.0.0",
                source="https://example.com/data",
                description="A test brainset",
                torch_brain_version="9.9.9",
            )


class TestSessionDescription:

    def test_basic_usage_with_all_parameters(self):
        recording_date = datetime.datetime(2024, 1, 15, 10, 30, 0)
        result = SessionDescription(
            id="session_1",
            recording_date=recording_date,
            extra_md="something-extra",
        )

        assert isinstance(result, SessionDescription)
        assert result.id == "session_1"
        assert result.recording_date == recording_date
        assert result.extra_md == "something-extra"

    def test_minimal_usage_with_only_id(self):
        result = SessionDescription(id="session_1")

        assert isinstance(result, SessionDescription)
        assert result.id == "session_1"
        assert result.recording_date is None

    def test_id_type_validation(self):
        with pytest.raises(ValueError, match="id must be a string, got"):
            SessionDescription(id=None)  # type: ignore

        with pytest.raises(ValueError, match="id must be a string, got"):
            SessionDescription(id=123)  # type: ignore

        with pytest.raises(ValueError, match="id cannot be an empty string"):
            SessionDescription(id="")

    def test_invalid_recording_date_raises(self):
        invalid_options = ["2024-01-15", 1705315800, 0.1, datetime.date(2024, 1, 15)]
        for recording_date in invalid_options:
            with pytest.raises(
                ValueError,
                match="recording_date must be None or a datetime.datetime object",
            ):
                SessionDescription(id="session_1", recording_date=recording_date)  # type: ignore


class TestDeviceDescription:

    def test_basic_usage_with_all_parameters(self):
        result = DeviceDescription(
            id="device_1",
            recording_tech="Neuropixels",
            extra_md="something-extra",
        )

        assert isinstance(result, DeviceDescription)
        assert result.id == "device_1"
        assert result.recording_tech == "Neuropixels"
        assert result.extra_md == "something-extra"

    def test_minimal_usage_with_only_id(self):
        result = DeviceDescription(id="device_1")

        assert isinstance(result, DeviceDescription)
        assert result.id == "device_1"
        assert result.recording_tech is None

    def test_id_type_validation(self):
        with pytest.raises(ValueError, match="id must be a string, got"):
            DeviceDescription(id=None)  # type: ignore

        with pytest.raises(ValueError, match="id must be a string, got"):
            DeviceDescription(id=123)  # type: ignore

        with pytest.raises(ValueError, match="id cannot be an empty string"):
            DeviceDescription(id="")

    def test_recording_tech_type_validation(self):
        invalid_options = [True, 0.1, 0, {1, 2}, (1, 2), [1, 2]]
        for recording_tech in invalid_options:
            with pytest.raises(
                ValueError, match="recording_tech must be a string or None"
            ):
                DeviceDescription(id="device_1", recording_tech=recording_tech)  # type: ignore

        with pytest.raises(
            ValueError, match="recording_tech cannot be an empty string"
        ):
            DeviceDescription(id="device_1", recording_tech="")
