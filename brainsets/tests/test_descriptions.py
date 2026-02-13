import pytest
from brainsets.descriptions import SubjectDescription
from brainsets.taxonomy import Species, Sex


class TestSubjectDescription:

    def test_basic_usage_with_all_parameters(self):
        result = SubjectDescription(
            id="subject_1",
            age=30.5,
            sex=Sex.MALE,
            species=Species.HUMAN,
        )

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age == 30.5
        assert result.sex == Sex.MALE
        assert result.species == Species.HUMAN

    def test_minimal_usage_with_only_id(self):
        result = SubjectDescription(id="subject_1")

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age == 0.0
        assert result.sex == Sex.UNKNOWN
        assert result.species == Species.UNKNOWN

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

    def test_age_as_string_invalid(self):
        result = SubjectDescription(id="subject_1", age="invalid")
        assert result.age == 0.0

    # Sex normalization tests
    def test_sex_as_enum(self):
        result = SubjectDescription(id="subject_1", species=None, sex=Sex.FEMALE)
        assert result.sex == Sex.FEMALE

    def test_sex_as_string_valid(self):
        test_cases = [
            ("M", Sex.MALE),
            ("F", Sex.FEMALE),
            ("O", Sex.OTHER),
            ("U", Sex.UNKNOWN),
            ("male", Sex.MALE),
            ("FEMALE", Sex.FEMALE),
            ("Other", Sex.OTHER),
            ("unknown", Sex.UNKNOWN),
        ]
        for sex_str, expected_sex in test_cases:
            result = SubjectDescription(id="subject_1", sex=sex_str)
            assert result.sex == expected_sex, f"Failed for input: {sex_str}"

    def test_sex_as_string_invalid(self):
        result = SubjectDescription(id="subject_1", sex="invalid_sex")
        assert result.sex == Sex.UNKNOWN

    def test_sex_as_int_valid(self):
        test_cases = [
            (0, Sex.UNKNOWN),
            (1, Sex.MALE),
            (2, Sex.FEMALE),
            (3, Sex.OTHER),
        ]
        for sex_int, expected_sex in test_cases:
            result = SubjectDescription(id="subject_1", sex=sex_int)
            assert result.sex == expected_sex, f"Failed for input: {sex_int}"

    def test_sex_as_int_invalid(self):
        result = SubjectDescription(id="subject_1", sex=99)
        assert result.sex == Sex.UNKNOWN

    def test_sex_as_boolean(self):
        """Test that passing a boolean for sex raises an error."""
        with pytest.raises(
            Exception, match="Sex must be a Sex enum, string, int, or None"
        ):
            SubjectDescription(id="subject_1", sex=True)

        with pytest.raises(
            Exception, match="Sex must be a Sex enum, string, int, or None"
        ):
            SubjectDescription(id="subject_1", sex=False)

    def test_sex_as_float(self):
        with pytest.raises(
            Exception, match="Sex must be a Sex enum, string, int, or None"
        ):
            SubjectDescription(id="subject_1", sex=3.14)

    def test_sex_as_tuple(self):
        with pytest.raises(
            Exception, match="Sex must be a Sex enum, string, int, or None"
        ):
            SubjectDescription(id="subject_1", sex=(1, 2))

    def test_species_as_list(self):
        with pytest.raises(
            Exception, match="Species must be a Species enum, string, int, or None"
        ):
            SubjectDescription(id="subject_1", species=[])

    # Species normalization tests
    def test_species_as_enum(self):
        result = SubjectDescription(id="subject_1", species=Species.HOMO_SAPIENS)
        assert result.species == Species.HOMO_SAPIENS

    def test_species_as_string_valid(self):
        test_cases = [
            ("MACACA_MULATTA", Species.MACACA_MULATTA),
            ("HOMO_SAPIENS", Species.HOMO_SAPIENS),
            ("MUS_MUSCULUS", Species.MUS_MUSCULUS),
            ("macaca_mulatta", Species.MACACA_MULATTA),
            ("Homo Sapiens", Species.HOMO_SAPIENS),
            ("mus musculus", Species.MUS_MUSCULUS),
        ]

        for species_str, expected_species in test_cases:
            result = SubjectDescription(id="s_1", species=species_str)
            assert (
                result.species == expected_species
            ), f"Failed for input: {species_str}"

    def test_species_as_string_invalid(self):
        result = SubjectDescription(id="subject_1", species="invalid_species")
        assert result.species == Species.UNKNOWN

    def test_species_as_int_valid(self):
        test_cases = [
            (0, Species.UNKNOWN),
            (1, Species.MACACA_MULATTA),
            (2, Species.HOMO_SAPIENS),
            (3, Species.MUS_MUSCULUS),
            (4, Species.MACACA_FASCICULARIS),
        ]

        for species_int, expected_species in test_cases:
            result = SubjectDescription(id="s_1", species=species_int)
            assert (
                result.species == expected_species
            ), f"Failed for input: {species_int}"

    def test_species_as_int_invalid(self):
        result = SubjectDescription(id="s_1", species=999)
        assert result.species == Species.UNKNOWN

    # Combined parameter tests
    def test_all_parameters_with_strings(self):
        result = SubjectDescription(
            id="subject_1",
            age="45.5",
            sex="M",
            species="HOMO_SAPIENS",
        )

        assert result.id == "subject_1"
        assert result.age == 45.5
        assert result.sex == Sex.MALE
        assert result.species == Species.HOMO_SAPIENS

    def test_all_parameters_with_ints(self):
        result = SubjectDescription(
            id="subject_1",
            age=30,
            sex=2,
            species=2,
        )

        assert result.id == "subject_1"
        assert result.age == 30.0
        assert result.sex == Sex.FEMALE
        assert result.species == Species.HOMO_SAPIENS

    def test_mixed_parameter_types(self):
        result = SubjectDescription(
            id="s_1",
            age=25.5,
            sex="F",
            species=Species.MACACA_MULATTA,
        )

        assert result.id == "s_1"
        assert result.age == 25.5
        assert result.sex == Sex.FEMALE
        assert result.species == Species.MACACA_MULATTA

    def test_invalid_inputs_fallback_to_defaults(self):
        result = SubjectDescription(
            id="subject_1",
            age="not_a_number",
            sex="invalid",
            species="invalid",
        )

        assert result.id == "subject_1"
        assert result.age == 0.0
        assert result.sex == Sex.UNKNOWN
        assert result.species == Species.UNKNOWN

    def test_edge_case_empty_strings(self):
        result = SubjectDescription(
            id="subject_1",
            age="",
            sex="",
            species="",
        )

        assert result.id == "subject_1"
        assert result.age == 0.0
        assert result.sex == Sex.UNKNOWN
        assert result.species == Species.UNKNOWN

    def test_zero_age(self):
        result = SubjectDescription(id="subject_1", species=None, age=0)
        assert result.age == 0.0

    def test_negative_age(self):
        with pytest.raises(ValueError, match="Age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5)

    def test_negative_age_float(self):
        with pytest.raises(ValueError, match="Age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5.5)

    def test_negative_age_string(self):
        with pytest.raises(ValueError, match="Age cannot be negative"):
            SubjectDescription(id="subject_1", age="-10")

    def test_age_with_unexpected_type(self):
        with pytest.raises(
            Exception, match="Age must be a float, int, numeric string, or None"
        ):
            SubjectDescription(id="subject_1", age=[])
