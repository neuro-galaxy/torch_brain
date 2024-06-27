import pytest
from brainsets.taxonomy import Decoder


def find_duplicates(enum):
    # Collect all enum names and their values
    enum_items = [(name, member.value) for name, member in enum.__members__.items()]

    # Find duplicates: create a dictionary where each value points to a list of names that share it
    value_to_names = {}
    for name, value in enum_items:
        if value in value_to_names:
            value_to_names[value].append(name)
        else:
            value_to_names[value] = [name]

    # Filter out entries with only one name, leaving only duplicates
    duplicates = {
        value: names for value, names in value_to_names.items() if len(names) > 1
    }
    return duplicates


def test_decoder_enum_has_no_duplicate_ids():
    duplicates = find_duplicates(Decoder)
    assert len(duplicates) == 0, f"Duplicate IDs found in Decoder enum: {duplicates}"
