import pytest
import inspect
from brainsets.core import StringIntEnum
import brainsets.taxonomy


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
    duplicates = {}
    for value, names in value_to_names.items():
        if len(names) > 1:
            # Check if names are allowed aliases (e.g., A = SOME_ALIAS = 0)
            if len(set(getattr(enum, name) for name in names)) > 1:
                duplicates[value] = names

    return duplicates


def get_all_stringintenum_subclasses():
    classes = []
    for module_name, module in inspect.getmembers(brainsets.taxonomy, inspect.ismodule):
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, StringIntEnum)
                and obj != StringIntEnum
            ):
                classes.append((name, obj))
    return classes


def test_all_stringintenum_classes_have_no_duplicate_ids():
    for name, cls in get_all_stringintenum_subclasses():
        duplicates = find_duplicates(cls)
        assert len(duplicates) == 0, f"Duplicate IDs found in {name} enum: {duplicates}"
