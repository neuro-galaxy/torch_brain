import numpy as np
import pytest

from temporaldata import Data, ArrayDict, IrregularTimeSeries
from torch_brain.transforms.region_selection import RandomRegionSelection


def test_basic_region_selection():
    """Test basic region selection functionality."""
    # Create test data with multiple regions
    timestamps = np.arange(100)
    unit_index = np.array([0] * 30 + [1] * 20 + [2] * 25 + [3] * 25)  # 4 units
    types = np.zeros(100)

    # Create regions: CA1, CA3, DG, void
    regions = np.array(["CA1", "CA3", "DG", "void"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2", "unit_3"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with default settings (should exclude "void")
    transform = RandomRegionSelection(seed=42)
    data_t = transform(data)

    # Check that only one region was selected (excluding "void")
    selected_region = data_t.units.region[0]
    assert selected_region in ["CA1", "CA3", "DG"]
    assert selected_region != "void"

    # Check that all units in the result have the same region
    assert np.all(data_t.units.region == selected_region)

    # Check that spikes correspond to the selected region
    selected_unit_indices = np.where(regions == selected_region)[0]
    assert len(data_t.units.id) == len(selected_unit_indices)

    # Check that all spikes belong to units in the selected region
    for spike_unit_idx in data_t.spikes.unit_index:
        assert spike_unit_idx < len(data_t.units.id)


def test_min_units_filtering():
    """Test that regions with insufficient units are filtered out."""
    timestamps = np.arange(50)
    unit_index = np.array([0] * 5 + [1] * 15 + [2] * 10 + [3] * 20)  # 4 units
    types = np.zeros(50)

    # Create regions with different unit counts per region
    regions = np.array(["small", "medium", "medium", "large"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2", "unit_3"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with min_units=2 (should exclude "small" and "large" regions with 1 unit each)
    transform = RandomRegionSelection(min_units=2, seed=42)
    data_t = transform(data)

    selected_region = data_t.units.region[0]
    assert selected_region == "medium"  # Only medium has 2 units
    assert selected_region not in ["small", "large"]


def test_exclude_regions():
    """Test that specified regions are excluded from selection."""
    timestamps = np.arange(60)
    unit_index = np.array([0] * 20 + [1] * 20 + [2] * 20)  # 3 units
    types = np.zeros(60)

    regions = np.array(["CA1", "CA3", "DG"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test excluding CA1 and CA3
    transform = RandomRegionSelection(exclude_regions=["CA1", "CA3"], seed=42)
    data_t = transform(data)

    selected_region = data_t.units.region[0]
    assert selected_region == "DG"
    assert selected_region not in ["CA1", "CA3"]


def test_no_available_regions_error():
    """Test that an error is raised when no regions meet the criteria."""
    timestamps = np.arange(30)
    unit_index = np.array([0] * 15 + [1] * 15)  # 2 units
    types = np.zeros(30)

    regions = np.array(["CA1", "CA3"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with min_units higher than any region has (each region has 1 unit)
    transform = RandomRegionSelection(min_units=2, seed=42)

    with pytest.raises(ValueError, match="No regions have at least 2 units"):
        transform(data)


def test_reset_index():
    """Test that unit indices are properly relabeled when reset_index=True."""
    timestamps = np.arange(40)
    unit_index = np.array([0] * 20 + [1] * 20)  # 2 units
    types = np.zeros(40)

    regions = np.array(["CA1", "CA3"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with reset_index=True (default)
    transform = RandomRegionSelection(seed=42)
    data_t = transform(data)

    # Check that unit indices are consecutive starting from 0
    unique_unit_indices = np.unique(data_t.spikes.unit_index)
    assert len(unique_unit_indices) == len(data_t.units.id)
    assert np.all(unique_unit_indices == np.arange(len(data_t.units.id)))


def test_no_reset_index():
    """Test that unit indices are preserved when reset_index=False."""
    timestamps = np.arange(40)
    unit_index = np.array([0] * 20 + [1] * 20)  # 2 units
    types = np.zeros(40)

    regions = np.array(["CA1", "CA3"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with reset_index=False
    transform = RandomRegionSelection(reset_index=False, seed=42)
    data_t = transform(data)

    # Check that unit indices are preserved (should be 0 for the selected region)
    assert np.all(data_t.spikes.unit_index == 0)


def test_unsupported_data_type():
    """Test that an error is raised for unsupported data types."""
    # Create data with RegularTimeSeries instead of IrregularTimeSeries
    data = Data(
        spikes=np.random.randn(100, 10),  # This would be RegularTimeSeries
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1"]),
            region=np.array(["CA1", "CA3"]),
        ),
        domain="auto",
    )

    transform = RandomRegionSelection()

    with pytest.raises(ValueError, match="Only IrregularTimeSeries is supported"):
        transform(data)


def test_deterministic_with_seed():
    """Test that the transform is deterministic when using the same seed."""
    timestamps = np.arange(60)
    unit_index = np.array([0] * 20 + [1] * 20 + [2] * 20)  # 3 units
    types = np.zeros(60)

    regions = np.array(["CA1", "CA3", "DG"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test with same seed
    transform1 = RandomRegionSelection(seed=123)
    transform2 = RandomRegionSelection(seed=123)

    data_t1 = transform1(data)
    data_t2 = transform2(data)

    # Should select the same region
    assert data_t1.units.region[0] == data_t2.units.region[0]

    # Test with different seeds
    transform3 = RandomRegionSelection(seed=456)
    data_t3 = transform3(data)

    # May or may not select the same region, but should be valid
    assert data_t3.units.region[0] in ["CA1", "CA3", "DG"]


def test_byte_string_regions():
    """Test that byte string regions are handled correctly."""
    timestamps = np.arange(60)
    unit_index = np.array([0] * 20 + [1] * 20 + [2] * 20)  # 3 units
    types = np.zeros(60)

    # Create regions as byte strings
    regions = np.array([b"CA1", b"CA3", b"DG"])

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["unit_0", "unit_1", "unit_2"]),
            region=regions,
        ),
        domain="auto",
    )

    # Test excluding regions using regular strings
    transform = RandomRegionSelection(exclude_regions=["CA1", "CA3"], seed=42)
    data_t = transform(data)

    selected_region = data_t.units.region[0]
    # The selected region should be the byte string b"DG"
    assert selected_region == b"DG"
    assert selected_region not in [b"CA1", b"CA3"]

    # Test that the region comparison works correctly
    assert np.all(data_t.units.region == b"DG")
