import numpy as np
import pytest
import copy
from temporaldata import Data, ArrayDict, IrregularTimeSeries
from torch_brain.transforms.select_by_region import SelectByRegion, RandomSelectByRegion


def test_select_by_region():
    """Test SelectByRegion selects the correct region and units."""
    timestamps = np.arange(60, dtype=np.float64)
    unit_index = np.array([0] * 20 + [1] * 20 + [2] * 20)  # 3 units
    types = np.zeros(60)
    regions = np.array(["CA3", "DG", "CA3"])

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

    # Select region "CA3"
    transform = SelectByRegion(region="CA3")
    data_t = transform(copy.deepcopy(data))

    # Only twos units and region should remain
    assert len(data_t.units.id) == 2
    assert np.all(data_t.units.region == "CA3")
    # All spikes should have unit_index 0 or 1 (after reset_index)
    assert np.all(data_t.spikes.unit_index <= 1)
    # The number of spikes should match the original count for CA3
    assert len(data_t.spikes.timestamps) == 40

    # Select region "CA3" with reset_index=False
    transform = SelectByRegion(region="CA3", reset_index=False)
    data_t = transform(copy.deepcopy(data))

    # Units should not be reindexed, so there should still be 3 units
    assert len(data_t.units.id) == 3
    # But only the spikes from the selected region should remain
    assert np.all(data_t.spikes.unit_index != 1)

    # Try to select a region that does not exist
    transform = SelectByRegion(region="NOT_A_REGION")
    with pytest.raises(ValueError, match="No units found in region NOT_A_REGION"):
        transform(copy.deepcopy(data))


def test_random_select_by_region():
    """Test basic region selection functionality."""
    # Create test data with multiple regions
    timestamps = np.arange(100)
    unit_index = np.array([0] * 30 + [1] * 20 + [2] * 25 + [3] * 25)  # 4 units
    types = np.zeros(100)

    # Create regions: CA3, DG, void
    regions = np.array(["CA3", "DG", "CA3", "void"])

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
    transform = RandomSelectByRegion(seed=42, exclude_regions=["void"])

    # Sample multiple times randomly to check that void is never selected
    for _ in range(10):
        data_t = transform(copy.deepcopy(data))

        # Check that only one region was selected (excluding "void")
        selected_region = data_t.units.region[0]
        assert selected_region in ["CA3", "DG"]
        assert selected_region != "void"

    # Check that all units in the result have the same region
    assert np.all(data_t.units.region == selected_region)

    # Test with min_units=2
    transform = RandomSelectByRegion(min_units=2, seed=42)

    for _ in range(10):
        data_t = transform(copy.deepcopy(data))
        # Only CA3 can be selected with min_units=2
        selected_region = data_t.units.region[0]
        assert selected_region == "CA3"

    # Test with min_units higher than any region has
    transform = RandomSelectByRegion(min_units=5, seed=42)

    with pytest.raises(ValueError, match="No regions have at least 5 units"):
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
    transform1 = RandomSelectByRegion(seed=123)
    transform2 = RandomSelectByRegion(seed=123)

    data_t1 = transform1(data)
    data_t2 = transform2(data)

    # Should select the same region
    assert data_t1.units.region[0] == data_t2.units.region[0]

    # Test with different seeds
    transform3 = RandomSelectByRegion(seed=456)
    data_t3 = transform3(data)

    # May or may not select the same region, but should be valid
    assert data_t3.units.region[0] in ["CA1", "CA3", "DG"]
