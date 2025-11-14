import numpy as np
import pytest

from temporaldata import (
    Data,
    ArrayDict,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)
from torch_brain.transforms.patching import Patching


@pytest.fixture
def irregular_data():
    """Create test data with IrregularTimeSeries (spikes)."""
    timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    unit_index = np.array([0, 0, 1, 1, 0, 1, 2, 0, 1, 2])
    types = np.zeros(10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            types=types,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(["a", "b", "c"]),
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    return data


@pytest.fixture
def regular_data():
    """Create test data with RegularTimeSeries (LFP)."""
    sampling_rate = 10.0  # 10 Hz
    num_samples = 100
    lfps = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 3),  # 3 channels
        domain="auto",
    )
    data = Data(lfps=lfps, domain=lfps.domain)
    return data


def test_basic_patching_irregular_exact_match(irregular_data):
    """Test patching with IrregularTimeSeries when patch duration equals data duration."""
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(irregular_data)
    
    # Should create exactly one patch
    assert len(patched_data.spikes.timestamps) == 1
    assert patched_data.spikes.timestamps[0] == 0.0  # First timestamp mode
    
    # Domain should be updated
    assert patched_data.domain.start[0] == 0.0
    assert patched_data.domain.end[-1] == patch_duration


def test_basic_patching_regular_exact_match(regular_data):
    """Test patching with RegularTimeSeries when patch duration equals data duration."""
    # regular_data has 100 samples at 10 Hz = 10 seconds
    patch_duration = 10.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(regular_data)
    
    # Should create exactly one patch
    # Shape should be (num_patches, channels, samples_per_patch)
    assert patched_data.lfps.data.shape[0] == 1  # num_patches
    assert patched_data.lfps.data.shape[1] == 3  # channels preserved
    assert patched_data.lfps.data.shape[2] == int(patch_duration * regular_data.lfps.sampling_rate)  # samples_per_patch
    
    # Sampling rate should be maintained
    assert patched_data.lfps.sampling_rate == regular_data.lfps.sampling_rate


def test_error_when_patch_shorter_than_data_irregular():
    """Test that error is raised when patch duration is shorter than data duration."""
    # Data spans 1.0 second
    timestamps = np.linspace(0.0, 1.0, 10)
    unit_index = np.arange(10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    
    patch_duration = 0.3  # Shorter than data duration (1.0)
    transform = Patching(patch_duration=patch_duration)
    
    with pytest.raises(ValueError, match="patch duration.*shorter than data duration"):
        transform(data)


def test_error_when_patch_shorter_than_data_regular():
    """Test that error is raised when patch duration is shorter than data duration for RegularTimeSeries."""
    sampling_rate = 10.0
    num_samples = 100  # 10 seconds of data
    lfps = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 2),
        domain="auto",
    )
    data = Data(lfps=lfps, domain=lfps.domain)
    
    patch_duration = 5.0  # Shorter than data duration (10.0)
    transform = Patching(patch_duration=patch_duration)
    
    with pytest.raises(ValueError, match="patch duration.*shorter than data duration"):
        transform(data)


def test_padding_non_divisible_regular():
    """Test zero-padding for RegularTimeSeries when patch duration doesn't divide evenly."""
    sampling_rate = 10.0
    num_samples = 23  # 2.3 seconds of data
    lfps = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 2),
        domain="auto",
    )
    data = Data(lfps=lfps, domain=lfps.domain)
    
    patch_duration = 2.5  # Longer than data duration, doesn't divide evenly
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(data)
    
    # Should create one patch with padding
    # Shape should be (num_patches, channels, samples_per_patch)
    expected_samples_per_patch = int(patch_duration * sampling_rate)
    assert patched_data.lfps.data.shape[0] == 1  # num_patches
    assert patched_data.lfps.data.shape[1] == 2  # channels
    assert patched_data.lfps.data.shape[2] == expected_samples_per_patch  # samples_per_patch
    
    # Last samples should be padded with zeros
    # Shape is (1, 2, 25), so we check the last dimension (samples_per_patch)
    patch = patched_data.lfps.data[0]  # Shape: (2, 25)
    # Original had 23 samples, patch needs 25, so last 2 samples should be zeros
    # Check each channel's last 2 samples
    assert np.allclose(patch[:, 23:], 0.0)


def test_multiple_time_series_fields():
    """Test patching with multiple time-series fields."""
    timestamps = np.linspace(0.0, 1.0, 10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=np.arange(10) % 2,
            domain="auto",
        ),
        lfp=RegularTimeSeries(
            sampling_rate=10.0,
            data=np.random.randn(10, 3),
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(data)
    
    # Both fields should be patched
    assert hasattr(patched_data, "spikes")
    assert hasattr(patched_data, "lfp")
    
    # Both should have the same number of patches (1)
    num_patches_spikes = len(patched_data.spikes.timestamps)
    num_patches_lfp = patched_data.lfp.data.shape[0]
    assert num_patches_spikes == num_patches_lfp == 1


def test_timestamp_mode_first():
    """Test timestamp_mode='first' (default)."""
    timestamps = np.linspace(0.0, 1.0, 10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=np.arange(10),
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration, timestamp_mode="first")
    patched_data = transform(data)
    
    # Timestamp should be at the start of the patch
    assert len(patched_data.spikes.timestamps) == 1
    assert patched_data.spikes.timestamps[0] == 0.0


def test_timestamp_mode_middle():
    """Test timestamp_mode='middle'."""
    timestamps = np.linspace(0.0, 1.0, 10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=np.arange(10),
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration, timestamp_mode="middle")
    patched_data = transform(data)
    
    # Timestamp should be at the middle of the patch
    assert len(patched_data.spikes.timestamps) == 1
    assert patched_data.spikes.timestamps[0] == 0.5  # 0.0 + 1.0/2


def test_timestamp_mode_last():
    """Test timestamp_mode='last'."""
    timestamps = np.linspace(0.0, 1.0, 10)
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=np.arange(10),
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([1.0])),
    )
    
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration, timestamp_mode="last")
    patched_data = transform(data)
    
    # Timestamp should be at the end of the patch
    assert len(patched_data.spikes.timestamps) == 1
    assert patched_data.spikes.timestamps[0] == 1.0  # 0.0 + 1.0


def test_preserve_non_temporal_fields(irregular_data):
    """Test that non-temporal fields (ArrayDict, scalars) are preserved."""
    # Add a scalar field
    irregular_data.session_id = "test_session"
    
    patch_duration = 1.0  # Exactly matches data duration
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(irregular_data)
    
    # Non-temporal fields should be preserved
    assert hasattr(patched_data, "units")
    assert hasattr(patched_data, "session_id")
    assert patched_data.session_id == "test_session"
    assert np.array_equal(patched_data.units.id, irregular_data.units.id)


def test_empty_data():
    """Test edge case with empty data."""
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([]),
            unit_index=np.array([]),
            domain=Interval(start=np.array([0.0]), end=np.array([0.0])),
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([0.0])),
    )
    
    patch_duration = 0.3
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(data)
    
    # Should handle empty data gracefully
    # Might create one patch with all zeros
    assert hasattr(patched_data, "spikes")


def test_data_shorter_than_patch_size():
    """Test edge case when data is shorter than patch size."""
    # Very short data: only 0.05 seconds
    timestamps = np.array([0.01, 0.02, 0.03])
    unit_index = np.array([0, 1, 0])
    
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain="auto",
        ),
        domain=Interval(start=np.array([0.0]), end=np.array([0.05])),
    )
    
    patch_duration = 0.3  # Much longer than data
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(data)
    
    # Should create at least one patch with padding
    assert len(patched_data.spikes.timestamps) >= 1


def test_regular_time_series_padding_consistency():
    """Test that RegularTimeSeries patches have consistent dimensions."""
    sampling_rate = 10.0
    num_samples = 17  # 1.7 seconds of data
    lfps = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 4),
        domain="auto",
    )
    data = Data(lfps=lfps, domain=lfps.domain)
    
    patch_duration = 2.0  # Longer than data duration, doesn't divide evenly
    transform = Patching(patch_duration=patch_duration)
    patched_data = transform(data)
    
    # Should create one patch with consistent dimensions
    # Shape should be (num_patches, channels, samples_per_patch)
    expected_samples_per_patch = int(patch_duration * sampling_rate)
    assert patched_data.lfps.data.shape[0] == 1  # num_patches
    assert patched_data.lfps.data.shape[1] == 4  # Channels preserved
    assert patched_data.lfps.data.shape[2] == expected_samples_per_patch  # samples_per_patch

