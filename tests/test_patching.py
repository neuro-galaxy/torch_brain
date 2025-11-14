import numpy as np
import pytest

from temporaldata import (
    Data,
    ArrayDict,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)
from torch_brain.transforms.patching import RegularPatching


@pytest.fixture
def basic_regular_data():
    """Create basic RegularTimeSeries data: 100 samples at 10 Hz = 10 seconds."""
    sampling_rate = 10.0
    num_samples = 100
    num_channels = 3
    
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, num_channels),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    return data


@pytest.fixture
def irregular_and_regular_data():
    """Create data with both IrregularTimeSeries and RegularTimeSeries."""
    # Regular time series: 100 samples at 10 Hz
    lfp = RegularTimeSeries(
        sampling_rate=10.0,
        data=np.random.randn(100, 3),
        domain="auto",
    )
    
    # Irregular time series (spikes)
    timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=np.array([0, 0, 1, 1, 0, 1, 2, 0, 1, 2]),
        domain="auto",
    )
    
    # Combine into Data object
    data = Data(
        lfp=lfp,
        spikes=spikes,
        units=ArrayDict(id=np.array(["a", "b", "c"])),
        session_id="test_session",
        domain=lfp.domain,
    )
    return data


def test_basic_patching_non_overlapping_exact_divisibility(basic_regular_data):
    """Test basic patching with non-overlapping patches that divide evenly.
    
    100 samples at 10 Hz, patch_duration=1.0, stride=1.0
    Expected: 10 patches of 10 samples each
    Shape: (100, 3) -> (10, 3, 10)
    """
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride, timestamp_mode="start")
    
    patched_data = transform(basic_regular_data)
    
    # Verify shape: (num_patches, channels, patch_samples)
    assert patched_data.lfp.data.shape == (10, 3, 10)
    
    # Verify sampling rate is now patches per second (1/stride)
    assert patched_data.lfp.sampling_rate == 1.0 / stride
    
    # Verify timestamps: should be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_timestamps = np.arange(10) * stride
    np.testing.assert_array_almost_equal(patched_data.lfp.timestamps, expected_timestamps)
    
    # Verify domain reflects new time structure
    assert patched_data.lfp.domain.start[0] == 0.0
    assert patched_data.lfp.domain.end[-1] == 9.0  # Last timestamp at 9.0


def test_patching_with_padding():
    """Test patching when data doesn't divide evenly - requires padding.
    
    95 samples at 10 Hz, patch_duration=1.0, stride=1.0
    Expected: 10 patches (last patch padded with 5 zeros)
    """
    sampling_rate = 10.0
    num_samples = 95  # Not divisible by patch size
    
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.ones((num_samples, 2)),  # Use ones to easily verify padding
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Should create 10 patches with shape (num_patches, channels, patch_samples)
    assert patched_data.lfp.data.shape == (10, 2, 10)
    
    # Last patch should have 5 zeros at the end (samples 90-94 are data, 95-99 are padding)
    last_patch = patched_data.lfp.data[-1]  # Shape: (2, 10)
    
    # First 5 samples should be ones (for both channels)
    np.testing.assert_array_equal(last_patch[:, :5], np.ones((2, 5)))
    
    # Last 5 samples should be zeros (padding)
    np.testing.assert_array_equal(last_patch[:, 5:], np.zeros((2, 5)))


def test_overlapping_patches():
    """Test patching with overlapping patches (stride < patch_duration).
    
    100 samples at 10 Hz, patch_duration=2.0, stride=1.0
    Expected: overlapping patches with 1 second overlap
    """
    sampling_rate = 10.0
    num_samples = 100
    
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 3),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 2.0  # 20 samples per patch
    stride = 1.0  # 10 samples between patch starts
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Calculate expected number of patches
    # num_patches = ceil((100 - 20) / 10) + 1 = ceil(80/10) + 1 = 9
    expected_num_patches = 9
    assert patched_data.lfp.data.shape == (expected_num_patches, 3, 20)
    
    # Verify timestamps for overlapping patches
    expected_timestamps = np.arange(expected_num_patches) * stride
    np.testing.assert_array_almost_equal(patched_data.lfp.timestamps, expected_timestamps)


def test_timestamp_mode_start():
    """Test that timestamp_mode='start' places timestamps at patch starts."""
    sampling_rate = 10.0
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(100, 2),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride, timestamp_mode="start")
    
    patched_data = transform(data)
    
    # Timestamps should be at the start of each patch: [0, 1, 2, ..., 9]
    expected_timestamps = np.arange(10) * stride
    np.testing.assert_array_almost_equal(patched_data.lfp.timestamps, expected_timestamps)



def test_timestamp_mode_middle_even_patch_size():
    """Test middle timestamp mode with even patch size.
    
    For even patch sizes, middle should be computed correctly.
    E.g., patch with 10 samples (0-9), middle is at index 4.5, 
    which corresponds to time (4 + 5) / 2 / sampling_rate from patch start.
    """
    sampling_rate = 10.0
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(100, 2),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 1.0  # 10 samples (even)
    stride = 0.5  # 5 samples
    transform = RegularPatching(patch_duration=patch_duration, stride=stride, timestamp_mode="middle")
    
    patched_data = transform(data)
    
    # Middle of 1.0 second patch is at 0.5 seconds from patch start
    # Patches start at: 0, 0.5, 1.0, 1.5, ..., 9.5
    # Middle timestamps: 0.5, 1.0, 1.5, 2.0, ..., 10.0
    num_patches = patched_data.lfp.data.shape[0]
    expected_timestamps = np.arange(num_patches) * stride + patch_duration / 2
    np.testing.assert_array_almost_equal(patched_data.lfp.timestamps, expected_timestamps)


def test_multiple_regular_time_series():
    """Test patching with multiple RegularTimeSeries objects in Data."""
    sampling_rate = 10.0
    num_samples = 100
    
    lfp1 = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 3),
        domain="auto",
    )
    
    lfp2 = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 5),
        domain="auto",
    )
    
    data = Data(lfp1=lfp1, lfp2=lfp2, domain=lfp1.domain)
    
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Both should be patched consistently
    assert patched_data.lfp1.data.shape == (10, 3, 10)
    assert patched_data.lfp2.data.shape == (10, 5, 10)
    
    # Both should have same timestamps
    np.testing.assert_array_equal(patched_data.lfp1.timestamps, patched_data.lfp2.timestamps)


def test_mixed_data_types(irregular_and_regular_data):
    """Test patching preserves non-RegularTimeSeries data correctly.
    
    IrregularTimeSeries should be left unchanged.
    ArrayDict and scalars should be preserved.
    """
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(irregular_and_regular_data)
    
    # RegularTimeSeries should be patched
    assert patched_data.lfp.data.shape == (10, 3, 10)
    
    # IrregularTimeSeries should be unchanged
    assert hasattr(patched_data, "spikes")
    np.testing.assert_array_equal(
        patched_data.spikes.timestamps,
        irregular_and_regular_data.spikes.timestamps
    )
    np.testing.assert_array_equal(
        patched_data.spikes.unit_index,
        irregular_and_regular_data.spikes.unit_index
    )
    
    # ArrayDict should be preserved
    assert hasattr(patched_data, "units")
    np.testing.assert_array_equal(patched_data.units.id, irregular_and_regular_data.units.id)
    
    # Scalar should be preserved
    assert patched_data.session_id == irregular_and_regular_data.session_id


def test_domain_consistency():
    """Test that domain is updated correctly to reflect patched structure."""
    sampling_rate = 10.0
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(100, 2),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Domain should reflect new time structure
    # 10 patches with timestamps [0, 1, 2, ..., 9]
    assert patched_data.domain.start[0] == 0.0
    assert patched_data.domain.end[-1] == 9.0  # Last timestamp at 9.0


def test_single_patch_short_data():
    """Test edge case: data shorter than patch_duration creates single patch with padding."""
    sampling_rate = 10.0
    num_samples = 5  # Only 0.5 seconds of data
    
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.ones((num_samples, 2)),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 1.0  # Longer than data
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Should create 1 patch with padding, shape (num_patches, channels, patch_samples)
    assert patched_data.lfp.data.shape == (1, 2, 10)
    
    # First 5 samples should be data (ones), last 5 should be padding (zeros)
    single_patch = patched_data.lfp.data[0]  # Shape: (2, 10)
    np.testing.assert_array_equal(single_patch[:, :5], np.ones((2, 5)))
    np.testing.assert_array_equal(single_patch[:, 5:], np.zeros((2, 5)))


def test_multidimensional_data():
    """Test patching with multi-dimensional data: (time, channels, height, width)."""
    sampling_rate = 10.0
    num_samples = 100
    num_channels = 3
    height = 8
    width = 8
    
    # Create 4D data: (time, channels, height, width)
    eeg = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, num_channels, height, width),
        domain="auto",
    )
    data = Data(eeg=eeg, domain=eeg.domain)
    
    patch_duration = 1.0
    stride = 1.0
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Shape should be (num_patches, channels, patch_samples, height, width)
    assert patched_data.eeg.data.shape == (10, 3, 10, 8, 8)


def test_very_small_stride_heavy_overlap():
    """Test edge case with very small stride causing heavy overlap."""
    sampling_rate = 10.0
    num_samples = 100
    
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(num_samples, 2),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    
    patch_duration = 2.0  # 20 samples
    stride = 0.1  # 1 sample stride - very heavy overlap
    transform = RegularPatching(patch_duration=patch_duration, stride=stride)
    
    patched_data = transform(data)
    
    # Calculate expected number of patches: ceil((100 - 20) / 1) + 1 = 81
    expected_num_patches = 81
    assert patched_data.lfp.data.shape == (expected_num_patches, 2, 20)
    
    # Verify timestamps
    expected_timestamps = np.arange(expected_num_patches) * stride
    np.testing.assert_array_almost_equal(patched_data.lfp.timestamps, expected_timestamps)


def test_invalid_timestamp_mode():
    """Test that invalid timestamp_mode raises ValueError."""
    with pytest.raises(ValueError, match="timestamp_mode must be"):
        RegularPatching(patch_duration=1.0, timestamp_mode="invalid")


def test_data_without_domain():
    """Test that Data without domain raises appropriate error."""
    data = Data(value=42)  # No time-based attributes
    
    transform = RegularPatching(patch_duration=1.0)
    
    with pytest.raises(ValueError, match="must have a domain"):
        transform(data)


def test_preserves_absolute_start():
    """Test that _absolute_start is preserved through patching."""
    sampling_rate = 10.0
    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        data=np.random.randn(100, 2),
        domain="auto",
    )
    data = Data(lfp=lfp, domain=lfp.domain)
    data._absolute_start = 5.0  # Simulate a sliced data object
    
    transform = RegularPatching(patch_duration=1.0, stride=1.0)
    patched_data = transform(data)
    
    # _absolute_start should be preserved
    assert patched_data._absolute_start == 5.0
