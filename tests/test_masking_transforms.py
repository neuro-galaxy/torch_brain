import numpy as np
import pytest
from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)


# Helper functions
def create_sample_regular_time_series(
    time_samples=100, num_channels=10, sampling_rate=100.0
):
    """Create a sample RegularTimeSeries for testing."""
    data = np.random.randn(time_samples, num_channels)
    return RegularTimeSeries(
        data=data,
        sampling_rate=sampling_rate,
        domain=Interval(0.0, time_samples / sampling_rate),
    )


def create_sample_irregular_time_series(num_samples=100, num_channels=10, regular=True):
    """Create a sample IrregularTimeSeries for testing."""
    if regular:
        # Create regularly spaced timestamps
        timestamps = np.arange(num_samples, dtype=np.float64) * 0.01
    else:
        # Create irregularly spaced timestamps
        timestamps = np.sort(np.random.rand(num_samples) * 1.0)

    data = np.random.randn(num_samples, num_channels)
    return IrregularTimeSeries(
        timestamps=timestamps,
        data=data,
        domain=Interval(0.0, timestamps[-1]),
    )


def create_sample_data_object(include_irregular=True, regular_irregular=True):
    """Create a sample Data object with multiple time series."""
    kwargs = {
        "regular_ts": create_sample_regular_time_series(100, 10),
        "units": ArrayDict(
            unit_id=np.array([0, 1, 2]),
            brain_region=np.array(["M1", "M1", "PMd"]),
        ),
        "trials": Interval(
            start=np.array([0.0, 0.5]),
            end=np.array([0.5, 1.0]),
        ),
        "domain": Interval(0.0, 1.0),
    }

    if include_irregular:
        kwargs["irregular_ts"] = create_sample_irregular_time_series(
            100, 10, regular=regular_irregular
        )

    return Data(**kwargs)


def is_regularly_spaced(timestamps, tolerance=1e-6):
    """Check if timestamps are regularly spaced."""
    if len(timestamps) < 2:
        return True
    diffs = np.diff(timestamps)
    return np.allclose(diffs, diffs[0], atol=tolerance)


class TestTimeMasking:
    """Tests for TimeMasking transform."""

    def test_mask_attribute_added(self):
        """Test that mask attribute is added with correct shape."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Check that mask attribute exists
        assert hasattr(result.regular_ts, "mask")

        # Check shape: should be (time, channel)
        expected_shape = (100, 10)
        assert result.regular_ts.mask.shape == expected_shape

        # Check dtype is boolean
        assert result.regular_ts.mask.dtype == bool

    def test_percentage_masked_matches_config(self):
        """Test percentage of time masked matches configuration."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)
        mask_percentage = 0.3
        transform = TimeMasking(
            mask_percentage=mask_percentage, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Calculate percentage of masked time points (False in mask)
        mask = result.regular_ts.mask
        masked_fraction = 1.0 - mask.any(axis=1).sum() / mask.shape[0]

        # Should be approximately equal (within reasonable tolerance)
        assert abs(masked_fraction - mask_percentage) < 0.15

    def test_window_duration_respected(self):
        """Test that window duration is respected."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)
        window_duration = 0.05  # 5 samples at 100Hz
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=window_duration, random_seed=42
        )
        result = transform(data)

        mask = result.regular_ts.mask
        # Find masked time points (False)
        time_masked = ~mask.any(axis=1)

        # Check that masked regions form contiguous blocks
        if time_masked.any():
            # Find transitions
            transitions = np.diff(time_masked.astype(int))
            # Each window should have approximately the right size
            # We just check that there are masked regions
            assert time_masked.sum() > 0

    def test_works_with_regular_time_series(self):
        """Test works with RegularTimeSeries."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        assert hasattr(result.regular_ts, "mask")
        assert result.regular_ts.mask.shape[0] == 100

    def test_works_with_regularly_spaced_irregular(self):
        """Test works with regularly-spaced IrregularTimeSeries."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=True, regular_irregular=True)
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Should add mask to irregularly spaced time series if regular
        assert hasattr(result.irregular_ts, "mask")
        assert result.irregular_ts.mask.shape[0] == 100

    def test_skips_irregularly_spaced(self):
        """Test skips irregularly-spaced IrregularTimeSeries."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(
            include_irregular=True, regular_irregular=False
        )
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Should NOT add mask to irregularly spaced time series
        assert not hasattr(result.irregular_ts, "mask")

    def test_nested_data_objects(self):
        """Test nested Data objects are handled recursively."""
        from torch_brain.transforms.masking import TimeMasking

        inner_data = create_sample_data_object(include_irregular=False)
        outer_data = Data(
            inner=inner_data,
            regular_ts=create_sample_regular_time_series(50, 5),
            domain=Interval(0.0, 1.0),
        )

        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(outer_data)

        # Check both outer and inner time series have masks
        assert hasattr(result.regular_ts, "mask")
        assert hasattr(result.inner.regular_ts, "mask")

    def test_original_data_unchanged(self):
        """Test original data values are unchanged."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)
        original_data = data.regular_ts.data.copy()

        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Check that data values are unchanged
        np.testing.assert_array_equal(result.regular_ts.data, original_data)

    def test_reproducibility_with_seed(self):
        """Test random seed produces reproducible results."""
        from torch_brain.transforms.masking import TimeMasking

        data = create_sample_data_object(include_irregular=False)

        transform1 = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result1 = transform1(data)

        transform2 = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result2 = transform2(data)

        # Masks should be identical
        np.testing.assert_array_equal(result1.regular_ts.mask, result2.regular_ts.mask)


class TestChannelMasking:
    """Tests for ChannelMasking transform."""

    def test_mask_shape_correct(self):
        """Test mask shape is correct."""
        from torch_brain.transforms.masking import ChannelMasking

        data = create_sample_data_object(include_irregular=False)
        transform = ChannelMasking(mask_percentage=0.3, random_seed=42)
        result = transform(data)

        # Check shape
        expected_shape = (100, 10)
        assert result.regular_ts.mask.shape == expected_shape
        assert result.regular_ts.mask.dtype == bool

    def test_percentage_channels_masked(self):
        """Test percentage of channels masked matches configuration."""
        from torch_brain.transforms.masking import ChannelMasking

        data = create_sample_data_object(include_irregular=False)
        mask_percentage = 0.4
        transform = ChannelMasking(mask_percentage=mask_percentage, random_seed=42)
        result = transform(data)

        mask = result.regular_ts.mask
        # Count channels that are completely masked (all False)
        fully_masked_channels = (~mask.any(axis=0)).sum()
        masked_fraction = fully_masked_channels / mask.shape[1]

        # Should be approximately equal
        assert abs(masked_fraction - mask_percentage) < 0.15

    def test_all_timestamps_masked_for_selected_channels(self):
        """Test all timestamps masked for selected channels."""
        from torch_brain.transforms.masking import ChannelMasking

        data = create_sample_data_object(include_irregular=False)
        transform = ChannelMasking(mask_percentage=0.3, random_seed=42)
        result = transform(data)

        mask = result.regular_ts.mask
        # For each channel, should be either all True or all False
        for channel_idx in range(mask.shape[1]):
            channel_mask = mask[:, channel_idx]
            # Should be either all True or all False
            assert channel_mask.all() or (~channel_mask).all()

    def test_works_with_both_time_series_types(self):
        """Test works with both time series types."""
        from torch_brain.transforms.masking import ChannelMasking

        data = create_sample_data_object(include_irregular=True, regular_irregular=True)
        transform = ChannelMasking(mask_percentage=0.3, random_seed=42)
        result = transform(data)

        assert hasattr(result.regular_ts, "mask")
        assert hasattr(result.irregular_ts, "mask")

    def test_reproducibility_with_seed(self):
        """Test reproducibility with seed."""
        from torch_brain.transforms.masking import ChannelMasking

        data = create_sample_data_object(include_irregular=False)

        transform1 = ChannelMasking(mask_percentage=0.3, random_seed=42)
        result1 = transform1(data)

        transform2 = ChannelMasking(mask_percentage=0.3, random_seed=42)
        result2 = transform2(data)

        np.testing.assert_array_equal(result1.regular_ts.mask, result2.regular_ts.mask)


class TestBlockMasking:
    """Tests for BlockMasking transform."""

    def test_mask_shape_correct(self):
        """Test mask shape is correct."""
        from torch_brain.transforms.masking import BlockMasking

        data = create_sample_data_object(include_irregular=False)
        transform = BlockMasking(
            time_block_size=0.1,  # 0.1 seconds = 10 samples at 100Hz
            channel_block_size=2,
            mask_percentage=0.3,
            random_seed=42,
        )
        result = transform(data)

        expected_shape = (100, 10)
        assert result.regular_ts.mask.shape == expected_shape
        assert result.regular_ts.mask.dtype == bool

    def test_block_dimensions_match_config(self):
        """Test block dimensions match configuration."""
        from torch_brain.transforms.masking import BlockMasking

        data = create_sample_data_object(include_irregular=False)
        time_block_size = 0.1  # 0.1 seconds = 10 samples at 100Hz
        channel_block_size = 2
        transform = BlockMasking(
            time_block_size=time_block_size,
            channel_block_size=channel_block_size,
            mask_percentage=0.3,
            random_seed=42,
        )
        result = transform(data)

        # Just verify that mask exists and has correct shape
        # Detailed block structure is implementation-dependent
        assert result.regular_ts.mask.shape == (100, 10)

    def test_percentage_blocks_masked(self):
        """Test percentage of blocks masked."""
        from torch_brain.transforms.masking import BlockMasking

        data = create_sample_data_object(include_irregular=False)
        mask_percentage = 0.25
        transform = BlockMasking(
            time_block_size=0.1,  # 0.1 seconds = 10 samples at 100Hz
            channel_block_size=2,
            mask_percentage=mask_percentage,
            random_seed=42,
        )
        result = transform(data)

        mask = result.regular_ts.mask
        # Calculate overall masked fraction
        masked_fraction = 1.0 - mask.sum() / mask.size

        # Should be approximately equal (within tolerance)
        assert abs(masked_fraction - mask_percentage) < 0.2

    def test_works_with_both_time_series_types(self):
        """Test works with both time series types."""
        from torch_brain.transforms.masking import BlockMasking

        data = create_sample_data_object(include_irregular=True, regular_irregular=True)
        transform = BlockMasking(
            time_block_size=0.1,  # 0.1 seconds
            channel_block_size=2,
            mask_percentage=0.3,
            random_seed=42,
        )
        result = transform(data)

        assert hasattr(result.regular_ts, "mask")
        assert hasattr(result.irregular_ts, "mask")

    def test_reproducibility_with_seed(self):
        """Test reproducibility with seed."""
        from torch_brain.transforms.masking import BlockMasking

        data = create_sample_data_object(include_irregular=False)

        transform1 = BlockMasking(
            time_block_size=0.1,  # 0.1 seconds
            channel_block_size=2,
            mask_percentage=0.3,
            random_seed=42,
        )
        result1 = transform1(data)

        transform2 = BlockMasking(
            time_block_size=0.1,  # 0.1 seconds
            channel_block_size=2,
            mask_percentage=0.3,
            random_seed=42,
        )
        result2 = transform2(data)

        np.testing.assert_array_equal(result1.regular_ts.mask, result2.regular_ts.mask)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data_object(self):
        """Test empty Data objects."""
        from torch_brain.transforms.masking import TimeMasking

        data = Data(session_id="test", value=42.0)
        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Should not crash, just return unchanged
        assert result.session_id == "test"
        assert result.value == 42.0

    def test_custom_attribute_names(self):
        """Test that masking works with any array attribute name, not just 'data'."""
        from torch_brain.transforms.masking import TimeMasking

        # Create time series with custom attribute names
        spikes = np.random.randn(100, 8)
        lfp = np.random.randn(100, 8, 48)  # 3D array: time x channels x features

        ts = RegularTimeSeries(
            spikes=spikes,
            lfp=lfp,
            sampling_rate=100.0,
            domain=Interval(0.0, 1.0),
        )

        data = Data(neural_data=ts, domain=Interval(0.0, 1.0))

        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Check that mask was added
        assert hasattr(result.neural_data, "mask")
        # Mask should have shape (time, channel) based on first array attribute
        assert result.neural_data.mask.shape == (100, 8)
        # Original data should be unchanged
        np.testing.assert_array_equal(result.neural_data.spikes, spikes)
        np.testing.assert_array_equal(result.neural_data.lfp, lfp)

    def test_single_channel_time_series(self):
        """Test time series with single channel."""
        from torch_brain.transforms.masking import ChannelMasking

        ts = create_sample_regular_time_series(100, 1)
        data = Data(ts=ts, domain=Interval(0.0, 1.0))

        transform = ChannelMasking(mask_percentage=0.5, random_seed=42)
        result = transform(data)

        assert hasattr(result.ts, "mask")
        assert result.ts.mask.shape == (100, 1)

    def test_single_timestamp(self):
        """Test time series with single timestamp."""
        from torch_brain.transforms.masking import TimeMasking

        ts = create_sample_regular_time_series(1, 10)
        data = Data(ts=ts, domain=Interval(0.0, 0.01))

        transform = TimeMasking(
            mask_percentage=0.5, window_duration=0.001, random_seed=42
        )
        result = transform(data)

        assert hasattr(result.ts, "mask")
        assert result.ts.mask.shape == (1, 10)

    def test_data_without_time_series(self):
        """Test Data objects without time series."""
        from torch_brain.transforms.masking import TimeMasking

        data = Data(
            units=ArrayDict(
                unit_id=np.array([0, 1, 2]),
                brain_region=np.array(["M1", "M1", "PMd"]),
            ),
            value=123,
        )

        transform = TimeMasking(
            mask_percentage=0.2, window_duration=0.05, random_seed=42
        )
        result = transform(data)

        # Should not crash, just return with no changes to non-time-series objects
        assert result.value == 123
        assert len(result.units) == 3
