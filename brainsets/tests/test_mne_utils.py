import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from brainsets.utils.mne_utils import (
        extract_measurement_date,
        extract_signal,
        extract_channels,
        concatenate_recordings,
    )
    from temporaldata import ArrayDict
except ImportError:
    extract_measurement_date = None
    extract_signal = None
    extract_channels = None
    concatenate_recordings = None
    ArrayDict = None


def create_mock_raw(
    n_channels=3,
    n_samples=1000,
    sfreq=256.0,
    meas_date=None,
    ch_names=None,
    ch_types=None,
):
    """Helper to create a mock MNE Raw object for testing."""
    mock_raw = MagicMock(spec=mne.io.Raw)

    if ch_names is None:
        ch_names = [f"CH{i}" for i in range(n_channels)]
    if ch_types is None:
        ch_types = ["eeg"] * n_channels

    mock_raw.info = {
        "sfreq": sfreq,
        "meas_date": meas_date,
        "bads": [],
    }
    mock_raw.ch_names = ch_names
    mock_raw.n_times = n_samples
    mock_raw.get_channel_types.return_value = ch_types
    mock_raw.get_montage.return_value = None

    mock_data = np.random.randn(n_channels, n_samples)
    mock_raw.get_data.return_value = mock_data

    return mock_raw


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractMeasurementDate:
    """Test extraction of measurement date from MNE Raw objects."""

    def test_returns_meas_date_when_present(self):
        """Test that the measurement date is returned when present."""
        expected_date = datetime.datetime(
            2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_measurement_date(mock_raw)
        assert result == expected_date

    def test_returns_unix_epoch_when_meas_date_none(self):
        """Test that Unix epoch is returned when measurement date is missing."""
        mock_raw = create_mock_raw(meas_date=None)
        with pytest.warns(UserWarning, match="No measurement date found"):
            result = extract_measurement_date(mock_raw)
        expected = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        assert result == expected

    def test_preserves_timezone_info(self):
        """Test that timezone information is preserved."""
        tz = datetime.timezone(datetime.timedelta(hours=5))
        expected_date = datetime.datetime(2023, 6, 15, 10, 30, 0, tzinfo=tz)
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_measurement_date(mock_raw)
        assert result.tzinfo == tz


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractSignal:
    """Test extraction of time-series signal from MNE Raw objects."""

    def test_returns_regular_time_series(self):
        """Test that a RegularTimeSeries object is returned."""
        mock_raw = create_mock_raw(n_channels=4, n_samples=500, sfreq=256.0)
        result = extract_signal(mock_raw)
        assert hasattr(result, "signal")
        assert hasattr(result, "sampling_rate")
        assert hasattr(result, "domain")

    def test_signal_shape_is_samples_by_channels(self):
        """Test that signal shape is (n_samples, n_channels)."""
        n_channels = 4
        n_samples = 500
        mock_raw = create_mock_raw(n_channels=n_channels, n_samples=n_samples)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (n_samples, n_channels)

    def test_sampling_rate_matches_mne_sfreq(self):
        """Test that sampling rate is correctly extracted."""
        sfreq = 512.0
        mock_raw = create_mock_raw(sfreq=sfreq)
        result = extract_signal(mock_raw)
        assert result.sampling_rate == sfreq

    def test_domain_start_is_zero(self):
        """Test that domain start is 0."""
        mock_raw = create_mock_raw()
        result = extract_signal(mock_raw)
        assert result.domain.start[0] == 0.0

    def test_domain_end_calculation(self):
        """Test that domain end is calculated as (n_samples - 1) / sfreq."""
        n_samples = 1000
        sfreq = 256.0
        mock_raw = create_mock_raw(n_samples=n_samples, sfreq=sfreq)
        result = extract_signal(mock_raw)
        expected_end = (n_samples - 1) / sfreq
        assert np.isclose(result.domain.end[0], expected_end)

    def test_raises_error_when_no_samples(self):
        """Test that ValueError is raised when recording contains no samples."""
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.get_data.return_value = np.array([]).reshape(3, 0)
        with pytest.raises(ValueError, match="Recording contains no samples"):
            extract_signal(mock_raw)

    def test_works_with_single_channel(self):
        """Test extraction with a single channel."""
        mock_raw = create_mock_raw(n_channels=1, n_samples=1000)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (1000, 1)

    def test_works_with_many_channels(self):
        """Test extraction with many channels."""
        n_channels = 64
        n_samples = 500
        mock_raw = create_mock_raw(n_channels=n_channels, n_samples=n_samples)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (n_samples, n_channels)

    def test_ignore_channels_excludes_specified_channels(self):
        """Test that ignore_channels excludes specified channels from extraction."""
        ch_names = ["CH0", "CH1", "CH2", "CH3"]
        n_samples = 1000
        mock_raw = create_mock_raw(
            ch_names=ch_names, n_channels=len(ch_names), n_samples=n_samples
        )
        result = extract_signal(mock_raw, ignore_channels=["CH1", "CH3"])
        assert result.signal.shape == (n_samples, 2)

    def test_ignore_channels_with_single_channel(self):
        """Test ignore_channels with a single channel to ignore."""
        ch_names = ["CH0", "CH1", "CH2"]
        n_samples = 500
        mock_raw = create_mock_raw(
            ch_names=ch_names, n_channels=len(ch_names), n_samples=n_samples
        )
        result = extract_signal(mock_raw, ignore_channels=["CH1"])
        assert result.signal.shape == (n_samples, 2)

    def test_ignore_channels_with_all_but_one(self):
        """Test ignore_channels with all but one channel ignored."""
        ch_names = ["CH0", "CH1", "CH2"]
        n_samples = 1000
        mock_raw = create_mock_raw(
            ch_names=ch_names, n_channels=len(ch_names), n_samples=n_samples
        )
        result = extract_signal(mock_raw, ignore_channels=["CH0", "CH2"])
        assert result.signal.shape == (n_samples, 1)

    def test_ignore_channels_with_none(self):
        """Test that None value for ignore_channels includes all channels."""
        ch_names = ["CH0", "CH1", "CH2"]
        n_samples = 500
        mock_raw = create_mock_raw(
            ch_names=ch_names, n_channels=len(ch_names), n_samples=n_samples
        )
        result = extract_signal(mock_raw, ignore_channels=None)
        assert result.signal.shape == (n_samples, 3)

    def test_ignore_channels_with_empty_list(self):
        """Test that empty list for ignore_channels includes all channels."""
        ch_names = ["CH0", "CH1", "CH2"]
        n_samples = 500
        mock_raw = create_mock_raw(
            ch_names=ch_names, n_channels=len(ch_names), n_samples=n_samples
        )
        result = extract_signal(mock_raw, ignore_channels=[])
        assert result.signal.shape == (n_samples, 3)


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractChannels:
    """Test extraction of channel metadata from MNE Raw objects."""

    # ---------- Basic structure and field presence ----------
    def test_returns_array_dict(self):
        """Test that an ArrayDict is returned."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert isinstance(result, ArrayDict)

    def test_contains_id_and_type_fields(self):
        """Test that returned ArrayDict has 'id' and 'type' fields."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert hasattr(result, "id")
        assert hasattr(result, "type")

    def test_id_field_contains_channel_names(self):
        """Test that 'id' field contains the channel names."""
        expected_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
        mock_raw = create_mock_raw(
            ch_names=expected_names, n_channels=len(expected_names)
        )
        result = extract_channels(mock_raw)
        np.testing.assert_array_equal(result.id, np.array(expected_names, dtype="U"))

    def test_type_field_contains_channel_types(self):
        """Test that 'type' field contains the channel types."""
        expected_types = ["eeg", "eog", "emg"]
        mock_raw = create_mock_raw(
            ch_types=expected_types, n_channels=len(expected_types)
        )
        result = extract_channels(mock_raw)
        np.testing.assert_array_equal(result.type, np.array(expected_types, dtype="U"))

    def test_id_dtype_is_unicode(self):
        """Test that 'id' field has unicode dtype."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert result.id.dtype.kind == "U"

    def test_type_dtype_is_unicode(self):
        """Test that 'type' field has unicode dtype."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert result.type.dtype.kind == "U"

    def test_typeerror_raised_for_non_dict_mappings(self):
        """Test that TypeError is raised if a mapping is provided and is not a dict."""
        mock_raw = create_mock_raw()
        # Non-dict inputs for the mapping arguments
        non_dicts = [
            ["not", "a", "dict"],  # list
            "not a dict",  # str
            123,  # int
            (1, 2, 3),  # tuple
        ]
        # channel_names_mapping
        for bad_value in non_dicts:
            with pytest.raises(
                TypeError, match="channel_names_mapping must be a dictionary"
            ):
                extract_channels(mock_raw, channel_names_mapping=bad_value)
        # type_channels_mapping
        for bad_value in non_dicts:
            with pytest.raises(
                TypeError, match="type_channels_mapping must be a dictionary"
            ):
                extract_channels(mock_raw, type_channels_mapping=bad_value)
        # channel_pos_mapping
        for bad_value in non_dicts:
            with pytest.raises(
                TypeError, match="channel_pos_mapping must be a dictionary"
            ):
                extract_channels(mock_raw, channel_pos_mapping=bad_value)

    def test_accepts_none_for_all_mappings(self):
        """Test that extract_channels accepts None for all three mapping arguments."""
        mock_raw = create_mock_raw(
            ch_names=["A", "B", "C"], ch_types=["eeg", "eog", "emg"], n_channels=3
        )
        # All mappings as None
        result = extract_channels(
            mock_raw,
            channel_names_mapping=None,
            type_channels_mapping=None,
            channel_pos_mapping=None,
        )
        # Should simply return the raw names/types
        np.testing.assert_array_equal(result.id, np.array(["A", "B", "C"], dtype="U"))
        np.testing.assert_array_equal(
            result.type, np.array(["eeg", "eog", "emg"], dtype="U")
        )

    # ---------- Channel name mapping tests ----------
    def test_name_mapping_applied(self):
        """Test that channel name mapping is correctly applied."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        name_mapping = {"CH0": "NewCH0", "CH1": "NewCH1"}

        result = extract_channels(mock_raw, channel_names_mapping=name_mapping)

        expected = np.array(["NewCH0", "NewCH1", "CH2"], dtype="U")
        np.testing.assert_array_equal(result.id, expected)

    # ---------- Channel type mapping tests ----------
    def test_type_mapping_with_original_channel_names(self):
        """Test type mapping using original channel names."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=ch_names,
            ch_types=["eeg", "eeg", "eeg"],
            n_channels=len(ch_names),
        )
        type_mapping = {"eog": ["CH0"], "emg": ["CH2"]}

        result = extract_channels(mock_raw, type_channels_mapping=type_mapping)

        expected = np.array(["eog", "eeg", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected)

    def test_type_mapping_with_renamed_channels(self):
        """Test type mapping when applied after name mapping."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names,
            ch_types=["eeg", "eeg", "eeg"],
            n_channels=len(original_names),
        )
        name_mapping = {"CH0": "EOG_L", "CH1": "EOG_R", "CH2": "EMG"}
        type_mapping = {"eog": ["EOG_L", "EOG_R"], "emg": ["EMG"]}

        result = extract_channels(
            mock_raw,
            channel_names_mapping=name_mapping,
            type_channels_mapping=type_mapping,
        )

        expected_types = np.array(["eog", "eog", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected_types)

    # ---------- 3D position extraction tests ----------
    def test_pos_not_included_when_montage_missing(self):
        """Test that 'pos' field is absent when no montage is available."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert not hasattr(result, "pos")

    def test_pos_included_when_montage_has_positions(self):
        """Test that 'pos' field is included when montage positions are available."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        montage = MagicMock()
        montage.get_positions.return_value = {
            "ch_pos": {
                "CH0": np.array([0.1, 0.2, 0.3]),
                "CH2": np.array([0.4, 0.5, 0.6]),
            }
        }
        mock_raw.get_montage.return_value = montage

        result = extract_channels(mock_raw)

        assert hasattr(result, "pos")
        assert result.pos.shape == (3, 3)
        np.testing.assert_allclose(
            result.pos,
            np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan], [0.4, 0.5, 0.6]]),
            equal_nan=True,
        )

    def test_montage_extraction_failure_graceful_fallback(self):
        """Test that position extraction failures are logged but don't fail the function."""
        ch_names = ["CH0", "CH1"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        montage = MagicMock()
        montage.get_positions.side_effect = RuntimeError("Montage error")
        mock_raw.get_montage.return_value = montage

        result = extract_channels(mock_raw)

        assert hasattr(result, "id")
        assert hasattr(result, "type")
        assert not hasattr(result, "pos")

    def test_pos_mapping_with_original_channel_names(self):
        """Test that pos_mapping works with original channel names."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        pos_mapping = {
            "CH0": np.array([0.1, 0.2, 0.3]),
            "CH2": np.array([0.4, 0.5, 0.6]),
        }

        result = extract_channels(mock_raw, channel_pos_mapping=pos_mapping)

        assert hasattr(result, "pos")
        assert result.pos.shape == (3, 3)
        np.testing.assert_allclose(
            result.pos,
            np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan], [0.4, 0.5, 0.6]]),
            equal_nan=True,
        )

    def test_pos_mapping_with_renamed_channel_ids(self):
        """Test that pos_mapping works with renamed channel IDs."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        name_mapping = {"CH0": "NewCH0", "CH1": "NewCH1", "CH2": "NewCH2"}
        pos_mapping = {
            "NewCH0": np.array([0.1, 0.2, 0.3]),
            "NewCH1": np.array([0.25, 0.35, 0.45]),
            "NewCH2": np.array([0.4, 0.5, 0.6]),
        }

        result = extract_channels(
            mock_raw,
            channel_names_mapping=name_mapping,
            channel_pos_mapping=pos_mapping,
        )

        assert hasattr(result, "pos")
        assert result.pos.shape == (3, 3)
        np.testing.assert_allclose(
            result.pos,
            np.array([[0.1, 0.2, 0.3], [0.25, 0.35, 0.45], [0.4, 0.5, 0.6]]),
            equal_nan=True,
        )

    def test_pos_mapping_takes_precedence_over_montage(self):
        """Test that pos_mapping takes precedence over montage positions."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        montage = MagicMock()
        montage.get_positions.return_value = {
            "ch_pos": {
                "CH0": np.array([0.1, 0.2, 0.3]),
                "CH1": np.array([0.2, 0.3, 0.4]),
                "CH2": np.array([0.3, 0.4, 0.5]),
            }
        }
        mock_raw.get_montage.return_value = montage
        pos_mapping = {
            "CH0": np.array([1.0, 2.0, 3.0]),
            "CH1": np.array([2.5, 3.5, 4.5]),
            "CH2": np.array([4.0, 5.0, 6.0]),
        }

        result = extract_channels(mock_raw, channel_pos_mapping=pos_mapping)

        assert hasattr(result, "pos")
        assert result.pos.shape == (3, 3)
        np.testing.assert_allclose(
            result.pos,
            np.array([[1.0, 2.0, 3.0], [2.5, 3.5, 4.5], [4.0, 5.0, 6.0]]),
            equal_nan=True,
        )

    def test_pos_not_included_when_pos_mapping_empty(self):
        """Test that 'pos' field is absent when pos_mapping is empty."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        pos_mapping = {}

        result = extract_channels(mock_raw, channel_pos_mapping=pos_mapping)

        assert not hasattr(result, "pos")

    # ---------- Bad channel handling ----------
    def test_bad_field_omitted_when_no_bad_channels(self):
        """Test that 'bad' field is omitted when there are no bad channels."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert not hasattr(result, "bad")

    def test_bad_field_included_when_bad_channels_exist(self):
        """Test that 'bad' field is included and marks bad channels correctly."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        mock_raw.info["bads"] = ["CH1"]
        result = extract_channels(mock_raw)
        assert hasattr(result, "bad")
        assert result.bad.dtype == bool
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.bad, expected)

    # ---------- Channel name mapping rejection (errors) ----------
    def test_ambiguous_channel_name_mapping_raises_value_error(self):
        """Test that ValueError is raised when channel name mapping is ambiguous."""
        original_names = ["A", "B"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        # Ambiguous swap mapping: "A" -> "B" and "B" -> "A"
        name_mapping = {"A": "B", "B": "A"}

        with pytest.raises(ValueError, match="Ambiguous channel name mapping detected"):
            extract_channels(mock_raw, channel_names_mapping=name_mapping)

    def test_duplicate_channel_name_mapping_raises_value_error(self):
        """Test that ValueError is raised when the final channel IDs have duplicates due to name mapping."""
        original_names = ["A", "B", "C"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        # This will cause both "A" and "B" to be renamed to "DUPLICATE"
        name_mapping = {"A": "DUPLICATE", "B": "DUPLICATE"}

        with pytest.raises(
            ValueError,
            match="Duplicate channel names in channel_names_mapping detected",
        ):
            extract_channels(mock_raw, channel_names_mapping=name_mapping)

    # ---------- ignore_channels argument tests ----------
    def test_ignore_channels_excludes_specified_channels(self):
        """Test that ignore_channels excludes specified channels from extraction."""
        ch_names = ["CH0", "CH1", "CH2", "CH3"]
        ch_types = ["eeg", "eeg", "eog", "emg"]
        mock_raw = create_mock_raw(
            ch_names=ch_names, ch_types=ch_types, n_channels=len(ch_names)
        )
        result = extract_channels(mock_raw, ignore_channels=["CH1", "CH3"])
        assert len(result.id) == 2
        expected_ids = np.array(["CH0", "CH2"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_ignore_channels_with_single_channel(self):
        """Test ignore_channels with a single channel to ignore."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        result = extract_channels(mock_raw, ignore_channels=["CH1"])
        assert len(result.id) == 2
        expected_ids = np.array(["CH0", "CH2"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_ignore_channels_with_none(self):
        """Test that None value for ignore_channels includes all channels."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        result = extract_channels(mock_raw, ignore_channels=None)
        assert len(result.id) == 3
        expected_ids = np.array(ch_names, dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_ignore_channels_with_empty_list(self):
        """Test that empty list for ignore_channels includes all channels."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        result = extract_channels(mock_raw, ignore_channels=[])
        assert len(result.id) == 3
        expected_ids = np.array(ch_names, dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_ignore_channels_with_name_mapping(self):
        """Test that ignore_channels works with channel name mapping."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        name_mapping = {"CH0": "NewCH0", "CH1": "NewCH1", "CH2": "NewCH2"}
        result = extract_channels(
            mock_raw,
            channel_names_mapping=name_mapping,
            ignore_channels=["CH1"],
        )
        assert len(result.id) == 2
        expected_ids = np.array(["NewCH0", "NewCH2"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_ignore_channels_with_type_mapping(self):
        """Test that ignore_channels works with channel type mapping."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=ch_names,
            ch_types=["eeg", "eeg", "eeg"],
            n_channels=len(ch_names),
        )
        type_mapping = {"eog": ["CH0"], "emg": ["CH2"]}
        result = extract_channels(
            mock_raw, type_channels_mapping=type_mapping, ignore_channels=["CH1"]
        )
        assert len(result.type) == 2
        expected_types = np.array(["eog", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected_types)

    def test_ignore_channels_preserves_bad_channel_mask(self):
        """Test that ignore_channels correctly updates bad channel mask."""
        ch_names = ["CH0", "CH1", "CH2", "CH3"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        mock_raw.info["bads"] = ["CH1", "CH3"]
        result = extract_channels(mock_raw, ignore_channels=["CH3"])
        assert hasattr(result, "bad")
        expected_bad = np.array([False, True, False])
        np.testing.assert_array_equal(result.bad, expected_bad)

    def test_ignore_channels_all_but_one(self):
        """Test ignore_channels with all but one channel ignored."""
        ch_names = ["CH0", "CH1", "CH2"]
        ch_types = ["eeg", "eog", "emg"]
        mock_raw = create_mock_raw(
            ch_names=ch_names, ch_types=ch_types, n_channels=len(ch_names)
        )
        result = extract_channels(mock_raw, ignore_channels=["CH0", "CH2"])
        assert len(result.id) == 1
        assert result.id[0] == "CH1"
        assert result.type[0] == "eog"

    # ---------- Mapping dicts: more/fewer keys or entries than channels ----------
    def test_channel_names_mapping_more_keys_than_channels(self):
        """Test name mapping has more keys than there are channels."""
        ch_names = ["A", "B"]
        name_mapping = {"A": "X", "B": "Y", "C": "Z"}
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=2)
        # "C": "Z" extra key should not cause problems
        result = extract_channels(mock_raw, channel_names_mapping=name_mapping)
        expected_ids = np.array(["X", "Y"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_channel_names_mapping_fewer_keys_than_channels(self):
        """Test name mapping has fewer keys than there are channels."""
        ch_names = ["A", "B", "C"]
        name_mapping = {"A": "X"}
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=3)
        # Only "A" should be renamed; others unchanged
        result = extract_channels(mock_raw, channel_names_mapping=name_mapping)
        expected_ids = np.array(["X", "B", "C"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)

    def test_channel_types_mapping_more_keys_than_channels(self):
        """Test type mapping has more keys (types) than there are channels."""
        ch_names = ["A", "B"]
        type_mapping = {"eeg": ["A"], "eog": ["B"], "emg": ["Q"]}
        mock_raw = create_mock_raw(
            ch_names=ch_names, ch_types=["eeg", "eog"], n_channels=2
        )
        # "emg" is not relevant, but should not cause an error
        result = extract_channels(mock_raw, type_channels_mapping=type_mapping)
        expected_types = np.array(["eeg", "eog"], dtype="U")
        np.testing.assert_array_equal(result.type, expected_types)

    def test_channel_types_mapping_fewer_keys_than_channels(self):
        """Test type mapping has fewer keys (types) than there are channels."""
        ch_names = ["A", "B", "C"]
        type_mapping = {"resp": ["A"]}
        mock_raw = create_mock_raw(
            ch_names=ch_names, ch_types=["eeg", "eog", "emg"], n_channels=3
        )
        result = extract_channels(mock_raw, type_channels_mapping=type_mapping)
        # Only A is in mapping, others keep original type
        expected_types = np.array(["resp", "eog", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected_types)

    def test_types_and_names_mapping_all_different_sizes(self):
        """Test different number of keys for type map, name map, and channels."""
        ch_names = ["A", "B", "C", "D"]
        name_mapping = {"A": "X", "B": "Y"}  # only two mapped, two not
        type_mapping = {"eeg": ["X"], "emg": ["C", "D"]}
        # Channel order after mapping: ["X", "Y", "C", "D"]; type mapping only provides "X", "C", "D"
        mock_raw = create_mock_raw(
            ch_names=ch_names, ch_types=["eeg", "eeg", "emg", "emg"], n_channels=4
        )
        result = extract_channels(
            mock_raw,
            channel_names_mapping=name_mapping,
            type_channels_mapping=type_mapping,
        )
        # X gets "eeg" from type_mapping, C and D get "emg" from type_mapping, Y falls back to original type
        expected_ids = np.array(["X", "Y", "C", "D"], dtype="U")
        expected_types = np.array(["eeg", "eeg", "emg", "emg"], dtype="U")
        np.testing.assert_array_equal(result.id, expected_ids)
        np.testing.assert_array_equal(result.type, expected_types)

    def test_channel_mapping_with_nonexistent_channels(self):
        """Test extract_channels handles mappings where none of the channels are present."""
        ch_names = ["A", "B", "C"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=3)
        # Provide mappings that do not match any channels in the raw data
        name_mapping = {"X": "Y", "D": "Z"}
        type_mapping = {"temp": ["Q", "Z"]}
        pos_mapping = {"Q": np.array([1, 2, 3])}

        # Should fall back to defaults/originals, not raise
        result = extract_channels(
            mock_raw,
            channel_names_mapping=name_mapping,
            type_channels_mapping=type_mapping,
            channel_pos_mapping=pos_mapping,
        )
        np.testing.assert_array_equal(result.id, np.array(ch_names, dtype="U"))
        np.testing.assert_array_equal(
            result.type,
            np.array([ch_type for ch_type in mock_raw.get_channel_types()], dtype="U"),
        )
        # Should not have 'pos' unless pos_mapping or montage covers the real channels
        assert not hasattr(result, "pos")


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestConcatenateRecordings:
    """Test concatenation of multiple MNE Raw objects."""

    # ----------- Basic input validation tests -----------
    def test_empty_list_raises_error(self):
        """Test that ValueError is raised for empty recordings list."""
        with pytest.raises(ValueError, match="Recordings list cannot be empty"):
            concatenate_recordings([])

    def test_non_list_input_raises_error(self):
        """Test that ValueError is raised when input is not a list."""
        mock_raw = create_mock_raw()
        with pytest.raises(TypeError, match="Recordings must be a list"):
            concatenate_recordings(mock_raw)

    def test_non_raw_object_raises_error(self):
        """Test that ValueError is raised for non-Raw-like objects."""
        with pytest.raises(ValueError, match="is not an MNE Raw-like object"):
            concatenate_recordings([{"not": "raw"}])

    def test_invalid_on_mismatch_policy_raises_error(self):
        """Test that ValueError is raised for invalid on_mismatch policy."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_mismatch must be one of"):
            concatenate_recordings([mock_raw], on_mismatch="invalid")

    def test_invalid_on_gap_policy_raises_error(self):
        """Test that ValueError is raised for invalid on_gap policy."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_gap must be one of"):
            concatenate_recordings([mock_raw], on_gap="invalid")

    def test_invalid_on_missing_meas_date_policy_raises_error(self):
        """Test that ValueError is raised for invalid on_missing_meas_date policy."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_missing_meas_date must be one of"):
            concatenate_recordings([mock_raw], on_missing_meas_date="invalid")

    # ----------- Channel name mismatch behavior -----------
    def test_different_channel_names_raise_error(self):
        """Test that ValueError is raised for different channel names with raise policy."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)

        with pytest.raises(
            ValueError, match="Mismatch in channel names and/or order across recordings"
        ):
            concatenate_recordings([mock_raw1, mock_raw2])

    # ----------- Concatenation basic behavior -----------
    def test_single_recording_concatenates(self):
        """Test that a single recording can be concatenated."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw.copy.return_value = mock_raw

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = mock_raw
            result = concatenate_recordings([mock_raw])
            mock_concat.assert_called_once()
            assert result == mock_raw

    def test_multiple_recordings_same_meas_date_concatenates(self):
        """Test that multiple recordings with same measurement date are concatenated."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw2 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_result = MagicMock()
            mock_concat.return_value = mock_result
            result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()
            call_args = mock_concat.call_args[0][0]
            assert len(call_args) == 2
            assert result == mock_result

    def test_recordings_sorted_by_meas_date(self):
        """Test that recordings are sorted by measurement date before concatenation."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)
        date3 = datetime.datetime(2023, 6, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date1)
        mock_raw2 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date2)
        mock_raw3 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date3)

        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2
        mock_raw3.copy.return_value = mock_raw3

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_result = MagicMock()
            mock_concat.return_value = mock_result
            concatenate_recordings([mock_raw1, mock_raw2, mock_raw3])

            call_args = mock_concat.call_args[0][0]
            assert call_args[0] == mock_raw3
            assert call_args[1] == mock_raw1
            assert call_args[2] == mock_raw2

    # ----------- Handling missing measurement dates -----------
    def test_missing_meas_date_raise_with_raise_policy(self):
        """Test that ValueError is raised when any meas_date is None with raise policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=None)

        with pytest.raises(
            ValueError, match="One or more recordings have missing measurement dates"
        ):
            concatenate_recordings([mock_raw1, mock_raw2], on_missing_meas_date="raise")

    def test_missing_meas_date_warn_with_warn_policy(self):
        """Test that warning is issued when any meas_date is None with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=None)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning,
                match="One or more recordings have missing measurement dates",
            ):
                result = concatenate_recordings(
                    [mock_raw1, mock_raw2], on_missing_meas_date="warn"
                )
            mock_concat.assert_called_once()

    def test_missing_meas_date_ignore_with_ignore_policy(self):
        """Test that missing meas_date is silently ignored with ignore policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=None)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            result = concatenate_recordings(
                [mock_raw1, mock_raw2], on_missing_meas_date="ignore"
            )
            mock_concat.assert_called_once()

    def test_default_on_missing_meas_date_policy_is_warn(self):
        """Test that default on_missing_meas_date policy is 'warn'."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=None)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning,
                match="One or more recordings have missing measurement dates",
            ):
                result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()

    # ----------- Timezone normalization/handling -----------
    def test_mixed_timezone_aware_datetimes_normalized(self):
        """Test that mixed timezone-aware datetimes are normalized correctly."""
        tz_plus5 = datetime.timezone(datetime.timedelta(hours=5))
        tz_minus3 = datetime.timezone(datetime.timedelta(hours=-3))
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=tz_plus5)
        date2 = datetime.datetime(2023, 6, 15, 2, 0, 0, tzinfo=tz_minus3)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()

    def test_mixed_naive_and_aware_datetimes_normalized(self):
        """Test that mixed naive and timezone-aware datetimes are normalized correctly."""
        naive_date = datetime.datetime(2023, 6, 15, 5, 0, 0)
        aware_date = datetime.datetime(
            2023, 6, 15, 5, 0, 0, tzinfo=datetime.timezone.utc
        )

        mock_raw1 = create_mock_raw(meas_date=naive_date)
        mock_raw2 = create_mock_raw(meas_date=aware_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()

    # ----------- Handling different measurement days (mismatch policy) -----------
    def test_different_meas_date_days_raise_with_raise_policy(self):
        """Test that ValueError is raised for different measurement date days with raise policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError, match="Measurement days are not uniform"):
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="raise")

    def test_different_meas_date_days_warn_with_warn_policy(self):
        """Test that warning is issued for different measurement date days with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(UserWarning, match="Measurement days are not uniform"):
                concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="warn")

    def test_different_meas_date_days_ignore_with_ignore_policy(self):
        """Test that different measurement dates are silently ignored with ignore policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="ignore")
            mock_concat.assert_called_once()

    def test_default_on_mismatch_policy_is_raise(self):
        """Test that default on_mismatch policy is 'raise'."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(
            2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc
        )  # Different day
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError, match="Measurement days are not uniform"):
            concatenate_recordings([mock_raw1, mock_raw2])

    # ----------- Gap checking behavior -----------
    def test_gap_within_max_gap_succeeds_with_warn_policy(self):
        """Test that recordings within max_gap gap pass with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            result = concatenate_recordings([mock_raw1, mock_raw2], on_gap="warn")
            mock_concat.assert_called_once()

    def test_gap_greater_than_max_gap_raise_with_raise_policy(self):
        """Test that ValueError is raised when gap > max_gap with raise policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(
            ValueError, match="Gap between recordings .* is greater than"
        ):
            concatenate_recordings([mock_raw1, mock_raw2], on_gap="raise")

    def test_gap_greater_than_max_gap_warn_with_warn_policy(self):
        """Test that warning is issued when gap > max_gap with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning, match="Gap between recordings .* is greater than"
            ):
                concatenate_recordings([mock_raw1, mock_raw2], on_gap="warn")

    def test_gap_greater_than_max_gap_ignore_with_ignore_policy(self):
        """Test that gap > 1 hour is silently ignored with ignore policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_gap="ignore")
            mock_concat.assert_called_once()

    def test_default_on_gap_policy_is_warn(self):
        """Test that default on_gap policy is 'warn'."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning, match="Gap between recordings .* is greater than"
            ):
                concatenate_recordings([mock_raw1, mock_raw2])

    def test_multiple_gaps_check_all_consecutive_pairs(self):
        """Test that gap check is applied to all consecutive recording pairs."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 30, 0, tzinfo=datetime.timezone.utc)
        date3 = datetime.datetime(2023, 6, 15, 13, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw3 = create_mock_raw(meas_date=date3)

        with pytest.raises(
            ValueError, match="Gap between recordings .* is greater than"
        ):
            concatenate_recordings([mock_raw1, mock_raw2, mock_raw3], on_gap="raise")

    # ----------- Internal mechanics (copies used for concatenation) -----------
    def test_concatenate_raws_called_with_copies(self):
        """Test that mne.concatenate_raws is called with copies of recordings."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(meas_date=meas_date)
        mock_raw2 = create_mock_raw(meas_date=meas_date)

        mock_copy1 = MagicMock()
        mock_copy2 = MagicMock()
        mock_raw1.copy.return_value = mock_copy1
        mock_raw2.copy.return_value = mock_copy2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2])

            call_args = mock_concat.call_args[0][0]
            assert mock_copy1 in call_args
            assert mock_copy2 in call_args


class TestCheckMneAvailable:
    """Test that functions raise ImportError when MNE is not available."""

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_measurement_date_raises_import_error(self):
        """Test that extract_measurement_date raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_measurement_date

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_measurement_date requires the MNE library"
        ):
            extract_measurement_date(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_signal_raises_import_error(self):
        """Test that extract_signal raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_signal

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_signal requires the MNE library"
        ):
            extract_signal(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_channels_raises_import_error(self):
        """Test that extract_channels raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_channels

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_channels requires the MNE library"
        ):
            extract_channels(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_concatenate_recordings_raises_import_error(self):
        """Test that concatenate_recordings raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import concatenate_recordings

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="concatenate_recordings requires the MNE library"
        ):
            concatenate_recordings([mock_raw])
