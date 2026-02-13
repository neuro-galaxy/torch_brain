"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
from temporaldata import ArrayDict, Interval, RegularTimeSeries

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    mne = None
    MNE_AVAILABLE = False


def _check_mne_available(func_name: str) -> None:
    """Raise ImportError if MNE is not available."""
    if not MNE_AVAILABLE:
        raise ImportError(
            f"{func_name} requires the MNE library which is not installed. "
            "Install it with `pip install mne`"
        )


def extract_measurement_date(
    recording_data: "mne.io.Raw",
) -> datetime.datetime:
    """Extract the measurement date from MNE Raw recording data.

    Args:
        recording_data: The MNE Raw object containing EEG data and metadata

    Returns:
        The measurement date if present, otherwise Unix epoch (1970-01-01 UTC)

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_measurement_date")
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]
    warnings.warn("No measurement date found, using Unix epoch as placeholder")
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def extract_eeg_signal(
    recording_data: "mne.io.Raw",
) -> RegularTimeSeries:
    """Extract the EEG signal as a RegularTimeSeries from MNE Raw data.

    Args:
        recording_data: The MNE Raw object containing EEG data

    Returns:
        RegularTimeSeries object with the EEG signal

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_eeg_signal")
    sfreq = recording_data.info["sfreq"]
    eeg_signal = recording_data.get_data().T
    if len(eeg_signal) == 0:
        raise ValueError("Recording contains no samples")

    return RegularTimeSeries(
        signal=eeg_signal,
        sampling_rate=sfreq,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(eeg_signal) - 1) / sfreq]),
        ),
    )


def extract_channels(
    recording_data: "mne.io.Raw",
) -> ArrayDict:
    """Extract channel names and types from MNE Raw data.

    Args:
        recording_data: The MNE Raw object containing EEG data

    Returns:
        ArrayDict with fields 'id' (channel names) and 'types' (channel types)

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_channels")
    return ArrayDict(
        ids=np.array(recording_data.ch_names, dtype="U"),
        types=np.array(recording_data.get_channel_types(), dtype="U"),
    )


def extract_psg_signal(raw_psg: "mne.io.Raw") -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract physiological signals from polysomnography (PSG) recording as a RegularTimeSeries.

    Args:
        raw_psg: The MNE Raw object containing PSG data from an EDF file

    Returns:
        A tuple containing:
        - RegularTimeSeries: The extracted physiological signals with
          sampling rate and time domain information
        - ArrayDict: Channel metadata with fields 'ch_id' (channel names)
          and 'ch_type' (channel types: EEG, EOG, EMG, RESP, or TEMP)

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If no signals are extracted from the PSG file.
    """
    _check_mne_available("extract_psg_signal")
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signal_list = []
    channel_meta = []

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        ch_type = None
        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            ch_type = "EEG"
        elif "eog" in ch_name_lower:
            ch_type = "EOG"
        elif "emg" in ch_name_lower:
            ch_type = "EMG"
        elif "resp" in ch_name_lower:
            ch_type = "RESP"
        elif "temp" in ch_name_lower:
            ch_type = "TEMP"
        else:
            continue

        signal_list.append(signal_data)

        channel_meta.append(
            {
                "ch_id": str(ch_name),
                "ch_type": ch_type,
            }
        )

    if not signal_list:
        raise ValueError("No signals extracted from PSG file")

    stacked_signals = np.stack(signal_list, axis=1)

    signals = RegularTimeSeries(
        signal=stacked_signals,
        sampling_rate=raw_psg.info["sfreq"],
        domain=Interval(start=times[0], end=times[-1]),
    )

    channels = ArrayDict(
        ch_id=np.array([ch["ch_id"] for ch in channel_meta], dtype="U"),
        ch_type=np.array([ch["ch_type"] for ch in channel_meta], dtype="U"),
    )

    return signals, channels
