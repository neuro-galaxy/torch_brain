"""Signal processing functions. Inspired by Stavisky et al. (2015).

https://dx.doi.org/10.1088/1741-2560/12/3/036009
"""

from typing import List, Tuple

import numpy as np
import tqdm
from scipy import signal

from temporaldata import Data, IrregularTimeSeries, ArrayDict
from brainsets.taxonomy import RecordingTech


def downsample_wideband(
    wideband: np.ndarray,
    timestamps: np.ndarray,
    wideband_Fs: float,
    lfp_Fs: float = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample wideband signal to LFP sampling rate.
    """
    assert wideband.shape[0] == timestamps.shape[0], "Time should be first dimension."
    # Decimate by a factor of 4
    dec_factor = 4
    if wideband.shape[0] % dec_factor != 0:
        wideband = wideband[: -(wideband.shape[0] % dec_factor), :]
        timestamps = timestamps[: -(timestamps.shape[0] % dec_factor)]
    wideband = wideband.reshape(-1, dec_factor, wideband.shape[1])
    wideband = wideband.mean(axis=1)

    timestamps = timestamps[::dec_factor]

    nyq = 0.5 * wideband_Fs / dec_factor  # Nyquist frequency
    cutoff = 0.333 * lfp_Fs  # remove everything above 170 Hz.
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False, output="ba")

    # Interpolation to achieve the desired sampling rate
    t_new = np.arange(timestamps[0], timestamps[-1], 1 / lfp_Fs)
    lfp = np.zeros((len(t_new), wideband.shape[1]))
    for i in range(wideband.shape[1]):
        # We do this one channel at a time to save memory.
        broadband_low = signal.filtfilt(b, a, wideband[:, i], axis=0)
        lfp[:, i] = np.interp(t_new, timestamps, broadband_low)

    return lfp, t_new


def extract_bands(
    lfps: np.ndarray, ts: np.ndarray, Fs: float = 1000, notch: float = 60
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Extract bands from LFP

    We prefer to extract bands from the LFP upstream rather than downstream, because
    it can be difficult to estimate e.g. the phase of low-frequency LFPs from
    short segments.

    We use the proposed bands from Stravisky et al. (2015), but we use the MNE toolbox
    rather than straight scipy signal.
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "This function requires the MNE library which you can install with "
            "`pip install mne`"
        )

    target_Fs = 50
    assert (
        Fs % target_Fs == 0
    ), "Sampling rate must be a multiple of the target frequency"

    assert lfps.shape[0] == ts.shape[0], "Time should be first dimension."
    info = mne.create_info(
        ch_names=lfps.shape[1], sfreq=Fs, ch_types=["eeg"] * lfps.shape[1]
    )
    data = mne.io.RawArray(lfps.T, info)
    data = data.notch_filter(np.arange(notch, notch * 5 + 1, notch), n_jobs=4)

    filtered = []
    band_names = ["delta", "theta", "alpha", "beta", "gamma", "lmp"]
    bands = [(1, 4), (3, 10), (12, 23), (27, 38), (50, 300)]
    for band_low, band_hi in bands:
        band = data.copy().filter(band_low, band_hi, fir_design="firwin", n_jobs=4)
        band = band.apply_function(lambda x: x**2, n_jobs=4)

        band = band.filter(18, None, fir_design="firwin", n_jobs=4)
        # It seems resample overwrites the original data, so we copy it first.
        band = band.resample(target_Fs, npad="auto", n_jobs=4)

        filtered.append(band.get_data().T)

    lmp = data.copy().filter(0.1, 20, fir_design="firwin", n_jobs=4)
    lmp = lmp.resample(target_Fs, npad="auto", n_jobs=4)
    filtered.append(lmp.get_data().T)

    ts = ts[int(Fs / target_Fs / 2) :: int(Fs / target_Fs)]
    stacked = np.stack(filtered, axis=2)

    # There can be off by one errors.
    if stacked.shape[0] != len(ts):
        stacked = stacked[: len(ts), :, :]

    return stacked, ts, band_names


def cube_to_long(
    ts: np.ndarray, cube: np.ndarray, channel_prefix="chan"
) -> Tuple[List[IrregularTimeSeries], Data]:
    """Convert a cube of threshold crossings to a list of trials and units."""
    assert cube.shape[1] == len(ts)
    assert cube.ndim == 3
    channels = np.arange(cube.shape[2])
    channels = np.tile(channels, [cube.shape[1], 1])

    # First dim is batch, second is time, third is channel.
    assert np.issubdtype(cube.dtype, np.integer)
    assert cube.min() >= 0

    ts = np.tile(ts.reshape((-1, 1)), [1, cube.shape[2]])
    assert ts.shape == channels.shape

    # The first dimension we map to a single trial.
    trials = []
    for b in tqdm.tqdm(range(cube.shape[0])):
        cube_ = cube[b, :, :]
        ts_ = []
        channels_ = []

        # This data is binned, so we create N identifical timestamps when there are N
        # spikes in a bin.
        for n in range(1, cube_.max() + 1):
            ts_.append(ts[cube_ >= n])
            channels_.append(channels[cube_ >= n])

        ts_ = np.concatenate(ts_)
        channels_ = np.concatenate(channels_)

        tidx = np.argsort(ts_)
        ts_ = ts_[tidx]
        channels_ = channels_[tidx]

        trials.append(
            IrregularTimeSeries(
                timestamps=ts_,
                unit_index=channels_,
                types=np.ones(len(ts_))
                * int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
                domain="auto",
            )
        )

    counts = cube.sum(axis=0).sum(axis=0)
    units = ArrayDict(
        count=np.array(counts.astype(int)),
        channel_name=np.array(
            [f"{channel_prefix}{c:03}" for c in range(cube.shape[2])]
        ),
        unit_number=np.zeros(cube.shape[2]),
        id=np.array([f"{channel_prefix}{c}" for c in range(cube.shape[2])]),
        channel_number=np.arange(cube.shape[2]),
        type=np.ones(cube.shape[2]) * int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
    )

    return trials, units
