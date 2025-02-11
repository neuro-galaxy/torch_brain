import argparse
import copy
import glob
import logging
import os

import h5py
import numpy as np
import scipy.signal as signal
import torch

from brainsets import serialize_fn_map
from temporaldata import Data, IrregularTimeSeries

logging.basicConfig(level=logging.INFO)


def process_spikes(data):
    spikes_copy = copy.deepcopy(data.spikes)
    spikes = spikes_copy.timestamps
    end = int((data.end - data.start) * 1000)
    spikes = ((spikes - data.start) * 1000).round(6).astype(int)
    spikes_copy = spikes_copy.select_by_mask(spikes < end)
    spikes = spikes[spikes < end]
    before_start_spikes = copy.deepcopy(spikes_copy)
    before_start_spikes = before_start_spikes.select_by_mask(spikes < 0)
    before_start_spikes.timestamps = (spikes[spikes < 0] + end) / 1000
    after_start_spikes = copy.deepcopy(spikes_copy)
    after_start_spikes = after_start_spikes.select_by_mask(spikes >= 0)
    after_start_spikes.timestamps = spikes[spikes >= 0] / 1000

    timestamps = np.concatenate(
        [after_start_spikes.timestamps, before_start_spikes.timestamps]
    )
    timestamps = timestamps + data.start
    unit_index = np.concatenate(
        [after_start_spikes.unit_index, before_start_spikes.unit_index]
    )
    waveforms = np.concatenate(
        [after_start_spikes.waveforms, before_start_spikes.waveforms]
    )
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        waveforms=waveforms,
        domain="auto",
    )

    spikes.sort()
    return spikes


def get_nb_samples(data, sampling_rate, bin_size_sec):
    nb_points = len(data.finger.timestamps)
    ses_size_sec = nb_points / sampling_rate
    points_per_bin = int(ses_size_sec / bin_size_sec)
    return points_per_bin


from temporaldata import Interval, RegularTimeSeries


def process_finger(data, sampling_rate=250, bin_size_ms=20):
    bin_size_sec = bin_size_ms / 1000
    nb_samples = get_nb_samples(data, sampling_rate, bin_size_sec)
    finger_pos = data.finger.pos[..., 1:3] / 100  # get x and y in meters
    finger_pos = signal.resample(finger_pos, nb_samples)  # is in m/bin
    finger_pos = torch.tensor(finger_pos)

    finger_vel = np.gradient(finger_pos, axis=0)
    finger_vel = finger_vel / bin_size_sec  # adjust to m/s
    finger_vel = torch.tensor(finger_vel).float()

    return RegularTimeSeries(
        pos=np.array(finger_pos),
        vel=np.array(finger_vel),
        sampling_rate=1 / bin_size_sec,
        domain=Interval(data.domain.start[0], data.domain.end[-1]),
    )


def process_data(data):
    spikes = process_spikes(data)
    finger = process_finger(data)
    data_copy = copy.deepcopy(data)
    data_copy.materialize()
    data_copy.spikes = spikes
    data_copy.finger = finger
    interval_split(data_copy)
    return data_copy


def interval_split(data):
    pass
    # intervals = Interval.linspace(data.domain.start[0], data.domain.end[-1], 10)
    # [
    #     train_sampling_intervals,
    #     valid_sampling_intervals,
    #     test_sampling_intervals,
    # ] = intervals.split([8, 1, 1], shuffle=True, random_seed=42)

    # data.set_train_domain(train_sampling_intervals)
    # data.set_valid_domain(valid_sampling_intervals)
    # data.set_test_domain(test_sampling_intervals)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    for file in files:
        logging.info(f"Processing file: {file}")
        with h5py.File(file, "r") as f:
            data = Data.from_hdf5(f)
            data = process_data(data)

        file_name = file.split("/")[-1]
        with h5py.File(f"{args.output_dir}/{file_name}", "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
