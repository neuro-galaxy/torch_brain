import argparse
import datetime
import logging
import os
import h5py


import numpy as np
from pynwb import NWBHDF5IO

from temporaldata import Data, IrregularTimeSeries, Interval
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map


logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "split": "split_indicator",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    # the dataset has pre-defined train/valid splits, we will use the valid split
    # as our test
    train_mask_nwb = trial_table.split_indicator.to_numpy() == "train"
    test_mask_nwb = trial_table.split_indicator.to_numpy() == "val"

    trials.train_mask_nwb = (
        train_mask_nwb  # Naming with "_" since train_mask is reserved
    )
    trials.test_mask_nwb = test_mask_nwb  # Naming with "_" since test_mask is reserved

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["hand_vel"].timestamps[:]
    hand_pos = nwbfile.processing["behavior"]["hand_pos"].data[:]
    hand_vel = nwbfile.processing["behavior"]["hand_vel"].data[:]
    eye_pos = nwbfile.processing["behavior"]["eye_pos"].data[:]

    # report accuracy only on the evaluation intervals
    eval_mask = np.zeros_like(timestamps, dtype=bool)

    for i in range(len(trials)):

        eval_mask[
            (timestamps >= (trials.move_onset_time[i] - 0.05))
            & (timestamps < (trials.move_onset_time[i] + 0.65))
        ] = True

    behavior = IrregularTimeSeries(
        timestamps=timestamps,
        hand_pos=hand_pos,
        hand_vel=hand_vel,
        eye_pos=eye_pos,
        eval_mask=eval_mask,
        domain="auto",
    )

    return behavior


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    brainset_description = BrainsetDescription(
        id="pei_pandarinath_nlb_2021",
        origin_version="dandi/000140/0.220113.0408",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000140",
        description="This dataset contains sorted unit spiking times and behavioral"
        " data from a macaque performing a delayed reaching task. The experimental task"
        " was a center-out reaching task with obstructing barriers forming a maze,"
        " resulting in a variety of straight and curved reaches.",
    )

    logging.info(f"Processing file: {args.input_file}")

    # open file
    io = NWBHDF5IO(args.input_file, "r")
    nwbfile = io.read()

    # extract subject metadata
    # this dataset is from dandi, which has structured subject metadata, so we
    # can use the helper function extract_subject_from_nwb
    subject = extract_subject_from_nwb(nwbfile)

    # extract experiment metadata
    recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    device_id = f"{subject.id}_{recording_date}"
    session_id = f"{subject.id}_maze"

    # register session
    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        task=Task.REACHING,
    )

    # register device
    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # extract spiking activity
    # this data is from dandi, we can use our helper function
    spikes, units = extract_spikes_from_nwbfile(
        nwbfile,
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # extract data about trial structure
    trials = extract_trials(nwbfile)

    data = Data(
        brainset=brainset_description,
        session=session_description,
        device=device_description,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        trials=trials,
        # domain
        domain="auto",
    )

    if not "test" in args.input_file:
        # extract behavior
        data.behavior = extract_behavior(nwbfile, trials)

        # split and register trials into train, validation and test
        train_trials, valid_trials = trials.select_by_mask(trials.train_mask_nwb).split(
            [0.8, 0.2], shuffle=True, random_seed=42
        )
        test_trials = trials.select_by_mask(trials.test_mask_nwb)

        data.train_domain = train_trials
        data.valid_domain = valid_trials
        data.test_domain = test_trials

    # close file
    io.close()

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
