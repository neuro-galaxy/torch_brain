"""Load data, processes it, save it."""

import argparse
import datetime
import logging

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)


from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)
from brainsets_utils.taxonomy import RecordingTech, Task
from brainsets_utils.taxonomy.task import REACHING


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

    # normalization
    hand_vel = hand_vel / 1000.0

    # create a behavior type segmentation mask
    behavior_type = np.ones_like(timestamps, dtype=np.int64) * int(REACHING.RANDOM)

    # report accuracy only on the evaluation intervals
    eval_mask = np.zeros_like(timestamps, dtype=bool)

    for i in range(len(trials)):
        # first we check whether the trials are valid or not
        if trials.success[i]:
            behavior_type[
                (timestamps >= trials.target_on_time[i])
                & (timestamps < trials.go_cue_time[i])
            ] = int(REACHING.HOLD)
            behavior_type[
                (timestamps >= trials.move_onset_time[i]) & (timestamps < trials.end[i])
            ] = int(REACHING.REACH)

        eval_mask[
            (timestamps >= (trials.move_onset_time[i] - 0.05))
            & (timestamps < (trials.move_onset_time[i] + 0.65))
        ] = True

    behavior = IrregularTimeSeries(
        timestamps=timestamps,
        hand_pos=hand_pos,
        hand_vel=hand_vel,
        eye_pos=eye_pos,
        subtask_index=behavior_type,
        eval_mask=eval_mask,
        domain="auto",
    )

    return behavior


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="mc_maze_small",
        origin_version="dandi/000140/0.220113.0408",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000140",
        description="This dataset contains sorted unit spiking times and behavioral"
        " data from a macaque performing a delayed reaching task. The experimental task"
        " was a center-out reaching task with obstructing barriers forming a maze,"
        " resulting in a variety of straight and curved reaches.",
    )

    # iterate over the .nwb files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
        if "test" in file_path:
            # test file does not have behavior, skipping
            continue

        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            # extract subject metadata
            # this dataset is from dandi, which has structured subject metadata, so we
            # can use the helper function extract_subject_from_nwb
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
            subject_id = subject.id
            sortset_id = f"{subject_id}_{recording_date}"
            session_id = f"{sortset_id}_maze"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity
            # this data is from dandi, we can use our helper function
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile,
                recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract data about trial structure
            trials = extract_trials(nwbfile)

            # extract behavior
            behavior = extract_behavior(nwbfile, trials)

            # close file
            io.close()

            # register session
            session_start, session_end = (
                behavior.timestamps[0].item(),
                behavior.timestamps[-1].item(),
            )

            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                behavior=behavior,
                # domain
                domain=Interval(session_start, session_end),
            )

            session.register_data(data)

            # split and register trials into train, validation and test
            train_trials, valid_trials = trials.select_by_mask(
                trials.train_mask_nwb
            ).split([0.8, 0.2], shuffle=True, random_seed=42)
            test_trials = trials.select_by_mask(trials.test_mask_nwb)

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
