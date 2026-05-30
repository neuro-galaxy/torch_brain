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

from brainsets_utils.taxonomy import RecordingTech, Task
from brainsets_utils import DatasetBuilder, find_files_by_extension

from brainsets_utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)


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
            "target_presentation_time": "target_on_time",
        }
    )

    # some sessions seem to have an incorrect trial table
    # TODO investigate
    index = np.where(
        (trial_table.start.to_numpy()[1:] - trial_table.start.to_numpy()[:-1]) < 0
    )[0]
    if len(index) > 0:
        logging.warning(
            f"Found {len(index) + 1} non contiguous blocks of trials in the "
            f"trial table. Truncating the table to the first contiguous block."
        )
        trial_table = trial_table.iloc[: index[0] + 1]

    trials = Interval.from_dataframe(trial_table)
    is_valid = np.logical_and(trials.discard_trial == 0.0, trials.task_success == 1.0)

    # for some reason some trials are overlapping, we will flag them as invalid
    overlapping_mask = np.zeros_like(is_valid, dtype=bool)
    overlapping_mask[1:] = trials.start[1:] < trials.end[:-1]
    overlapping_mask[:-1] = np.logical_or(overlapping_mask[:-1], overlapping_mask[1:])
    if np.any(overlapping_mask):
        logging.warning(
            f"Found {np.sum(overlapping_mask)} overlapping trials. "
            f"Marking them as invalid."
        )
    is_valid[overlapping_mask] = False

    trials.is_valid = is_valid

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    # TODO what's the difference between hand and cursor?
    timestamps = nwbfile.processing["behavior"]["Position"]["Cursor"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["Cursor"].data[:]  # 2d
    hand_pos = nwbfile.processing["behavior"]["Position"]["Hand"].data[:]
    eye_pos = nwbfile.processing["behavior"]["Position"]["Eye"].data[:]  # 2d

    # derive the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    # derive the velocity and acceleration of the hand
    hand_vel = np.gradient(hand_pos, timestamps, edge_order=1, axis=0)
    hand_acc = np.gradient(hand_vel, timestamps, edge_order=1, axis=0)

    # normalization
    hand_vel = hand_vel / 800.0
    hand_acc = hand_acc / 800.0

    # create a behavior type segmentation mask
    subtask_index = np.ones_like(timestamps, dtype=np.int64) * int(Task.REACHING.RANDOM)
    for i in range(len(trials)):
        # first we check whether the trials are valid or not
        if trials.is_valid[i]:
            subtask_index[
                (timestamps >= trials.target_on_time[i])
                & (timestamps < trials.go_cue_time[i])
            ] = int(Task.REACHING.HOLD)
            subtask_index[
                (timestamps >= trials.move_begins_time[i])
                & (timestamps < trials.move_ends_time[i])
            ] = int(Task.REACHING.REACH)
            subtask_index[
                (timestamps >= trials.move_ends_time[i]) & (timestamps < trials.end[i])
            ] = int(Task.REACHING.RETURN)

    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(hand_acc, axis=1)
    mask = hand_acc_norm > 100.0
    # we dilate the mask to make sure we are not missing any outliers
    structure = np.ones(50, dtype=bool)
    mask = binary_dilation(mask, structure)
    subtask_index[mask] = int(Task.REACHING.OUTLIER)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        subtask_index=subtask_index,
        domain="auto",
    )

    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos_2d=hand_pos,
        vel_2d=hand_vel,
        acc_2d=hand_acc,
        subtask_index=subtask_index,
        domain="auto",
    )

    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos_2d=eye_pos,
        subtask_index=subtask_index,
        domain="auto",
    )

    return cursor, hand, eye


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
        experiment_name="churchland_shenoy_neural_2012",
        origin_version="dandi/000070/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000070",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd) using two 96 channel high density Utah Arrays (Blackrock Microsystems) "
        "while performing reaching tasks with right hand.",
    )

    # iterate over the .nwb files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
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
            session_id = f"{sortset_id}_center_out_reaching"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity
            # this data is from dandi, we can use our helper function
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract data about trial structure
            trials = extract_trials(nwbfile)

            # extract behavior
            cursor, hand, eye = extract_behavior(nwbfile, trials)

            # close file
            io.close()

            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=cursor,
                hand=hand,
                eye=eye,
                domain=hand.domain,
            )

            session.register_data(data)

            # split trials into train, validation and test
            successful_trials = trials.select_by_mask(trials.is_valid)
            assert successful_trials.is_disjoint()

            _, valid_trials, test_trials = successful_trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            train_sampling_intervals = data.domain.difference(
                (valid_trials | test_trials).dilate(3.0)
            )

            session.register_split("train", train_sampling_intervals)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
