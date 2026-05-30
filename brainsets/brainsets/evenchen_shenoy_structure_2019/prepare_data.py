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

logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start, end and target on time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "target_shown_time": "target_on_time",
        }
    )
    trial_table["target_acquire_time"] = trial_table["target_acquire_time"].apply(max)
    trials = Interval.from_dataframe(trial_table)
    trials.is_successful = trials.is_successful == 1.0

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["Position"]["Cursor"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["Cursor"].data[:]  # 2d
    hand_pos = nwbfile.processing["behavior"]["Position"]["Hand"].data[:]
    eye_pos = nwbfile.processing["behavior"]["Position"]["Eye"].data[:]  # 2d

    # derive the velocity and acceleration of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    # derive the velocity and acceleration of the hand
    hand_vel = np.gradient(hand_pos, timestamps, edge_order=1, axis=0)
    hand_acc = np.gradient(hand_vel, timestamps, edge_order=1, axis=0)

    # TODO: revisit need for normalization (cursor and hand)

    # create a behavior type segmentation mask
    subtask_index = np.ones_like(timestamps, dtype=np.int64) * Task.REACHING.RANDOM
    for i in range(len(trials)):
        # first we check whether the trials are valid or not
        if trials.is_successful[i]:
            subtask_index[
                (timestamps >= trials.target_on_time[i])
                & (timestamps < trials.go_cue_time[i])
            ] = Task.REACHING.HOLD
            subtask_index[
                (timestamps >= trials.target_acquire_time[i])
                & (timestamps < trials.target_held_time[i])
            ] = Task.REACHING.HOLD
            # return to center (recorded as separate trial)
            if trials.target_pos[i][:2].count_nonzero == 0:
                subtask_index[
                    (timestamps >= trials.go_cue_time[i])
                    & (timestamps < trials.target_acquire_time[i])
                ] = Task.REACHING.RETURN
            # reach target that is not center
            else:
                subtask_index[
                    (timestamps >= trials.go_cue_time[i])
                    & (timestamps < trials.target_acquire_time[i])
                ] = Task.REACHING.REACH
        else:
            subtask_index[
                (timestamps >= trials.start[i]) & (timestamps <= trials.end[i])
            ] = Task.REACHING.INVALID

    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(hand_acc, axis=1)
    mask = hand_acc_norm > 5e6
    # we dilate the mask to make sure we are not missing any outliers
    structure = np.ones(150, dtype=bool)
    mask = binary_dilation(mask, structure)
    subtask_index[mask] = Task.REACHING.OUTLIER

    # we also want to identify out of bound segments
    mask = np.logical_or(cursor_pos[:, 1] < -200, cursor_pos[:, 1] > 200)
    mask = np.logical_or(mask, cursor_pos[:, 1] < -200)
    mask = np.logical_or(mask, cursor_pos[:, 1] > 200)
    subtask_index[mask] = int(Task.REACHING.OUTLIER)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        subtask_index=subtask_index,
        domain="auto",
    )
    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos=hand_pos,
        vel=hand_vel,
        acc=hand_acc,
        subtask_index=subtask_index,
        domain="auto",
    )
    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos=eye_pos,
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
        experiment_name="evenchen_shenoy_structure_2019",
        origin_version="dandi/000121/0.220124.2156",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000121",
        description="Extracellular physiology data of monkeys implanted with 96 channel "
        "Blackrock Utah arrays in the Motor cortex and Premotor cortex. Each file "
        "contains a session's worth of data for one monkey (total 2) performing 1 of the "
        "four cursor movement task designs. The data contains hand, eye and cursor "
        "position data, LFP, sorted spikes and other task related trialized data.",
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
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
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

            # register session
            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=cursor,
                hand=hand,
                eye=eye,
                domain=cursor.domain,
            )

            session.register_data(data)

            # split trials into train, validation and test
            successful_trials = trials.select_by_mask(trials.is_successful)
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
