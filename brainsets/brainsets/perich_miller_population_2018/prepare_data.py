import argparse
import datetime
import logging

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation, binary_erosion

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


def extract_trials(nwbfile, task):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    if task == "center_out_reaching":
        trials.is_valid = np.logical_and(
            np.logical_and(trials.result == "R", ~(np.isnan(trials.target_id))),
            (trials.end - trials.start) < 6.0,
        )

    elif task == "random_target_reaching":
        trials.is_valid = np.logical_and(
            np.logical_and(trials.result == "R", trials.num_attempted == 4),
            (trials.end - trials.start) < 10.0,
        )

    return trials


def extract_behavior(nwbfile, trials, task):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["Position"]["cursor_pos"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["cursor_pos"].data[:]  # 2d
    cursor_vel = nwbfile.processing["behavior"]["Velocity"]["cursor_vel"].data[:]
    cursor_acc = nwbfile.processing["behavior"]["Acceleration"]["cursor_acc"].data[:]

    # normalization
    cursor_vel = cursor_vel / 20.0

    # create a behavior type segmentation mask
    subtask_index = np.ones_like(timestamps, dtype=np.int64) * int(Task.REACHING.RANDOM)
    if task == "center_out_reaching":
        for i in range(len(trials)):
            # first we check whether the trials are valid or not
            if trials.is_valid[i]:
                subtask_index[
                    (timestamps >= trials.target_on_time[i])
                    & (timestamps < trials.go_cue_time[i])
                ] = int(Task.REACHING.HOLD)
                subtask_index[
                    (timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end[i])
                ] = int(Task.REACHING.REACH)
                subtask_index[
                    (timestamps >= trials.start[i])
                    & (timestamps < trials.target_on_time[i])
                ] = int(Task.REACHING.RETURN)
    elif task == "random_target_reaching":
        for i in range(len(trials)):
            if trials.is_valid[i]:
                subtask_index[
                    (timestamps >= trials.start[i])
                    & (timestamps < trials.go_cue_time_array[i][0])
                ] = int(Task.REACHING.HOLD)

    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(cursor_acc, axis=1)
    mask = hand_acc_norm > 1500.0
    mask = binary_dilation(mask, structure=np.ones(2, dtype=bool))
    subtask_index[mask] = int(Task.REACHING.OUTLIER)

    # we also want to identify out of bound segments
    mask = np.logical_or(cursor_pos[:, 0] < -10, cursor_pos[:, 0] > 10)
    mask = np.logical_or(mask, cursor_pos[:, 1] < -10)
    mask = np.logical_or(mask, cursor_pos[:, 1] > 10)
    # dilate than erode
    mask = binary_dilation(mask, np.ones(400, dtype=bool))
    mask = binary_erosion(mask, np.ones(100, dtype=bool))
    subtask_index[mask] = int(Task.REACHING.OUTLIER)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        subtask_index=subtask_index,
        domain="auto",
    )

    return cursor


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
        experiment_name="perich_miller_population_2018",
        origin_version="dandi/000688/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000688",
        description="This dataset contains electrophysiology and behavioral data from "
        "three macaques performing either a center-out task or a continuous random "
        "target acquisition task. Neural activity was recorded from "
        "chronically-implanted electrode arrays in the primary motor cortex (M1) or "
        "dorsal premotor cortex (PMd) of four rhesus macaque monkeys. A subset of "
        "sessions includes recordings from both regions simultaneously. The data "
        "contains spiking activity—manually spike sorted in three subjects, and "
        "threshold crossings in the fourth subject—obtained from up to 192 electrodes "
        "per session, cursor position and velocity, and other task related metadata.",
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
            sortset_id = f"{subject.id}_{recording_date}"
            task = (
                "center_out_reaching" if "CO" in file_path else "random_target_reaching"
            )
            session_id = f"{sortset_id}_{task}"

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
            trials = extract_trials(nwbfile, task)

            # extract behavior
            cursor = extract_behavior(nwbfile, trials, task)

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
                # domain
                domain=cursor.domain,
            )

            session.register_data(data)

            # split trials into train, validation and test
            successful_trials = trials.select_by_mask(trials.is_valid)
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
