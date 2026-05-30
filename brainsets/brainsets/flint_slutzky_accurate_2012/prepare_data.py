import argparse
import logging

import numpy as np
from scipy.io import loadmat
import pandas as pd

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)


from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.taxonomy import Task, Species, Sex, SubjectDescription


logging.basicConfig(level=logging.INFO)


def extract_units(mat):
    """
    Get unit metadata for the session.
    This is the same for all trials.

    ..note::
        Currently only populating unit_name and unit_number.
        TODO: add more unit metadata like group_name, recording-tech etc.

    """
    values = mat["Subject"][0][0][0]

    # Only get unit meta from the first trial
    unit_meta = []
    neurons = values["Neuron"][0][0]
    for i, _ in enumerate(neurons):
        unit_name = f"unit_{i}"
        unit_meta.append(
            {
                "id": unit_name,
                "unit_number": i,
            }
        )

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))
    return units


def extract_behavior(mat, trials):
    """
    Get lists of behavior timestamps and hand velocity for each trial.
    These timestamps are regularly sampled every 100ms.

    Parameters:
    - mat: A dictionary containing the data extracted from a MATLAB file.

    Returns:
    - behavior object of type IrregularTimeSeries
    """
    values = mat["Subject"][0][0][0]
    behavior_timestamps_list = []
    hand_vel_list = []
    for trial_id in range(len(values["Time"])):
        behavior_timestamps = values["Time"][trial_id][0][:, 0]
        hand_vel = values["HandVel"][trial_id][0][:, :2]
        behavior_timestamps_list.append(behavior_timestamps)
        hand_vel_list.append(hand_vel)

    behavior_timestamps = np.concatenate(behavior_timestamps_list)
    hand_vel = np.concatenate(hand_vel_list)

    hand = IrregularTimeSeries(
        timestamps=behavior_timestamps,
        vel=hand_vel * 2.5,
        subtask_index=np.ones_like(behavior_timestamps, dtype=np.int64)
        * int(Task.REACHING.RANDOM),
        domain=trials,
    )
    return hand


def extract_spikes(mat):
    """
    Extracts spike timestamps and unit ids for each trial from a MATLAB file.

    Parameters:
    - mat: A dictionary containing the data extracted from a MATLAB file.

    Returns:
    - spikes object of type IrregularTimeSeries


    """
    values = mat["Subject"][0][0][0]
    spike_timestamps_list = []
    unit_id_list = []
    trial_start = []
    trial_end = []
    for trial_id in range(len(values["Time"])):
        neurons = values["Neuron"][trial_id][0]
        tstart = np.inf
        tend = 0
        for i, neuron in enumerate(neurons):
            spiketimes = neuron[0][0]
            if len(spiketimes) == 0:
                continue
            spiketimes = spiketimes[:, 0]
            spike_timestamps_list.append(spiketimes)
            unit_id_list.append(np.ones_like(spiketimes, dtype=np.int64) * i)
            tstart = spiketimes.min() if spiketimes.min() < tstart else tstart
            tend = spiketimes.max() if spiketimes.max() > tend else tend
        trial_start.append(tstart)
        trial_end.append(tend)

    spikes = np.concatenate(spike_timestamps_list)
    unit_ids = np.concatenate(unit_id_list)
    spikes = IrregularTimeSeries(
        timestamps=spikes,
        unit_index=unit_ids,
        domain=Interval(np.array(trial_start), np.array(trial_end)),
    )
    spikes.sort()
    return spikes


def extract_trials(mat):
    """
    Get a list of trial intervals for each trial.
    These intervals are defined by the start and end of the behavior timestamps.

    Parameters:
    - mat: A dictionary containing the data extracted from a MATLAB file.

    Returns:
    - A list of trial intervals for each trial of type Interval.
    """
    values = mat["Subject"][0][0][0]
    trial_starts = []
    trial_ends = []
    for trial_id in range(len(values["Time"])):
        behavior_timestamps = values["Time"][trial_id][0][:, 0]
        trial_starts.append(behavior_timestamps.min())
        trial_ends.append(behavior_timestamps.max())
    trials = Interval(
        start=np.array(trial_starts),
        end=np.array(trial_ends),
    )
    return trials


def main():
    experiment_name = "flint_slutzky_accurate_2012"

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name=experiment_name,
        origin_version="0.0.0",
        derived_version="1.0.0",
        source="https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd)  128-channel acquisition system (Cerebus,Blackrock, Inc.)  "
        "while performing reaching tasks on right hand",
    )

    for file_path in find_files_by_extension(raw_folder_path, ".mat"):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            mat = loadmat(file_path)

            subject = SubjectDescription(
                id="monkey_c",
                species=Species.MACACA_MULATTA,
                sex=Sex.UNKNOWN,
            )
            session.register_subject(subject)

            session_tag = file_path.split("_")[-1].split(".mat")[0]  # e1, e2, e3...
            subject_id = subject.id
            sortset_id = f"{subject_id}_{session_tag}"
            session_id = f"{sortset_id}_center_out_reaching"

            # register session
            session.register_session(
                id=session_id,
                recording_date="20130530",  # using .mat file creation date for now.
                task=Task.REACHING,
            )

            units = extract_units(mat)  # Data obj

            spikes = extract_spikes(mat)  # IrregularTimeSeries obj

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            trials = extract_trials(mat)  # Interval obj

            hand = extract_behavior(mat, trials)  # IrregularTimeSeries obj

            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                hand=hand,
                domain=trials,
            )

            session.register_data(data)

            # split trials into train, validation and test
            train_trials, valid_trials, test_trials = trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            # save samples
            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for entire dataset
    db.finish()


if __name__ == "__main__":
    main()
