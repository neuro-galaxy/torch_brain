import argparse
import logging
import os

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
)

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import (
    RecordingTech,
    Species,
    Task,
    Sex,
)


logging.basicConfig(level=logging.INFO)


def extract_units(mat):
    """
    Get unit metadata for the session.
    This is the same for all trials.

    ..note::
        Currently only populating unit_name and unit_number.
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

    behavior_timestamps = np.concatenate(behavior_timestamps_list).astype(np.float64)
    hand_vel = np.concatenate(hand_vel_list)

    hand = IrregularTimeSeries(
        timestamps=behavior_timestamps,
        vel=hand_vel,
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

    spikes = np.concatenate(spike_timestamps_list).astype(np.float64)
    unit_ids = np.concatenate(unit_id_list)
    spikes = IrregularTimeSeries(
        timestamps=spikes,
        unit_index=unit_ids,
        domain=Interval(
            np.array(trial_start, dtype=np.float64),
            np.array(trial_end, dtype=np.float64),
        ),
    )
    spikes.sort()
    assert spikes.domain.is_disjoint()
    assert spikes.domain.is_sorted()
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
        start=np.array(trial_starts, dtype=np.float64),
        end=np.array(trial_ends, dtype=np.float64),
    )
    assert trials.is_disjoint()
    assert trials.is_sorted()
    return trials


def main():
    # Use argparse to extract two arguments from the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="flint_slutzky_accurate_2012",
        origin_version="0.0.0",
        derived_version="1.0.0",
        source="https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd)  128-channel acquisition system (Cerebus,Blackrock, Inc.)  "
        "while performing reaching tasks on right hand",
    )

    logging.info(f"Processing file: {args.input_file}")

    # open file
    mat = loadmat(args.input_file)

    subject = SubjectDescription(
        id="monkey_c",
        species=Species.MACACA_MULATTA,
        sex=Sex.UNKNOWN,
    )

    session_tag = args.input_file.split("_")[-1].split(".mat")[0]  # e1, e2, e3...
    device_id = f"{subject.id}_{session_tag}"
    session_id = f"{device_id}_reaching"

    # register session
    session = SessionDescription(
        id=session_id,
        recording_date="20130530",  # using .mat file creation date
        task=Task.REACHING,
    )

    device = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    units = extract_units(mat)  # Data obj

    spikes = extract_spikes(mat)  # IrregularTimeSeries obj

    trials = extract_trials(mat)  # Interval obj

    hand = extract_behavior(mat, trials)  # IrregularTimeSeries obj

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session,
        device=device,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        trials=trials,
        hand=hand,
        domain=trials,
    )

    # split trials into train, validation and test
    train_trials, valid_trials, test_trials = trials.split(
        [0.7, 0.1, 0.2], shuffle=True, random_seed=42
    )

    data.set_train_domain(train_trials)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
