import argparse
import datetime
import logging
import h5py
import os

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation, binary_erosion

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


def extract_behavior(nwbfile):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2).
    """
    timestamps = nwbfile.processing["behavior"]["Position"]["cursor_pos"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["cursor_pos"].data[:]  # 2d
    cursor_vel = nwbfile.processing["behavior"]["Velocity"]["cursor_vel"].data[:]
    cursor_acc = nwbfile.processing["behavior"]["Acceleration"]["cursor_acc"].data[:]

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        domain="auto",
    )

    return cursor


def extract_trials(nwbfile, task, cursor):
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

    # next we extract the different periods in the trials
    if task == "center_out_reaching":
        # isolate valid trials based on success
        trials.is_valid = np.logical_and(
            np.logical_and(trials.result == "R", ~(np.isnan(trials.target_id))),
            (trials.end - trials.start) < 6.0,
        )
        valid_trials = trials.select_by_mask(trials.is_valid)

        movement_phases = Data(
            hold_period=Interval(
                start=valid_trials.target_on_time, end=valid_trials.go_cue_time
            ),
            reach_period=Interval(start=valid_trials.go_cue_time, end=valid_trials.end),
            return_period=Interval(
                start=valid_trials.start, end=valid_trials.target_on_time
            ),
            domain="auto",
        )

    elif task == "random_target_reaching":
        # isolate valid trials based on success
        trials.is_valid = np.logical_and(
            np.logical_and(trials.result == "R", trials.num_attempted == 4),
            (trials.end - trials.start) < 10.0,
        )
        valid_trials = trials.select_by_mask(trials.is_valid)

        movement_phases = Data(
            hold_period=Interval(
                start=valid_trials.start, end=valid_trials.go_cue_time_array[:, 0]
            ),
            domain="auto",
        )

    # everything outside of the different identified periods will be marked as random
    movement_phases.random_period = cursor.domain.difference(movement_phases.domain)

    return trials, movement_phases


def detect_outliers(cursor):
    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(cursor.acc, axis=1)
    mask_acceleration = hand_acc_norm > 1500.0
    mask_acceleration = binary_dilation(
        mask_acceleration, structure=np.ones(2, dtype=bool)
    )

    # we also want to identify out of bound segments
    mask_position = np.logical_or(cursor.pos[:, 0] < -10, cursor.pos[:, 0] > 10)
    mask_position = np.logical_or(mask_position, cursor.pos[:, 1] < -10)
    mask_position = np.logical_or(mask_position, cursor.pos[:, 1] > 10)
    # dilate than erode
    mask_position = binary_dilation(mask_position, np.ones(400, dtype=bool))
    mask_position = binary_erosion(mask_position, np.ones(100, dtype=bool))

    outlier_mask = np.logical_or(mask_acceleration, mask_position)

    # convert to interval, you need to find the start and end of the outlier segments
    start = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == 1)[0]]
    if outlier_mask[0]:
        start = np.insert(start, 0, cursor.timestamps[0])

    end = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == -1)[0]]
    if outlier_mask[-1]:
        end = np.insert(end, 0, cursor.timestamps[-1])

    cursor_outlier_segments = Interval(start=start, end=end)

    return cursor_outlier_segments


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    brainset_description = BrainsetDescription(
        id="perich_miller_population_2018",
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
    task = (
        "center_out_reaching" if "CO" in args.input_file else "random_target_reaching"
    )
    session_id = f"{device_id}_{task}"

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
        nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
    )

    # extract behavior
    cursor = extract_behavior(nwbfile)
    cursor_outlier_segments = detect_outliers(cursor)

    # extract data about trial structure
    trials, movement_phases = extract_trials(nwbfile, task, cursor)

    # close file
    io.close()

    # register session
    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        trials=trials,
        movement_phases=movement_phases,
        cursor=cursor,
        cursor_outlier_segments=cursor_outlier_segments,
        # domain
        domain=cursor.domain,
    )

    # split trials into train, validation and test
    successful_trials = trials.select_by_mask(trials.is_valid)
    _, valid_trials, test_trials = successful_trials.split(
        [0.7, 0.1, 0.2], shuffle=True, random_seed=42
    )

    train_sampling_intervals = data.domain.difference(
        (valid_trials | test_trials).dilate(3.0)
    )

    data.set_train_domain(train_sampling_intervals)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
