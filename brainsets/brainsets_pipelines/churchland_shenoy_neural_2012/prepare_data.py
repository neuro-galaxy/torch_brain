import argparse
import datetime
import logging
import h5py
import os

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation
import pandas as pd
from temporaldata import (
    Data,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets.utils.dandi_utils import extract_subject_from_nwb
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile, session_id):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()
    # rename a few columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "target_presentation_time": "target_on_time",
        }
    )
    timekeys = [
        "start",
        "end",
        "target_on_time",
        "go_cue_time",
        "move_begins_time",
        "move_ends_time",
    ]

    # some sessions' raw data is messed up, we need to fix it
    artifact_index = np.where(
        trial_table.start.to_numpy()[1:] < trial_table.end.to_numpy()[:-1]
    )[0]
    artifact_dict = {}
    if len(artifact_index) > 0:
        logging.info(f"Found {len(artifact_index)} artifacts in the trial table.")
        if session_id in [
            "nitschke_20090910_center_out_reaching",
            "nitschke_20090920_center_out_reaching",
        ]:
            # the first artifact corresponds to two overlapping trials
            # we will skip those two trials
            skip_trials = [artifact_index[0], artifact_index[0] + 1]
            logging.info(f"Dropping trials {skip_trials} due to artifacts.")
            artifact_dict["skip_start"] = trial_table.start.to_numpy()[skip_trials][0]
            artifact_dict["skip_end"] = trial_table.end.to_numpy()[skip_trials][1]
            trial_table = trial_table.drop(index=skip_trials).reset_index(drop=True)
            artifact_index = artifact_index[1:]
            artifact_index = artifact_index - 2

        epoch_start_indices = np.concatenate([np.array([0]), artifact_index + 1])
        epoch_end_indices = np.concatenate(
            [artifact_index + 1, np.array([len(trial_table)])]
        )

        # we split the trial table into epochs
        trial_table_list = [
            trial_table.iloc[start_idx:end_idx].copy()
            for start_idx, end_idx in zip(epoch_start_indices, epoch_end_indices)
        ]

        # we estimate the start and end of each epoch
        epoch_start_times = [
            trial_table_list[i].start.to_numpy()[0]
            for i in range(len(trial_table_list))
        ]
        epoch_end_times = [
            trial_table_list[i].end.to_numpy()[-1] for i in range(len(trial_table_list))
        ]

        artifact_dict["original_start_times"] = epoch_start_times
        artifact_dict["original_end_times"] = epoch_end_times
        artifact_dict["fixed_start_times"] = [epoch_start_times[0]]
        artifact_dict["fixed_end_times"] = [epoch_end_times[0]]
        gap = 60  # seconds
        for i, trial_block in enumerate(trial_table_list):
            if i == 0:
                # we keep the first epoch as is
                continue
            # the next epochs will be reset to start "gap" seconds after the previous epoch
            for key in timekeys:
                trial_block[key] = (
                    trial_block[key]
                    - artifact_dict["original_start_times"][i]
                    + artifact_dict["fixed_end_times"][i - 1]
                    + gap
                )
            artifact_dict["fixed_start_times"].append(
                epoch_start_times[i]
                - artifact_dict["original_start_times"][i]
                + artifact_dict["fixed_end_times"][i - 1]
                + gap
            )
            artifact_dict["fixed_end_times"].append(
                epoch_end_times[i]
                - artifact_dict["original_start_times"][i]
                + artifact_dict["fixed_end_times"][i - 1]
                + gap
            )

        # then we recombine everything!
        trial_table = pd.concat(trial_table_list)

    trials = Interval.from_dataframe(trial_table, timekeys=timekeys)

    assert trials.is_disjoint()
    assert trials.is_sorted()

    trials.is_valid = np.logical_and(
        trials.discard_trial == 0.0, trials.task_success == 1.0
    )
    valid_trials = trials.select_by_mask(trials.is_valid)

    movement_phases = Data(
        hold_period=Interval(
            start=valid_trials.target_on_time, end=valid_trials.go_cue_time
        ),
        reach_period=Interval(
            start=valid_trials.move_begins_time, end=valid_trials.move_ends_time
        ),
        return_period=Interval(start=valid_trials.move_ends_time, end=valid_trials.end),
        domain="auto",
    )

    return trials, movement_phases, artifact_dict


def extract_behavior(nwbfile, artifact_dict):
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

    # some sessions' raw data is messed up, we need to fix it
    artifact_index = np.where(timestamps[1:] < timestamps[:-1])[0]
    assert (len(artifact_index) > 0) == (len(artifact_dict) > 0)
    if len(artifact_dict) > 0:
        if "skip_start" in artifact_dict:
            mask = np.logical_and(
                timestamps >= artifact_dict["skip_start"],
                timestamps <= artifact_dict["skip_end"],
            )
            mask[artifact_index[0] - 1 :] = False
            timestamps = timestamps[~mask]
            cursor_pos = cursor_pos[~mask]
            hand_pos = hand_pos[~mask]
            eye_pos = eye_pos[~mask]

            artifact_index = np.where(timestamps[1:] < timestamps[:-1])[0]
            assert len(artifact_index) > 0

        assert len(artifact_dict["original_start_times"]) == len(artifact_index) + 1
        block_index = np.concatenate(
            [np.array([0]), artifact_index + 1, np.array([len(timestamps)])]
        )

        timestamps_blocks = [
            timestamps[block_index[i] : block_index[i + 1]]
            for i in range(len(block_index) - 1)
        ]
        for i in range(len(timestamps_blocks)):
            assert np.all(
                timestamps_blocks[i] >= artifact_dict["original_start_times"][i]
            )
            assert np.all(
                timestamps_blocks[i] <= artifact_dict["original_end_times"][i]
            )
            timestamps_blocks[i] = (
                timestamps_blocks[i]
                - artifact_dict["original_start_times"][i]
                + artifact_dict["fixed_start_times"][i]
            )

        timestamps = np.concatenate(timestamps_blocks)
        cursor_pos = np.concatenate(
            [
                cursor_pos[block_index[i] : block_index[i + 1]]
                for i in range(len(block_index) - 1)
            ]
        )
        hand_pos = np.concatenate(
            [
                hand_pos[block_index[i] : block_index[i + 1]]
                for i in range(len(block_index) - 1)
            ]
        )
        eye_pos = np.concatenate(
            [
                eye_pos[block_index[i] : block_index[i + 1]]
                for i in range(len(block_index) - 1)
            ]
        )

    # check that timestamps is sorted
    assert np.all(np.diff(timestamps) >= 0)

    # the behavior is not contiguous, so we need to determine the domain
    expected_period = 0.001  # the sampling rate of this data should be 1000 Hz
    domain_index = np.where((np.diff(timestamps) - expected_period) > 1e-4)[0]
    domain_start_index = np.insert(domain_index + 1, 0, 0)
    domain_end_index = np.append(domain_index, len(timestamps) - 1)
    domain = Interval(
        start=timestamps[domain_start_index], end=timestamps[domain_end_index] + 1e-4
    )
    assert domain.is_disjoint()
    assert domain.is_sorted()

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        domain=domain,
    )

    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos_2d=hand_pos,
        domain=domain,
    )

    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos=eye_pos,
        domain=domain,
    )

    def compute_gradient(data, timestamps, domain_start_index, domain_end_index):
        gradient = []
        # compute the velocity
        for start_index, end_index in zip(domain_start_index, domain_end_index):
            data_slice = data[start_index : end_index + 1]
            timestamps_slice = timestamps[start_index : end_index + 1]

            gradient.append(
                np.gradient(
                    data_slice,
                    timestamps_slice,
                    edge_order=1,
                    axis=0,
                )
            )
        gradient = np.concatenate(gradient)
        return gradient

    cursor.vel = compute_gradient(
        cursor.pos, timestamps, domain_start_index, domain_end_index
    )
    cursor.acc = compute_gradient(
        cursor.vel, timestamps, domain_start_index, domain_end_index
    )
    hand.vel_2d = compute_gradient(
        hand.pos_2d, timestamps, domain_start_index, domain_end_index
    )
    hand.acc_2d = compute_gradient(
        hand.vel_2d, timestamps, domain_start_index, domain_end_index
    )

    return cursor, hand, eye


def detect_outliers(cursor):
    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    cursor_acc_norm = np.linalg.norm(cursor.acc, axis=1)
    mask = cursor_acc_norm > 80000.0
    # we dilate the mask to make sure we are not missing any outliers
    structure = np.ones(50, dtype=bool)
    mask = binary_dilation(mask, structure)

    # convert to interval, you need to find the start and end of the outlier segments
    start = cursor.timestamps[np.where(np.diff(mask.astype(int)) == 1)[0]]
    if mask[0]:
        start = np.insert(start, 0, cursor.timestamps[0])

    end = cursor.timestamps[np.where(np.diff(mask.astype(int)) == -1)[0]]
    if mask[-1]:
        end = np.append(end, cursor.timestamps[-1])

    cursor_outlier_segments = Interval(start=start, end=end)
    assert cursor_outlier_segments.is_disjoint()
    return cursor_outlier_segments


def extract_spikes(nwbfile, trials, artifact_dict):
    # spikes
    timestamps = []
    unit_index = []

    # units
    unit_meta = []

    electrodes = nwbfile.units.electrodes.table
    trial_table = nwbfile.trials.to_dataframe()
    unit_ctr = 0
    num_skipped_units = 0
    # all these units are obtained using threshold crossings
    for i in range(len(nwbfile.units)):
        spikes_times = nwbfile.units[i].spike_times[i]

        artifact_index = np.where(spikes_times[1:] < spikes_times[:-1])[0]
        if len(artifact_dict) > 0:
            if "skip_start" in artifact_dict:
                mask = np.logical_and(
                    spikes_times >= artifact_dict["skip_start"],
                    spikes_times <= artifact_dict["skip_end"],
                )
                if len(artifact_index) > 0:
                    mask[artifact_index[0] + 1 :] = False
                    spikes_times = spikes_times[~mask]

                artifact_index = np.where(spikes_times[1:] < spikes_times[:-1])[0]

            if len(artifact_dict["original_start_times"]) < len(artifact_index) + 1:
                logging.warning(
                    f"Unit {i} has {len(artifact_index) + 1} non-contiguous blocks, but "
                    f"found {len(artifact_dict['original_start_times'])} in the "
                    f"trials. Skipping this unit."
                )
                num_skipped_units += 1
                continue
            block_index = np.concatenate(
                [np.array([0]), artifact_index + 1, np.array([len(spikes_times)])]
            )

            spikes_times_blocks = [
                spikes_times[block_index[j] : block_index[j + 1]]
                for j in range(len(block_index) - 1)
            ]

            flag = True
            for j in range(len(spikes_times_blocks)):
                if not (
                    np.all(
                        spikes_times_blocks[j]
                        >= artifact_dict["original_start_times"][j]
                    )
                    and np.all(
                        spikes_times_blocks[j] <= artifact_dict["original_end_times"][j]
                    )
                ):
                    logging.warning(
                        f"Unit {i} has a spike time block that is not contiguous. Skipping this block."
                    )
                    num_skipped_units += 1
                    flag = False
                    break

                spikes_times_blocks[j] = (
                    spikes_times_blocks[j]
                    - artifact_dict["original_start_times"][j]
                    + artifact_dict["fixed_start_times"][j]
                )

            if not flag:
                continue

            spikes_times = np.concatenate(spikes_times_blocks)

            obs_intervals = nwbfile.units[i].obs_intervals[i]
            assert np.allclose(
                obs_intervals[:, 0], trial_table["start_time"]
            ), f"Unit {i}"
            assert np.allclose(
                obs_intervals[:, 1], trial_table["stop_time"]
            ), f"Unit {i}"

        else:
            if len(artifact_index) != 0:
                logging.warning(
                    f"Unit {i} has {len(artifact_index)} non-contiguous blocks, but no "
                    f"artifact_dict. Skipping this unit."
                )
                num_skipped_units += 1
                continue

        assert np.all(np.diff(spikes_times) >= 0)

        # label unit
        group_name = electrodes["group_name"][i]
        unit_id = f"group_{group_name}/elec{i}"

        timestamps.append(spikes_times)

        if len(spikes_times) > 0:
            unit_index.append([unit_ctr] * len(spikes_times))

        # extract unit metadata
        unit_meta.append(
            {
                "id": unit_id,
                "unit_number": unit_ctr,
                "count": len(spikes_times),
                "type": int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
            }
        )

        unit_ctr += 1

    if num_skipped_units > 0:
        logging.warning(
            f"Unable to resolve {num_skipped_units} out of {len(nwbfile.units)} units. "
            f"These units were excluded."
        )

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(
        unit_meta_df,
        unsigned_to_long=True,
    )

    # concatenate spikes
    timestamps = np.concatenate(timestamps)
    unit_index = np.concatenate(unit_index)

    # create spikes object
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain=Interval(trials.start, trials.end),
    )
    spikes.sort()
    return spikes, units


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="churchland_shenoy_neural_2012",
        origin_version="dandi/000070/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000070",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd) using two 96 channel high density Utah Arrays (Blackrock Microsystems) "
        "while performing reaching tasks with right hand.",
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
    subject_id = subject.id
    device_id = f"{subject_id}_{recording_date}"
    session_id = f"{device_id}_center_out_reaching"

    # register session
    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        task=Task.REACHING,
    )

    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
    )

    # extract data about trial structure
    trials, movement_phases, artifact_dict = extract_trials(nwbfile, session_id)
    # extract spiking activity
    # this data is from dandi, we can use our helper function
    spikes, units = extract_spikes(nwbfile, trials, artifact_dict)

    # extract behavior
    cursor, hand, eye = extract_behavior(nwbfile, artifact_dict)
    assert len(cursor.domain) == len(trials)

    cursor_outlier_segments = detect_outliers(cursor)

    for key in movement_phases.keys():
        setattr(
            movement_phases,
            key,
            getattr(movement_phases, key).difference(cursor_outlier_segments),
        )

    # close file
    io.close()

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
        hand=hand,
        eye=eye,
        cursor_outlier_segments=cursor_outlier_segments,
        domain=cursor.domain | spikes.domain,
    )

    # split trials into train, validation and test
    successful_trials = trials.select_by_mask(trials.is_valid)
    assert successful_trials.is_disjoint()

    train_trials, valid_trials, test_trials = successful_trials.split(
        [0.7, 0.1, 0.2], shuffle=True, random_seed=42
    )

    # we will still use the invalid trials for training
    train_trials = train_trials | trials.select_by_mask(~trials.is_valid)

    data.set_train_domain(train_trials)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
