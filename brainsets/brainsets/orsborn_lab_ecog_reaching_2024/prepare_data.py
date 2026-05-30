import argparse
import datetime
import logging

import numpy as np
import pandas as pd
import copy
import mne
import pickle
import os

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.taxonomy import RecordingTech, Task
from brainsets_utils.taxonomy import (
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    SubjectDescription,
    Task,
    RecordingTech,
    Species,
    Sex,
    Macaque,
)

logging.basicConfig(level=logging.INFO)


def extract_concat_ecog_cursor(
    data_dict, decimate_signal=None, decimate_further_behavior=None, raw_signal=False
):
    """
    this dataset already has chopped trajectories of each trial,
    but we concat them together to be compatible with DatasetBuilder

    Between each trials, we add dummy number (-10000.) as sanity check.

    Return the concated ecog and cursor movements for this data files (one subject data across multiple days)
    """

    ecog_segments = data_dict["lfp_segments"]
    traj_segments = data_dict["kin_segments"]
    sampling_rate = data_dict["samplerate"]
    ecog_list = []
    traj_list = []
    trial_ranges = []
    time_idx = 0
    invalid_number = -1e4
    total_length = 0
    iir_params = dict(order=4, ftype="butter", output="sos")
    for i in range(len(ecog_segments)):

        # signal filtering
        ecog_loaded = copy.deepcopy(ecog_segments[i])
        ecog_loaded = ecog_loaded.astype(float)

        # shift to 0 mean
        ecog_loaded = ecog_loaded - np.mean(ecog_loaded, axis=0, keepdims=True)

        info = mne.create_info(
            ch_names=ecog_loaded.shape[1],
            sfreq=sampling_rate,
            ch_types=["ecog"] * ecog_loaded.shape[1],
        )
        ecog_raw = mne.io.RawArray(ecog_loaded.T, info)
        if not raw_signal:
            # here use fir with reflect pad
            ecog_p = ecog_raw.filter(
                70,
                150,
                method="fir",
                phase="zero-double",
                pad="reflect",
            ).apply_hilbert(envelope=True)
        else:
            ecog_p = ecog_raw
        if decimate_signal is None:
            ecog_proc = ecog_p.get_data().T  # length x n_channels
            traj_proc = traj_segments[i]  # length x 2
            sampling_rate_proc = sampling_rate
        else:
            assert isinstance(decimate_signal, int), "decimate factor must be int"
            assert (
                decimate_signal < sampling_rate
            ), "decimate factor cannot over sampling rate"
            target_fs = sampling_rate / decimate_signal
            decim = np.round(sampling_rate / target_fs).astype(int)
            obtained_fs = sampling_rate / decim
            ecog_p_lp = ecog_p.filter(None, obtained_fs / 3.0)

            # decimate
            sampling_rate_proc = obtained_fs
            ecog_proc = ecog_p_lp.get_data().T
            traj_proc = traj_segments[i]
            decimated_time = np.arange(0, ecog_proc.shape[0], decim).astype(int)
            ecog_proc = ecog_proc[decimated_time]
            traj_proc = traj_proc[decimated_time]

        valid_trajectory_length = ecog_proc.shape[0]
        ecog_list.append(
            np.pad(
                ecog_proc, ((1, 1), (0, 0)), "constant", constant_values=invalid_number
            )
        )
        traj_list.append(
            np.pad(
                traj_proc, ((1, 1), (0, 0)), "constant", constant_values=invalid_number
            )
        )
        trial_ranges.append((time_idx + 1, time_idx + 1 + valid_trajectory_length))
        time_idx += 1 + valid_trajectory_length + 1  # skip a whole padded length
        total_length += ecog_list[-1].shape[0]
    ecog_cat = np.concatenate(ecog_list, axis=0)
    traj_cat = np.concatenate(traj_list, axis=0)

    # sanity check
    mask = np.zeros(total_length).astype(bool)
    for trial_interval in trial_ranges:
        trial_start, trial_end = trial_interval
        mask[trial_start:trial_end] = True
    assert (ecog_cat[~mask] == invalid_number).all() and (
        ecog_cat[mask] != invalid_number
    ).all()
    assert (traj_cat[~mask] == invalid_number).all() and (
        traj_cat[mask] != invalid_number
    ).all()

    assert total_length == ecog_cat.shape[0]
    assert total_length == traj_cat.shape[0]

    concat_ecog_signals = RegularTimeSeries(
        ecogs=ecog_cat,  # dim (total time length x total channels)
        sampling_rate=sampling_rate_proc,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(ecog_cat) - 1) / sampling_rate_proc]),
        ),
    )

    subtask_index = np.ones_like(ecog_cat[:, 0], dtype=np.int64) * int(
        Task.REACHING.REACH
    )
    subtask_index[ecog_cat[:, 0] == invalid_number] = int(Task.REACHING.INVALID)

    implied_timestamps = np.array([0.0]) + np.arange(total_length) / sampling_rate_proc

    # if we want to further downsample behavior data, we will decimate carefully in the concatenated vector,
    # to keep that the added dummy boundary points have the same timestamps
    if not (decimate_further_behavior is None):
        assert isinstance(decimate_further_behavior, int)
        behavior_mask = np.zeros_like(mask).astype(bool)
        for trial_interval in trial_ranges:
            trial_start, trial_end = trial_interval
            for keep_trial_idx in np.arange(
                trial_start + 1, trial_end, decimate_further_behavior
            ):
                behavior_mask[int(keep_trial_idx)] = True

        assert np.sum(~mask & behavior_mask) == 0
        implied_timestamps = implied_timestamps[behavior_mask]
        traj_cat = traj_cat[behavior_mask]
        assert all(subtask_index[behavior_mask] != int(Task.REACHING.INVALID))
        subtask_index = subtask_index[behavior_mask]

    concat_behavior = IrregularTimeSeries(
        timestamps=implied_timestamps,
        pos=traj_cat,  # dim (total time length x 2)
        subtask_index=subtask_index,
        domain="auto",
    )

    # channel info
    recording_tech = RecordingTech.MICRO_ECOG_ARRAY_ECOGS
    channel_meta = []

    for i in range(ecog_cat.shape[1]):
        channel_id = f"group_microECoGArray/channel_{i}"
        channel_meta.append(
            {
                "id": channel_id,
                "unit_number": i,
                "count": -1,
                "type": int(recording_tech),
            }
        )

    channel_meta_df = pd.DataFrame(channel_meta)  # list of dicts to dataframe
    channels = ArrayDict.from_dataframe(
        channel_meta_df,
        unsigned_to_long=True,
    )

    # make trial_ranges as Intervals
    trial_start = (
        np.array(trial_ranges)[:, 0].astype(float) + 1
    ) / sampling_rate_proc  # convert to second
    trial_end = np.array(trial_ranges)[:, 1].astype(float) / sampling_rate_proc
    reaching_trials = Interval(
        start=trial_start,
        end=trial_end,
        tasks=np.array([int(Task.REACHING.REACH)] * len(trial_ranges)),
        timekeys=["start", "end"],
    )
    assert reaching_trials.is_disjoint()

    return concat_ecog_signals, concat_behavior, reaching_trials, channels


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
        experiment_name="orsborn_lab_ecog_reaching_2024",
        origin_version="1.0.0",  # this is not from dandi
        derived_version="1.0.0",
        source="private",
        description="This dataset contains ECoG and behavioral data from "
        "two monkey subjects performing a center-out task "
        "ECoG was recorded from 240 channel micro ECOG array over motor cortex",
    )

    # iterate over the pickle files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".pkl"):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            f = open(file_path, "rb")
            pklfile = pickle.load(f)
            base_file_path = os.path.basename(file_path)

            # extract subject metadata
            subject = SubjectDescription(
                id=(
                    "Affogato"
                    if base_file_path.split("_")[0].startswith("a")
                    else "Beignet"
                ),
                species=Species.MACACA_MULATTA,
                sex=Sex.UNKNOWN,
            )
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = str(base_file_path.split("_")[1])  # take the start time
            sortset_id = f"{subject.id}_{recording_date}"
            task = "center_out_reaching"
            session_id = f"{sortset_id}_{task}"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y-%m-%d"),
                task=Task.REACHING,
            )

            # extract ecog signals and cursor positions
            (
                concat_ecog_signals,
                concat_cursor_movements,
                successful_trials,
                channels,
            ) = extract_concat_ecog_cursor(
                pklfile, decimate_signal=5, decimate_further_behavior=1
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=channels,
            )

            # close file
            f.close()

            # register session
            data = Data(
                # ecog
                signals=concat_ecog_signals,
                units=channels,
                # behavior
                cursor=concat_cursor_movements,
                # domain
                domain=concat_cursor_movements.domain,
            )

            session.register_data(data)

            # split trials into train, validation and test
            train_trials, valid_trials, test_trials = successful_trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
