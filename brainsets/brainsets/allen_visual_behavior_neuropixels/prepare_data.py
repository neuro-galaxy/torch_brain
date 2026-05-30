"""Load data, processes it, save it."""

import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)

from brainsets_utils import DatasetBuilder

from brainsets_utils.taxonomy import (
    RecordingTech,
    Species,
    SubjectDescription,
    Sex,
    Task,
)

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def extract_spikes(session, prefix):
    units = session.units
    spiketimes_dict = session.spike_times

    spikes = []
    unit_index = []
    types = []
    # waveforms = []
    unit_meta = []

    for i, unit_id in enumerate(spiketimes_dict.keys()):
        metadata = units.loc[unit_id]
        probe_id = metadata["probe_id"]
        probe_channel_id = metadata["probe_channel_number"]
        unit_name = f"{prefix}/{probe_id}/{probe_channel_id}/{unit_id}"

        spiketimes = spiketimes_dict[unit_id]
        spikes.append(spiketimes)
        unit_index.append([i] * len(spiketimes))
        types.append(np.ones_like(spiketimes) * int(RecordingTech.NEUROPIXELS_SPIKES))

        unit_meta.append(
            {
                "count": len(spiketimes),
                "channel_name": probe_channel_id,
                "electrode_row": metadata["probe_horizontal_position"],
                "electrode_col": 0,
                "id": unit_name,
                "area_name": metadata["structure_acronym"],
                "channel_number": probe_channel_id,
                "unit_number": i,
                "type": int(RecordingTech.NEUROPIXELS_SPIKES),
            }
        )

    spikes = np.concatenate(spikes)
    # waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)
    types = np.concatenate(types)

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    # waveforms = waveforms[sorted]
    unit_index = unit_index[sorted]
    types = types[sorted]

    spikes = IrregularTimeSeries(
        domain="auto",
        timestamps=spikes,
        # waveforms=waveforms,
        unit_index=unit_index,
        types=types,
    )

    return spikes, units


def extract_natural_scenes(stimulus_pres):
    ns_presentations = stimulus_pres[
        np.array(stimulus_pres["stimulus_name"] == "natural_scenes")
        & np.array(stimulus_pres["frame"] != "null")
    ]

    if len(ns_presentations) == 0:
        return None

    start_times = ns_presentations["start_time"].values
    end_times = ns_presentations["stop_time"].values
    image_ids = ns_presentations["frame"].values.astype(np.int64)  # ids span -1 to 117
    image_ids = image_ids + 1  # now they span 0 to 118

    return Interval(
        start=start_times,
        end=end_times,
        image_ids=image_ids,
        timestamps=start_times / 2.0 + end_times / 2.0,
        timekeys=["start", "end", "timestamps"],
    )


def extract_gabors(stimulus_pres):
    gabors_presentations = stimulus_pres[
        np.array(stimulus_pres["stimulus_name"] == "gabors")
        & np.array(stimulus_pres["orientation"] != "null")
    ]

    unique_x_pos = np.unique(gabors_presentations.x_position)
    unique_y_pos = np.unique(gabors_presentations.y_position)
    unique_orientations = np.unique(gabors_presentations.orientation)
    unique_x_pos.sort()
    unique_y_pos.sort()
    unique_orientations.sort()

    def calculate_gabors_ori(row):
        gabors_or_map = {"0": 0, "45": 1, "90": 2}
        return gabors_or_map[str(int(row["orientation"]))]

    def calculate_gabors_pos2d(row):
        x_class = np.where(unique_x_pos == row["x_position"])[0][0]
        y_class = np.where(unique_y_pos == row["y_position"])[0][0]
        return np.array([x_class, y_class])

    gabors_start_times = gabors_presentations["start_time"].values
    gabors_end_times = gabors_presentations["stop_time"].values

    gabors_trials = Interval(
        start=gabors_start_times,
        end=gabors_end_times,
        pos_2d=np.vstack(
            gabors_presentations.apply(calculate_gabors_pos2d, axis=1).to_numpy()
        ),  # (N, 2)
        gabors_orientation=gabors_presentations.apply(
            calculate_gabors_ori, axis=1
        ).to_numpy(),
        timestamps=gabors_start_times / 2.0 + gabors_end_times / 2.0,
        timekeys=["start", "end", "timestamps"],
        # other data that might be useful later
        orientation=gabors_presentations["orientation"].values.astype(
            np.float32
        ),  # (N,)
        spatial_frequency=gabors_presentations["spatial_frequency"].values.astype(
            np.float32
        ),
        temporal_frequency=gabors_presentations["temporal_frequency"].values.astype(
            np.float32
        ),
        x_position=gabors_presentations["x_position"].values.astype(np.float32),
        y_position=gabors_presentations["y_position"].values.astype(np.float32),
    )
    return gabors_trials


def extract_running_speed(session):
    """
    Extracts running_speed data from the given session.
    Args:
        session: The session object containing the running_speed data.
    Returns:
        running_speed_obj: An IrregularTimeSeries object representing the running speed running_speed.
    """
    running_speed_obj = None
    running_speed_df = session.running_speed
    if running_speed_df is not None:
        running_speed_df = running_speed_df[~running_speed_df.isnull().any(axis=1)]
        running_speed_times = (
            running_speed_df["start_time"]
            + (running_speed_df["end_time"] - running_speed_df["start_time"]) / 2
        )
        running_speed_obj = IrregularTimeSeries(
            domain="auto",
            timestamps=running_speed_times.values,
            running_speed=running_speed_df["velocity"]
            .values.astype(np.float32)
            .reshape(-1, 1),  # continues values needs to be 2 dimensional
        )
    # NOTE: we don't normalize or z-scale the target values here,
    # as they should be done post-processing, seperately using brainsets/calculate_zscales.py
    # and updated into the dataset config yaml manually for the model to scale dynamically
    return running_speed_obj


def extract_gaze(session):
    gaze_df = session.get_screen_gaze_data(include_filtered_data=True)

    if gaze_df is None:
        return None

    # Filter out rows with nan values
    gaze_df = gaze_df[~gaze_df.isnull().any(axis=1)]

    gaze_obj = IrregularTimeSeries(
        domain="auto",
        timestamps=gaze_df.index.values,
        raw_eye_area=gaze_df["raw_eye_area"].values.astype(np.float32),
        raw_pupil_area=gaze_df["raw_pupil_area"].values.astype(np.float32),
        raw_screen_coordinates_x_cm=gaze_df[
            "raw_screen_coordinates_x_cm"
        ].values.astype(np.float32),
        raw_screen_coordinates_y_cm=gaze_df[
            "raw_screen_coordinates_y_cm"
        ].values.astype(np.float32),
        raw_screen_coordinates_spherical_x_deg=gaze_df[
            "raw_screen_coordinates_spherical_x_deg"
        ].values.astype(np.float32),
        raw_screen_coordinates_spherical_y_deg=gaze_df[
            "raw_screen_coordinates_spherical_y_deg"
        ].values.astype(np.float32),
        filtered_eye_area=gaze_df["filtered_eye_area"].values.astype(np.float32),
        filtered_pupil_area=gaze_df["filtered_pupil_area"].values.astype(np.float32),
        filtered_screen_coordinates_x_cm=gaze_df[
            "filtered_screen_coordinates_x_cm"
        ].values.astype(np.float32),
        filtered_screen_coordinates_y_cm=gaze_df[
            "filtered_screen_coordinates_y_cm"
        ].values.astype(np.float32),
        filtered_screen_coordinates_spherical_x_deg=gaze_df[
            "filtered_screen_coordinates_spherical_x_deg"
        ].values.astype(np.float32),
        filtered_screen_coordinates_spherical_y_deg=gaze_df[
            "filtered_screen_coordinates_spherical_y_deg"
        ].values.astype(np.float32),
    )

    # store in pos_2d attribute
    # NOTE: we don't normalize or z-scale the target values here,
    # as they should be done post-processing, seperately using brainsets/calculate_zscales.py
    # and updated into the dataset config yaml manually for the model to scale dynamically
    gaze_obj.pos_2d = np.stack(
        [
            gaze_obj.filtered_screen_coordinates_x_cm,
            gaze_obj.filtered_screen_coordinates_y_cm,
        ],
        axis=-1,
    )  # (N, 2)

    return gaze_obj


def extract_pupil(session):
    pupil_df = session.get_pupil_data()
    if pupil_df is None:
        return None

    pupil_df = pupil_df[~pupil_df.isnull().any(axis=1)]
    pupil_obj = IrregularTimeSeries(
        domain="auto",
        timestamps=pupil_df.index.values,
        corneal_reflection_center_x=pupil_df[
            "corneal_reflection_center_x"
        ].values.astype(np.float32),
        corneal_reflection_center_y=pupil_df[
            "corneal_reflection_center_y"
        ].values.astype(np.float32),
        corneal_reflection_height=pupil_df["corneal_reflection_height"].values.astype(
            np.float32
        ),
        corneal_reflection_width=pupil_df["corneal_reflection_width"].values.astype(
            np.float32
        ),
        corneal_reflection_phi=pupil_df["corneal_reflection_phi"].values.astype(
            np.float32
        ),
        pupil_center_x=pupil_df["pupil_center_x"].values.astype(np.float32),
        pupil_center_y=pupil_df["pupil_center_y"].values.astype(np.float32),
        pupil_height=pupil_df["pupil_height"].values.astype(np.float32),
        pupil_width=pupil_df["pupil_width"].values.astype(np.float32),
        pupil_phi=pupil_df["pupil_phi"].values.astype(np.float32),
        eye_center_x=pupil_df["eye_center_x"].values.astype(np.float32),
        eye_center_y=pupil_df["eye_center_y"].values.astype(np.float32),
        eye_height=pupil_df["eye_height"].values.astype(np.float32),
        eye_width=pupil_df["eye_width"].values.astype(np.float32),
        eye_phi=pupil_df["eye_phi"].values.astype(np.float32),
    )

    # NOTE: we don't normalize or z-scale the target values here,
    # as they should be done post-processing, seperately using brainsets/calculate_zscales.py
    # and updated into the dataset config yaml manually for the model to scale dynamically
    pupil_obj.size_2d = np.stack(
        [pupil_obj.pupil_height, pupil_obj.pupil_width], axis=-1
    )  # (N, 2)

    return pupil_obj


def extract_static_gratings(stimulus_pres):
    static_gratings = stimulus_pres[
        (stimulus_pres["stimulus_name"] == "static_gratings")
        & (stimulus_pres["orientation"] != "null")
    ]

    start_times = static_gratings["start_time"].values
    end_times = static_gratings["stop_time"].values
    orientations = static_gratings["orientation"].values.astype(np.float32)
    orientation_classes = np.round(orientations / 30.0).astype(np.int64)
    output_timestamps = (start_times + end_times) / 2

    return Interval(
        start=start_times,
        end=end_times,
        orientation=orientation_classes,  # (N,)
        timestamps=output_timestamps,  # (N,)
        timekeys=["start", "end", "timestamps"],
    )


def extract_drifting_gratings(stimulus_pres):
    drifting_gratings = stimulus_pres[
        (
            (
                stimulus_pres["stimulus_name"] == "drifting_gratings"
            )  # brain_observatory_1.1
            | (
                stimulus_pres["stimulus_name"] == "drifting_gratings_75_repeats"
            )  # functional_connectivity
        )
        & (stimulus_pres["orientation"] != "null")
    ]

    start_times = drifting_gratings["start_time"].values
    end_times = drifting_gratings["stop_time"].values
    orientations = drifting_gratings["orientation"].values.astype(np.float32)
    orientations = np.round(orientations / 45.0).astype(np.int64)
    temp_freq_mapping = {1.0: 0, 2.0: 1, 4.0: 2, 8.0: 3, 15.0: 4}
    temp_freq = np.array(
        [
            temp_freq_mapping[freq]
            for freq in drifting_gratings["temporal_frequency"].values
        ]
    )
    drifting_gratings_obj = Interval(
        start=start_times,
        end=end_times,
        orientation=orientations,  # (N,)
        temp_freq=temp_freq,  # (N,)
        # NOTE for now, we will center all timestamps assuming a context window of 1s
        timestamps=np.ones_like(start_times) * 0.5,
    )
    assert np.all(
        drifting_gratings_obj.end - drifting_gratings_obj.start > 1
    ), "All trials must have a duration greater than 1."
    return drifting_gratings_obj


def get_behavior_region(running_speed_obj, pupil_obj=None, gaze_obj=None):
    # # extract session start and end times
    session_start = min(
        running_speed_obj.timestamps.min() if running_speed_obj is not None else np.inf,
        pupil_obj.timestamps.min() if pupil_obj is not None else np.inf,
        gaze_obj.timestamps.min() if gaze_obj is not None else np.inf,
    )
    session_end = max(
        running_speed_obj.timestamps.max() if running_speed_obj is not None else 0,
        pupil_obj.timestamps.max() if pupil_obj is not None else 0,
        gaze_obj.timestamps.max() if gaze_obj is not None else 0,
    )
    assert (
        session_start < session_end
    ), "Atleast one of running_speed, pupil or gaze data must be present."
    return session_start, session_end


def get_drifting_gratings_splits(drifting_gratings_obj):
    if drifting_gratings_obj is None or len(drifting_gratings_obj) == 0:
        return {"train": None, "valid": None, "test": None}
    train_trials, valid_trials, test_trials = drifting_gratings_obj.split(
        [0.7, 0.1, 0.2]
    )
    return {"train": train_trials, "valid": valid_trials, "test": test_trials}


def get_static_gratings_splits(static_gratings_obj):
    if static_gratings_obj is None or len(static_gratings_obj) == 0:
        return {"train": None, "valid": None, "test": None}
    train_trials, valid_trials, test_trials = static_gratings_obj.split([0.7, 0.1, 0.2])
    splits_dict = {}
    for split, split_trial in [
        ("train", train_trials),
        ("valid", valid_trials),
        ("test", test_trials),
    ]:
        split_trial = split_trial.coalesce()
        mask = (split_trial.end - split_trial.start) >= 1.0
        # even after coalescing, some trials might be less than 1 second long
        # so we filter them out
        split_trial = split_trial.select_by_mask(mask)
        assert (
            split_trial.end - split_trial.start > 1.0
        ).all(), "Split trials must be at least 1 second long."
        splits_dict[split] = split_trial
    return splits_dict


get_natural_scenes_splits = get_static_gratings_splits  # same splitting logic


def get_gabors_splits(gabors_obj, split_ratios=[0.7, 0.1, 0.2]):
    if gabors_obj is None or len(gabors_obj) == 0:
        return {"train": None, "valid": None, "test": None}
    import math

    train_boundary = math.floor(len(gabors_obj) * split_ratios[0])
    valid_boundary = math.floor(len(gabors_obj) * (split_ratios[0] + split_ratios[1]))
    test_boundary = math.floor(len(gabors_obj) * sum(split_ratios))
    train_trials = Interval(
        start=np.array([gabors_obj.start[0]]),
        end=np.array([gabors_obj.end[train_boundary - 1]]),
    )
    valid_trials = Interval(
        start=np.array([gabors_obj.start[train_boundary]]),
        end=np.array([gabors_obj.end[valid_boundary - 1]]),
    )
    test_trials = Interval(
        start=np.array([gabors_obj.start[valid_boundary]]),
        end=np.array([gabors_obj.end[test_boundary - 1]]),
    )
    return {"train": train_trials, "valid": valid_trials, "test": test_trials}


def collate_splits(
    stimuli_splits_by_key: Dict[str, dict],
    supervision_dict: dict,
    include_all_behavior: bool = False,
) -> tuple[Dict[str, Interval], float, float]:
    """
    Collates the splits of stimuli data and behavior data into a single dictionary.

    This helper function takes a dictionary of splits for different stimuli types and behavior data,
    and combines them into a single dictionary. It also calculates the session start and end times.

    Args:
        stimuli_splits_by_key (Dict[str, dict]): A dictionary containing splits for different stimuli types
            and behavior data. The keys are the stimuli types and behavior, and the values are dictionaries
            containing the train, valid, and test splits.
        supervision_dict (dict): A dictionary containing the behavior data.
        include_all_behavior (bool): A flag indicating whether to include the residual behavior regions in each session.
        Behavior data is available distributed across the entire session. Therefore, all the stimuli patches also contain behavior data.
        If this flag is not set, then the behavior information (running speed, gaze, pupil) is taken from only from those
        patches of the train, valid, and test splits of the stimuli data.

    Returns:
        Tuple[Dict[str, IntervalSet], float, float]: A tuple containing the final splits dictionary,
            the session start time, and the session end time. The final splits dictionary contains the
            combined train, valid, and test splits for each stimuli type and behavior. The session start
            and end times represent the overall start and end times of the session.

    """
    final_splits_dict = {}
    every_interval_by_key = {}

    for key, splits in stimuli_splits_by_key.items():
        every_interval_by_key[f"{key}:train"] = splits["train"]
        every_interval_by_key[f"{key}:valid"] = splits["valid"]
        every_interval_by_key[f"{key}:test"] = splits["test"]

        if not all([splits["train"], splits["valid"], splits["test"]]):
            continue

        final_splits_dict["train"] = (
            final_splits_dict["train"] | splits["train"]
            if final_splits_dict.get("train")
            else splits["train"]
        )
        final_splits_dict["valid"] = (
            final_splits_dict["valid"] | splits["valid"]
            if final_splits_dict.get("valid")
            else splits["valid"]
        )
        final_splits_dict["test"] = (
            final_splits_dict["test"] | splits["test"]
            if final_splits_dict.get("test")
            else splits["test"]
        )

    if include_all_behavior:
        # include all behavior regions in the training data
        behavior_start, behavior_end = get_behavior_region(
            supervision_dict.get("running_speed", None),
            supervision_dict.get("pupil", None),
            supervision_dict.get("gaze", None),
        )
        behavior_region_interval = Interval(start=behavior_start, end=behavior_end)
        # remove the patches that are used for validation and testing from the remaining trainable region
        remaining_trainable_region = behavior_region_interval.difference(
            final_splits_dict["valid"] | final_splits_dict["test"]
        )

        every_interval_by_key["remaining_trainable_region_for_behavior"] = (
            remaining_trainable_region
        )

        final_splits_dict["train"] = (
            final_splits_dict["train"] | remaining_trainable_region
        )

    session_start = min(s.start.min() for s in final_splits_dict.values())
    session_end = max(s.end.max() for s in final_splits_dict.values())

    return final_splits_dict, session_start, session_end


def main():

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument(
        "--include_all_behavior",
        action="store_true",
        help="Include residual regions in each session that have only behavior information as part of the trainable data.",
    )

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="allen_visual_behavior_neuropixels_2019",
        origin_version="v2",  # allensdk version
        derived_version="0.0.1",  # This variant
        metadata_version="0.0.1",
        source="http://api.brain-map.org/",
        description="Visual Coding - Neuropixels from Allen Brain Observatory (2019).",
    )

    # get the project cache from the warehouse
    manifest_path = os.path.join(args.input_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    # get sessions
    sessions = cache.get_session_table()

    # looping through all sessions, there are 58 in total
    for session_id, row in tqdm(sessions.iterrows()):
        # load nwb file through the allen sdk
        session_data = cache.get_session_data(session_id)
        stimulus_presentations = session_data.stimulus_presentations

        logging.info(f"Processing session: {session_id}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:

            # extract subject metadata
            animal = f"mouse_{row['specimen_id']}"
            sex_map = {"M": Sex.MALE, "F": Sex.FEMALE}
            subject = SubjectDescription(
                id=str(row["specimen_id"]),
                species=Species.MUS_MUSCULUS,
                age=row["age_in_days"],
                sex=sex_map[row["sex"]],
                genotype=row["full_genotype"],
            )

            session.register_subject(subject)

            # extract experimental metadata
            # there is only one session per subject.
            recording_date = session_data.session_start_time.strftime("%Y%m%d")
            sortset_id = f"{animal}_{recording_date}"

            # extract spiking activity
            spikes, units = extract_spikes(session_data, prefix=sortset_id)

            # extract behavior and stimuli data
            # using dedicated extract_* helpers into a dictionary
            supervision_dict = {
                "running_speed": extract_running_speed(session_data),
                "gaze": extract_gaze(session_data),
                "pupil": extract_pupil(session_data),
                "drifting_gratings": extract_drifting_gratings(stimulus_presentations),
                "static_gratings": extract_static_gratings(stimulus_presentations),
                "gabors": extract_gabors(stimulus_presentations),
                "natural_scenes": extract_natural_scenes(stimulus_presentations),
            }

            # register session
            session.register_session(
                id=str(session_id),
                recording_date=recording_date,
                task=Task.DISCRETE_VISUAL_CODING,
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # split each stimuli/behavior and combine them
            # using dedicated get_*_splits helpers into a dictionary
            stimuli_splits_by_key = {
                "drifting_gratings": get_drifting_gratings_splits(
                    supervision_dict.get("drifting_gratings", None)
                ),
                "static_gratings": get_static_gratings_splits(
                    supervision_dict.get("static_gratings", None)
                ),
                "gabors": get_gabors_splits(supervision_dict.get("gabors", None)),
                "natural_scenes": get_natural_scenes_splits(
                    supervision_dict.get("natural_scenes", None)
                ),
            }

            final_splits_dict, session_start, session_end = collate_splits(
                stimuli_splits_by_key,
                supervision_dict,
                include_all_behavior=args.include_all_behavior,
            )

            data = Data(
                # metadata
                domain=Interval(start=session_start, end=session_end),
                session=session_id,
                sortset=sortset_id,
                subject=subject.id,
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                **supervision_dict,
            )

            session.register_data(data)

            session.register_split("train", final_splits_dict["train"])
            session.register_split("valid", final_splits_dict["valid"])
            session.register_split("test", final_splits_dict["test"])

            session.save_to_disk()
            logging.info(f"Saved to disk session: {session_id}")

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
