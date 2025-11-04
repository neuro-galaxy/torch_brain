import argparse
import h5py
import logging
import os

import numpy as np
import pandas as pd

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.core.brain_observatory_nwb_data_set import (
    NoEyeTrackingException,
    EpochSeparationException,
)

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species, Sex, Cre_line, RecordingTech
from brainsets.taxonomy.mice import BrainRegion
from brainsets.taxonomy.allen import (
    TEMPORAL_FREQ_5_map,
    SPATIAL_FREQ_5_map,
    ORIENTATION_12_CLASSES_map,
    ORIENTATION_8_CLASSES_map,
    PHASE_4_map,
)
from brainsets import serialize_fn_map
from brainsets.utils.split import generate_train_valid_test_splits

logging.basicConfig(level=logging.INFO)


def extract_calcium_traces(nwbfile):
    timestamps, traces = nwbfile.get_dff_traces()
    traces = np.transpose(traces)

    timestamps_diff = np.diff(timestamps)
    period = np.mean(timestamps_diff)
    if not np.allclose(timestamps_diff, period, rtol=1e-03, atol=1e-02):
        raise ValueError(
            f"Timestamps are not uniformly spaced, found a deviation of {timestamps_diff.max()-timestamps_diff.min()}."
        )
    sampling_rate = 1.0 / period

    calcium_traces = RegularTimeSeries(
        df_over_f=np.array(traces),
        sampling_rate=sampling_rate,
        domain="auto",
        domain_start=timestamps[0],
    )

    return calcium_traces


def extract_units(nwbfile):
    roi_ids = nwbfile.get_roi_ids()
    roi_masks = nwbfile.get_roi_mask()
    num_rois = len(roi_ids)

    # compute center of the mask, and bounding box
    unit_position = np.zeros((num_rois, 2))
    unit_area = np.zeros(num_rois)
    unit_height = np.zeros(num_rois)
    unit_width = np.zeros(num_rois)

    for i, roi_mask in enumerate(roi_masks):
        roi_mask_numpy = roi_mask.get_mask_plane()
        rows, cols = np.nonzero(roi_mask_numpy)
        unit_position[i] = [np.mean(rows), np.mean(cols)]
        unit_height[i] = np.max(rows) - np.min(rows) + 1
        unit_width[i] = np.max(cols) - np.min(cols) + 1
        unit_area[i] = len(rows)

    units = ArrayDict(
        id=roi_ids.astype(str),
        imaging_plane_xy=unit_position,
        imaging_plane_area=unit_area,
        imaging_plane_width=unit_width,
        imaging_plane_height=unit_height,
    )

    return units


def extract_drifting_gratings(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    drifting_gratings_df = nwbfile.get_stimulus_table("drifting_gratings")

    # drop blank sweeps
    drifting_gratings_df = drifting_gratings_df[
        drifting_gratings_df["blank_sweep"] == 0.0
    ]

    drifting_gratings_df = drifting_gratings_df.loc[
        :, ["start", "end", "temporal_frequency", "orientation"]
    ]

    # convert frames to timestamps
    drifting_gratings_df["start"] = timestamps[drifting_gratings_df["start"].values]
    drifting_gratings_df["end"] = timestamps[drifting_gratings_df["end"].values]

    drifting_gratings = Interval.from_dataframe(drifting_gratings_df)
    drifting_gratings.timestamps = (drifting_gratings.start + drifting_gratings.end) / 2
    drifting_gratings.register_timekey("timestamps")

    # quantize orientation, temporal and spatial frequency
    drifting_gratings.orientation_id = np.array(
        [
            ORIENTATION_8_CLASSES_map[orientation]
            for orientation in drifting_gratings.orientation
        ],
        dtype=np.int64,
    )

    drifting_gratings.temporal_frequency_id = np.array(
        [
            TEMPORAL_FREQ_5_map[round(float(temporal_frequency), 0)]
            for temporal_frequency in drifting_gratings.temporal_frequency
        ],
        dtype=np.int64,
    )

    return drifting_gratings


def extract_static_grating(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    static_gratings_df = nwbfile.get_stimulus_table("static_gratings")

    # drop blank sweeps
    static_gratings_df = static_gratings_df[~pd.isna(static_gratings_df["orientation"])]

    static_gratings_df = static_gratings_df.loc[
        :, ["start", "end", "orientation", "spatial_frequency", "phase"]
    ]

    # convert frames to timestamps
    static_gratings_df["start"] = timestamps[static_gratings_df["start"].values]
    static_gratings_df["end"] = timestamps[static_gratings_df["end"].values]

    static_gratings = Interval.from_dataframe(static_gratings_df)
    static_gratings.timestamps = (static_gratings.start + static_gratings.end) / 2
    static_gratings.register_timekey("timestamps")

    # quantize orientation, temporal and spatial frequency
    static_gratings.orientation_id = np.array(
        [
            ORIENTATION_12_CLASSES_map[orientation]
            for orientation in static_gratings.orientation
        ],
        dtype=np.int64,
    )

    static_gratings.spatial_frequency_id = np.array(
        [
            SPATIAL_FREQ_5_map[round(float(spatial_frequency), 2)]
            for spatial_frequency in static_gratings.spatial_frequency
        ],
        dtype=np.int64,
    )

    static_gratings.phase_id = np.array(
        [PHASE_4_map[phase * 360] for phase in static_gratings.phase], dtype=np.int64
    )

    return static_gratings


def extract_natural_movie_one(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    natural_movie_one_df = nwbfile.get_stimulus_table("natural_movie_one")

    # convert frames to timestamps
    natural_movie_one_df["start"] = timestamps[natural_movie_one_df["start"].values]
    natural_movie_one_df["end"] = timestamps[natural_movie_one_df["end"].values]

    natural_movie_one_df = natural_movie_one_df.loc[
        :, ["start", "end", "frame", "repeat"]
    ]

    natural_movie_one = Interval.from_dataframe(natural_movie_one_df)
    natural_movie_one.timestamps = (natural_movie_one.start + natural_movie_one.end) / 2
    natural_movie_one.register_timekey("timestamps")

    return natural_movie_one


def extract_natural_movie_two(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    natural_movie_two_df = nwbfile.get_stimulus_table("natural_movie_two")

    # convert frames to timestamps
    natural_movie_two_df["start"] = timestamps[natural_movie_two_df["start"].values]
    natural_movie_two_df["end"] = timestamps[natural_movie_two_df["end"].values]

    natural_movie_two_df = natural_movie_two_df.loc[
        :, ["start", "end", "frame", "repeat"]
    ]

    print(natural_movie_two_df)
    natural_movie_two = Interval.from_dataframe(natural_movie_two_df)
    natural_movie_two.timestamps = (natural_movie_two.start + natural_movie_two.end) / 2
    natural_movie_two.register_timekey("timestamps")

    return natural_movie_two


def extract_natural_movie_three(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    natural_movie_three_df = nwbfile.get_stimulus_table("natural_movie_three")

    # convert frames to timestamps
    natural_movie_three_df["start"] = timestamps[natural_movie_three_df["start"].values]
    natural_movie_three_df["end"] = timestamps[natural_movie_three_df["end"].values]

    natural_movie_three_df = natural_movie_three_df.loc[
        :, ["start", "end", "frame", "repeat"]
    ]

    natural_movie_three = Interval.from_dataframe(natural_movie_three_df)
    natural_movie_three.timestamps = (
        natural_movie_three.start + natural_movie_three.end
    ) / 2
    natural_movie_three.register_timekey("timestamps")

    return natural_movie_three


def extract_natural_scenes(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    natural_scenes_df = nwbfile.get_stimulus_table("natural_scenes")

    # convert frames to timestamps
    natural_scenes_df["start"] = timestamps[natural_scenes_df["start"].values]
    natural_scenes_df["end"] = timestamps[natural_scenes_df["end"].values]

    natural_scenes_df = natural_scenes_df.loc[:, ["start", "end", "frame"]]
    natural_scenes = Interval.from_dataframe(natural_scenes_df)

    natural_scenes.timestamps = (natural_scenes.start + natural_scenes.end) / 2
    natural_scenes.register_timekey("timestamps")

    natural_scenes.frame = (natural_scenes.frame + 1).astype(np.int64)

    return natural_scenes


def extract_locally_sparse_noise(nwbfile, session_type):
    r"""The Locally Sparse Noise stimulus consisted of a 16 x 28 array of pixels, each 4.65 degrees on a side.
    For each frame of the stimulus (which was presented for 0.25 seconds), a small number of pixels were white and a small number were black, while the rest were mean gray.
    Each frame of the stimulus has ~11 spots (mean 11.4 ± 1.3 st dev) including both white and black.
    """
    timestamps, _ = nwbfile.get_dff_traces()
    if session_type == "three_session_C":
        locally_sparse_noise_df = nwbfile.get_stimulus_table("locally_sparse_noise")
        locally_sparse_noise_df["pixel_size"] = 4.65
    elif session_type == "three_session_C2":
        locally_sparse_noise_4deg_df = nwbfile.get_stimulus_table(
            "locally_sparse_noise_4deg"
        )
        locally_sparse_noise_4deg_df["pixel_size"] = 4.65
        locally_sparse_noise_8deg_df = nwbfile.get_stimulus_table(
            "locally_sparse_noise_8deg"
        )
        locally_sparse_noise_8deg_df["pixel_size"] = 9.3

        locally_sparse_noise_df = pd.concat(
            [locally_sparse_noise_4deg_df, locally_sparse_noise_8deg_df]
        )

    # convert frames to timestamps
    locally_sparse_noise_df["start"] = timestamps[
        locally_sparse_noise_df["start"].values
    ]
    locally_sparse_noise_df["end"] = timestamps[locally_sparse_noise_df["end"].values]

    locally_sparse_noise_df = locally_sparse_noise_df.loc[
        :, ["start", "end", "frame", "pixel_size"]
    ]
    locally_sparse_noise = Interval.from_dataframe(locally_sparse_noise_df)

    locally_sparse_noise.timestamps = (
        locally_sparse_noise.start + locally_sparse_noise.end
    ) / 2
    locally_sparse_noise.register_timekey("timestamps")

    locally_sparse_noise.sort()

    return locally_sparse_noise


def extract_running_speed(nwbfile):
    running_speed, timestamps = nwbfile.get_running_speed()
    nan_mask = np.isnan(running_speed)
    running_speed = running_speed[~nan_mask]
    timestamps = timestamps[~nan_mask]

    assert len(running_speed) == len(timestamps)

    running_speed = IrregularTimeSeries(
        timestamps=timestamps,
        running_speed=running_speed.astype(np.float32).reshape(
            -1, 1
        ),  # continues values needs to be 2 dimensional
        domain="auto",
    )

    return running_speed


def extract_pupil_info(nwbfile):
    try:
        timestamps, pupil_location = nwbfile.get_pupil_location()
        _, pupil_size = nwbfile.get_pupil_size()
    except NoEyeTrackingException as e:
        return None

    # Check for NaNs
    nan_mask = np.logical_or(np.isnan(pupil_location).any(axis=1), np.isnan(pupil_size))

    pupil = IrregularTimeSeries(
        timestamps=timestamps[~nan_mask],
        location=pupil_location[~nan_mask].astype(np.float32),
        size=pupil_size[~nan_mask].astype(np.float32),
        domain="auto",
    )

    return pupil


def extract_stimulus_epochs(nwbfile):
    r"""Extracts the stimulus epochs from a given session. An epoch is defined as a
    contiguous period of time during which a stimulus is presented. Most stimuli will
    be presented once or three times in a session.
    Returns a dictionary where the keys are the stimulus names and the values are
    intervals representing the epochs of that stimulus.
    """
    timestamps, _ = nwbfile.get_dff_traces()
    try:
        df = nwbfile.get_stimulus_epoch_table()
    except EpochSeparationException as e:
        print(f"An error occurred while getting the stimulus epoch table: {e}")
        return None

    epoch_dict = (
        df.groupby("stimulus")
        .apply(lambda x: list(zip(timestamps[x.start], timestamps[x.end])))
        .to_dict()
    )

    epoch_dict = {f"{k}_domain": Interval.from_list(v) for k, v in epoch_dict.items()}

    # # allow split mask overlap
    # for epoch in epoch_dict.values():
    #     epoch.allow_split_mask_overlap()

    return epoch_dict


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="allen_visual_coding_ophys_2016",
        origin_version="unknown",
        derived_version="1.0.0",
        source="https://observatory.brain-map.org/visualcoding/",
        description="This dataset includes all experiments from "
        "Allen Institute Brain Observatory.",
    )

    logging.info(f"Processing file: {args.input_file}")

    # get nwbfile
    manifest_path = os.path.join(os.path.dirname(args.input_file), "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    session_id = int(args.input_file[-13:-4])
    nwbfile = boc.get_ophys_experiment_data(
        ophys_experiment_id=session_id, file_name=args.input_file
    )

    # extract subject metadata
    session_meta_data = nwbfile.get_metadata()
    subject = SubjectDescription(
        id=str(session_meta_data["experiment_container_id"]),
        species=Species.MUS_MUSCULUS,
        age=session_meta_data["age_days"],
        sex=Sex.from_string(session_meta_data["sex"]),
        cre_line=Cre_line.from_string(
            session_meta_data["cre_line"].replace("-", "_").split("/")[0]
        ),
    )

    # extract experiment metadata
    recording_date = session_meta_data["session_start_time"]
    session_type = session_meta_data["session_type"]

    # register session
    session_description = SessionDescription(
        id=str(session_id),
        recording_date=recording_date,
    )

    device_description = DeviceDescription(
        id=str(session_id),
        recording_tech=RecordingTech.TWO_PHOTON_IMAGING,
        imaging_depth=session_meta_data["imaging_depth_um"],
        target_area=BrainRegion.from_string(session_meta_data["targeted_structure"]),
    )

    # extract calcium traces
    calcium_traces = extract_calcium_traces(nwbfile)
    units = extract_units(nwbfile)

    # extract stimulus and behavior data
    stimuli_and_behavior_dict = {
        "running": extract_running_speed(nwbfile),
    }

    pupil = extract_pupil_info(nwbfile)
    if pupil is not None:
        stimuli_and_behavior_dict["pupil"] = pupil

    epoch_dict = extract_stimulus_epochs(nwbfile)
    if epoch_dict is None:
        # allensdk bug; skip this session
        #
        logging.warn(f"Skipping session {session_id} due to allensdk bug.")
        with open("./bad_sessions.txt", "a") as f:
            f.write(f"{session_id}\n")
            f.write("\n")
        return

    # three different types of sessions contain different stimuli
    if session_type == "three_session_A":
        stimuli_and_behavior_dict["drifting_gratings"] = extract_drifting_gratings(
            nwbfile
        )
        stimuli_and_behavior_dict["natural_movie_one"] = extract_natural_movie_one(
            nwbfile
        )
        stimuli_and_behavior_dict["natural_movie_three"] = extract_natural_movie_three(
            nwbfile
        )

    elif session_type == "three_session_B":
        stimuli_and_behavior_dict["natural_movie_one"] = extract_natural_movie_one(
            nwbfile
        )
        stimuli_and_behavior_dict["natural_scenes"] = extract_natural_scenes(nwbfile)
        stimuli_and_behavior_dict["static_gratings"] = extract_static_grating(nwbfile)

    elif session_type == "three_session_C" or session_type == "three_session_C2":
        stimuli_and_behavior_dict["natural_movie_one"] = extract_natural_movie_one(
            nwbfile
        )
        stimuli_and_behavior_dict["natural_movie_two"] = extract_natural_movie_two(
            nwbfile
        )
        stimuli_and_behavior_dict["locally_sparse_noise"] = (
            extract_locally_sparse_noise(nwbfile, session_type=session_type)
        )

    else:
        raise ValueError("Unidentified session type.")

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # neural activity
        calcium_traces=calcium_traces,
        units=units,
        # stimuli and behavior
        **stimuli_and_behavior_dict,
        # domain
        domain=calcium_traces.domain,
        **epoch_dict,
    )

    # make grid along which splits will be allowed
    grid = Interval(np.array([]), np.array([]))
    for name, interval in stimuli_and_behavior_dict.items():
        if name != "running" and isinstance(interval, Interval):
            grid = grid | interval

    train_intervals, valid_intervals, test_intervals = generate_train_valid_test_splits(
        epoch_dict, grid
    )

    data.set_train_domain(train_intervals)
    data.set_valid_domain(valid_intervals)
    data.set_test_domain(test_intervals)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
