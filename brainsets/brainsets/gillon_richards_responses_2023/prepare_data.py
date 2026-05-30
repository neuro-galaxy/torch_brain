import argparse
import datetime
import logging

from pynwb import NWBHDF5IO
import numpy as np

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)

from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.dandi_utils import extract_subject_from_nwb
from brainsets_utils.taxonomy import RecordingTech, Task


def extract_calcium_traces(ophys):
    df_over_f = ophys.get_data_interface("DfOverF")
    roi_traces = df_over_f.roi_response_series["RoiResponseSeries"]

    # estimate sampling rate
    timestamps = roi_traces.timestamps
    assert np.std(np.diff(timestamps)) < 1e-5  # less than 10 ns
    sampling_rate = 1 / np.mean(np.diff(timestamps))

    calcium_traces = RegularTimeSeries(
        df_over_f=roi_traces.data[:],  # dim (frames x curr_num_roi)
        sampling_rate=sampling_rate,
        domain=Interval(
            timestamps[0], timestamps[0] + (len(timestamps) - 1) / sampling_rate
        ),
    )

    return calcium_traces


def extract_units(ophys):
    # extract features from roi masks
    ps = ophys.get_data_interface("ImageSegmentation").plane_segmentations[
        "PlaneSegmentation"
    ]

    roi_masks = ps.columns[0].data[:]  # (num_rois, height, width)
    num_rois = roi_masks.shape[0]

    # first position of rois in the image
    roi_positions = np.zeros((num_rois, 2))
    roi_areas = np.zeros(num_rois)
    roi_heights = np.zeros(num_rois)
    roi_widths = np.zeros(num_rois)

    for i in range(num_rois):
        y_coords, x_coords = np.nonzero(roi_masks[i])
        assert len(y_coords) > 0

        roi_positions[i, 0] = np.mean(x_coords)
        roi_positions[i, 1] = np.mean(y_coords)
        roi_areas[i] = np.sum(roi_masks[i])
        roi_heights[i] = np.max(y_coords) - np.min(y_coords) + 1
        roi_widths[i] = np.max(x_coords) - np.min(x_coords) + 1

    # hardcoded min and max scaling
    roi_areas = 2 * (roi_areas - 6.0) / (1108.0 - 6.0) - 1
    roi_widths = 2 * (roi_widths - 3.0) / (101.0 - 3.0) - 1
    roi_heights = 2 * (roi_heights - 1.0) / (119.0 - 1.0) - 1

    units = ArrayDict(
        id=np.arange(num_rois).astype(str),
        imaging_plane_xy=np.array(roi_positions),
        imaging_plane_area=np.array(roi_areas),
        imaging_plane_width=np.array(roi_widths),
        imaging_plane_height=np.array(roi_heights),
    )

    return units


def extract_behavior(behavior):
    # get running velocity
    bts = behavior.get_data_interface("BehavioralTimeSeries")
    running_velocity = bts.time_series["running_velocity"].data[:]
    running_timestamps = bts.time_series["running_velocity"].timestamps[:]

    pupiltracking = behavior.get_data_interface("PupilTracking")
    pupil_diameter = pupiltracking.time_series["pupil_diameter"].data[:]
    pupil_position_x = pupiltracking.time_series["pupil_position_x"].data[:]
    pupil_position_y = pupiltracking.time_series["pupil_position_y"].data[:]
    pupil_timestamps = pupiltracking.time_series["pupil_diameter"].timestamps[:]

    running = IrregularTimeSeries(
        timestamps=running_timestamps,
        velocity=running_velocity,
        domain="auto",
    )

    pupil = IrregularTimeSeries(
        timestamps=pupil_timestamps,
        diameter=pupil_diameter,
        position_x=pupil_position_x,
        position_y=pupil_position_y,
        domain="auto",
    )

    return running, pupil


def extract_gabor_trials(trial_table):
    gabor_start_times = trial_table.loc[
        (trial_table["stimulus_type"] == "gabors")
        & (trial_table["gabor_frame"] == "A"),
        "start_time",
    ]
    gabor_end_times = trial_table.loc[
        (trial_table["stimulus_type"] == "gabors")
        & (trial_table["gabor_frame"] == "G"),
        "stop_time",
    ]
    gabor_orientations = trial_table.loc[
        (trial_table["stimulus_type"] == "gabors")
        & (trial_table["gabor_frame"] == "A"),
        "gabor_mean_orientation",
    ]
    gabor_or_map = {"0": 0, "45": 1, "90": 2, "135": 3}
    gabor_orientations = [gabor_or_map[str(int(angle))] for angle in gabor_orientations]

    gabor_trials = Interval(
        start=np.array(gabor_start_times),
        end=np.array(gabor_end_times),
        timestamps=(np.array(gabor_start_times) + np.array(gabor_end_times)) / 2,
        gabor_orientation=np.array(gabor_orientations),
        timekeys=["start", "end", "timestamps"],
    )
    return gabor_trials


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
        experiment_name="openscope_calcium",
        origin_version="dandi/000037/0.240209.1623",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000037/",
        description="This dataset contains 2-photon calcium imaging dataset"
        "from mouse PVC. Mice were performing passive viewing tasks."
        "It's either gabors or moving squares.",
    )

    # Looping through all prod sessions (50 in total):
    for file_path in find_files_by_extension(db.raw_folder_path, "_behavior+ophys.nwb"):
        logging.info(f"Processing file: {file_path}")

        with db.new_session() as session:

            io = NWBHDF5IO(file_path, mode="r")
            nwbfile = io.read()

            # extract subject metadata
            # this dataset is from dandi, which has structured subject metadata, so we
            # can use the helper function extract_subject_from_nwb
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
            sortset_id = f"{subject.id}_{recording_date}"
            session_id = f"{nwbfile.session_id}"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.DISCRETE_VISUAL_CODING,
            )

            # extract calcium traces
            ophys = nwbfile.processing["ophys"]
            calcium_traces = extract_calcium_traces(ophys)
            units = extract_units(ophys)

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract continuous behavior
            behavior = nwbfile.processing["behavior"]
            running, pupil = extract_behavior(behavior)

            # extract stim info
            trial_table = nwbfile.trials.to_dataframe()
            gabor_trials = extract_gabor_trials(trial_table)

            # close file
            io.close()

            data = Data(
                calcium_traces=calcium_traces,
                units=units,
                running=running,
                pupil=pupil,
                gabor_trials=gabor_trials,
                domain=calcium_traces.domain,
            )

            session.register_data(data)

            # split and register trials into train, validation and test
            train_trials, valid_trials, test_trials = gabor_trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            gabor_trials.allow_split_mask_overlap()

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
