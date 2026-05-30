"""Load data, processes it, save it."""

import argparse
import datetime
import logging
import os
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import sys
import csv
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from tqdm import tqdm

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)

from brainsets_utils.taxonomy import *
from brainsets_utils.taxonomy.mice import *
from brainsets_utils.taxonomy.subject import *
from brainsets_utils import DatasetBuilder, find_files_by_extension


logging.basicConfig(level=logging.INFO)

# copying from neuropixel dataloader:
# https://github.com/nerdslab/project-kirby/blob/venky/allen/brainsets/allen_visual_behavior_neuropixels/prepare_data.py
WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def min_max_scale(data, data_min, data_max):
    return 2 * ((data - data_min) / (data_max - data_min)) - 1


def get_roi_position(ROI_masks, num_rois):
    """
    input: ROI object from allensdk

    output: sinusoidal encoding of all ROIs position
    """
    centroids = np.zeros((num_rois, 2))

    for count, curr in enumerate(ROI_masks):
        mask = ROI_masks[count].get_mask_plane()
        y_coords, x_coords = np.nonzero(mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)

        centroids[count] = [centroid_y, centroid_x]

    return centroids


def get_roi_feats(ROI_masks, num_rois):
    """
    input: ROI object from allensdk

    output: minmax scaled area, height and width for each ROI
    """
    areas = np.zeros(num_rois)
    heights = np.zeros(num_rois)
    widths = np.zeros(num_rois)

    for count, curr in enumerate(ROI_masks):
        mask = ROI_masks[count].get_mask_plane()
        areas[count] = np.count_nonzero(mask)

        rows, cols = np.where(mask)
        heights[count] = np.max(rows) - np.min(rows) + 1
        widths[count] = np.max(cols) - np.min(cols) + 1

    normalized_areas = min_max_scale(areas, 101.0, 551.0)
    normalized_heights = min_max_scale(heights, 8.0, 44.0)
    normalized_widths = min_max_scale(widths, 9.0, 42.0)

    return normalized_areas, normalized_heights, normalized_widths


def extract_dg_stim_trials(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    master_stim_table = nwbfile.get_stimulus_table("drifting_gratings")

    stim_df = pd.DataFrame(master_stim_table)

    start_times = timestamps[stim_df.loc[(stim_df["blank_sweep"] == 0.0), "start"]]
    end_times = timestamps[stim_df.loc[(stim_df["blank_sweep"] == 0.0), "end"]]
    temp_freqs = stim_df.loc[
        (stim_df["blank_sweep"] == 0.0), "temporal_frequency"
    ].values
    orientations = stim_df.loc[(stim_df["blank_sweep"] == 0.0), "orientation"].values
    temp_freq_map = {1.0: 0, 2.0: 1, 4.0: 2, 8.0: 3, 15.0: 4}

    trials = Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        orientation=np.round(orientations / 45.0).astype(np.int64),
        temp_freq=np.array([temp_freq_map[freq] for freq in temp_freqs]).astype(
            np.int64
        ),
        timekeys=["start", "end", "timestamps"],
    )

    return trials


def extract_nm1_stim_trials(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    master_stim_table = nwbfile.get_stimulus_table("natural_movie_one")
    stim_df = pd.DataFrame(master_stim_table)

    start_times = timestamps[stim_df["start"].values]
    end_times = timestamps[stim_df["end"].values]

    frame_number = stim_df["frame"].values

    trials = Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        frame_number=np.array(frame_number).astype(np.float32).reshape(-1, 1),
        timekeys=["start", "end", "timestamps"],
    )

    return trials


def get_nm1_split(nwbfile):
    timestamps, _ = nwbfile.get_dff_traces()
    master_stim_table = nwbfile.get_stimulus_table("natural_movie_one")
    stim_df = pd.DataFrame(master_stim_table)

    start_times = []
    end_times = []
    for i in range(10):
        curr_start = stim_df.loc[
            (stim_df["repeat"] == i) & (stim_df["frame"] == 0.0), "start"
        ].values[0]
        start_times.append(timestamps[curr_start])
        curr_end = stim_df.loc[
            (stim_df["repeat"] == i) & (stim_df["frame"] == 899.0), "end"
        ].values[0]
        end_times.append(timestamps[curr_end])

    natural_movie_one_split = Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        mv_start=np.array(start_times),
        repeat_num=np.arange(10).astype(np.int64),
        timekeys=["start", "end", "mv_start"],  # Not sure what this should be
    )
    return natural_movie_one_split


def get_maps():
    cre_line_map = {
        "Cux2-CreERT2/Cux2-CreERT2": Cre_line.CUX2_CREERT2,
        "Cux2-CreERT2/wt": Cre_line.CUX2_CREERT2,
        "Emx1-IRES-Cre/wt": Cre_line.EXM1_IRES_CRE,
        "Fezf2-CreER/wt": Cre_line.FEZF2_CREER,
        "Nr5a1-Cre/wt": Cre_line.NR5A1_CRE,
        "Ntsr1-Cre_GN220/wt": Cre_line.NTSR1_CRE_GN220,
        "Pvalb-IRES-Cre/wt": Cre_line.PVALB_IRES_CRE,
        "Rbp4-Cre_KL100/wt": Cre_line.RBP4_CRE_KL100,
        "Rorb-IRES2-Cre/wt": Cre_line.RORB_IRES2_CRE,
        "Scnn1a-Tg3-Cre/wt": Cre_line.SCNN1A_TG3_CRE,
        "Slc17a7-IRES2-Cre/wt": Cre_line.SLC17A7_IRES2_CRE,
        "Sst-IRES-Cre/wt": Cre_line.SST_IRES_CRE,
        "Tlx3-Cre_PL56/wt": Cre_line.TLX3_CRE_PL56,
        "Vip-IRES-Cre/wt": Cre_line.VIP_IRES_CRE,
    }
    sex_map = {"male": Sex.MALE, "female": Sex.FEMALE}

    vis_area_map = {
        "VISrl": Vis_areas.VIS_RL,
        "VISpm": Vis_areas.VIS_PM,
        "VISal": Vis_areas.VIS_AL,
        "VISam": Vis_areas.VIS_AM,
        "VISp": Vis_areas.VIS_P,
        "VISl": Vis_areas.VIS_L,
    }
    return cre_line_map, sex_map, vis_area_map


def get_depth_mapping(nwbfile):
    depth = nwbfile.get_metadata()["imaging_depth_um"]
    if depth <= 250:
        return Depth_classes.DEPTH_CLASS_1
    elif depth <= 350:
        return Depth_classes.DEPTH_CLASS_2
    elif depth <= 500:
        return Depth_classes.DEPTH_CLASS_3
    elif depth <= 600:
        return Depth_classes.DEPTH_CLASS_4
    else:
        return Depth_classes.DEPTH_CLASS_5


def extract_calcium_traces(nwbfile):
    timestamps, traces = nwbfile.get_dff_traces()
    traces = np.transpose(traces)
    # timestamps = np.array(timestamps).astype(np.float32)
    # traces = np.array(traces.astype(np.float32))

    # estimate sampling rate
    # assert np.std(np.diff(timestamps)) < 1e-5  # less than 10 ns
    sampling_rate = 1 / np.mean(np.diff(timestamps))

    calcium_traces = RegularTimeSeries(
        df_over_f=np.array(traces),
        sampling_rate=sampling_rate,
        domain=Interval(
            np.array([float(timestamps[0])]),
            np.array([float(timestamps[0] + (len(timestamps) - 1) / sampling_rate)]),
        ),
    )

    return calcium_traces


def extract_units(nwbfile):
    roi_ids = nwbfile.get_roi_ids()

    ROI_masks = nwbfile.get_roi_mask()
    unit_positions = get_roi_position(ROI_masks, roi_ids.shape[0])
    unit_area, unit_width, unit_height = get_roi_feats(ROI_masks, roi_ids.shape[0])

    units = ArrayDict(
        id=roi_ids.astype(str),
        imaging_plane_xy=np.array(unit_positions),
        imaging_plane_area=np.array(unit_area),
        imaging_plane_width=np.array(unit_width),
        imaging_plane_height=np.array(unit_height),
    )

    return units


def main():
    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_path)

    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for dataset
        experiment_name="allen_brain_observatory_calcium",
        origin_version="v2",
        derived_version="0.0.1",
        source="https://observatory.brain-map.org/visualcoding/",
        description="This dataset includes all experiments from "
        "Allen Institute Brain Observatory with stimulus drifting gratings.",
    )

    nwb_file_path = os.path.join(db.raw_folder_path, "ophys_experiment_data")

    # for count, curr_sess_id in enumerate(sess_ids):
    for count, file_path in enumerate(find_files_by_extension(nwb_file_path, ".nwb")):
        logging.info(f"Processing file {count}: {file_path}")
        # FOR TESTING!!!!
        # if count >= 10:
        #    break
        with db.new_session() as session:

            session_id = int(file_path[-13:-4])
            nwbfile = boc.get_ophys_experiment_data(
                ophys_experiment_id=session_id, file_name=file_path
            )

            session_meta_data = nwbfile.get_metadata()
            cre_line_map, sex_map, vis_area_map = get_maps()

            subject = SubjectDescription(
                id=str(session_id),
                species=Species.MUS_MUSCULUS,
                sex=sex_map[session_meta_data["sex"]],
                cre_line=cre_line_map[session_meta_data["cre_line"]],
                depth=str(session_meta_data["imaging_depth_um"]),
                depth_class=get_depth_mapping(nwbfile),
                target_area=vis_area_map[session_meta_data["targeted_structure"]],
            )

            session.register_subject(subject)

            recording_date = session_meta_data["session_start_time"]
            # sortset_id = f"{session_id}_{vis_area_map[session_meta_data['targeted_structure']]}"
            sortset_id = str(session_id)

            # register session
            session.register_session(
                id=str(session_id),
                recording_date=recording_date,
                task=Task.DISCRETE_VISUAL_CODING,
            )

            calcium_traces = extract_calcium_traces(nwbfile)
            units = extract_units(nwbfile)

            drifting_gratings = extract_dg_stim_trials(nwbfile)
            natural_movie_one = extract_nm1_stim_trials(nwbfile)

            session.register_sortset(id=sortset_id, units=units)

            data = Data(
                calcium_traces=calcium_traces,
                units=units,
                drifting_gratings=drifting_gratings,
                natural_movie_one=natural_movie_one,
                domain=calcium_traces.domain,
            )

            session.register_data(data)

            # split and register trials into train, validation and test
            train_trials, valid_trials, test_trials = drifting_gratings.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            natural_movie_one_split = get_nm1_split(nwbfile)
            # splitting natural movie one stimulus the same way as cebra PLEASE CHECK!
            nm1_train_trials, nm1_valid_trials, nm1_test_trials = (
                natural_movie_one_split.split([0.8, 0.1, 0.1], shuffle=False)
            )

            data.natural_movie_one_epochs = natural_movie_one_split

            session.register_split("train", train_trials | nm1_train_trials)
            session.register_split("valid", valid_trials | nm1_valid_trials)
            session.register_split("test", test_trials | nm1_test_trials)

            # stim_trials.allow_split_mask_overlap()

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
