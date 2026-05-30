import argparse
import logging
import os
import numpy as np
import scipy

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets_utils import DatasetBuilder, make_directory
from brainsets_utils.taxonomy import (
    Sex,
    SubjectDescription,
    Species,
    Task,
    Orientation_8_Classes,
)

logging.basicConfig(level=logging.INFO)


def extract_units(AL_spikes, V1_spikes):
    """
    Extracts units from AL and V1 spike data.

    Args:
        AL_spikes (dict): Dictionary containing AL spike data.
        V1_spikes (dict): Dictionary containing V1 spike data.

    Returns:
        ArrayDict: Dictionary containing unit information.
    """
    AL_ROI = np.transpose(np.array(AL_spikes["ROI"]), (2, 0, 1))
    V1_ROI = np.transpose(np.array(V1_spikes["ROI"]), (2, 0, 1))

    ROIs = np.concatenate([V1_ROI, AL_ROI], axis=0)

    return ArrayDict(
        unit_name=np.array(["unit_" + str(i) for i in range(515)]),
        id=np.array([i for i in range(515)]),
        unit_area_name=np.array(["V1"] * 352 + ["AL"] * 163),
        regions_of_interest=ROIs,
    )


def extract_calcium(AL_trace, V1_trace, calcium_fr):
    """
    Extracts calcium traces from AL and V1 trace data.

    Args:
        AL_trace (dict): Dictionary containing AL trace data.
        V1_trace (dict): Dictionary containing V1 trace data.
        calcium_fr (float): Calcium frame rate.

    Returns:
        RegularTimeSeries: Regular time series containing combined calcium traces.
    """
    AL_trace_new = np.array(AL_trace["Combo3_AL_trace"])
    V1_trace_new = np.array(V1_trace["Combo3_V1_trace"])

    # Extract only valid frames, where the timestamps match the spike data
    V1_trace_new = V1_trace_new[33:31996, :]
    AL_trace_new = AL_trace_new[33:31996, :]

    combined_trace = np.concatenate([V1_trace_new, AL_trace_new], axis=1)

    return RegularTimeSeries(
        sampling_rate=calcium_fr,
        values=combined_trace,
        domain=Interval(start=0, end=2400),
    )


def extract_intervals():
    """
    Extracts intervals representing different stimuli.

    Returns:
        Interval: Interval object containing start times, end times, stimulus IDs,
        and stimulus names.
    """
    time = 0
    start_times = []
    end_times = []
    stimulus_id = []
    stimulus_name = []
    stimulus_dict = {
        0: "gray screen",
        1: "drifting gratings",
        2: "natural movie 1",
        3: "natural movie 2",
    }

    # Each stimulus is presented for 32 seconds, preceded by an 8 second gray screen
    # The next stimulus follows a gray screen of 8 seconds (3 types of stimuli total)
    # There are 20 repetitions of this block of stimuli,
    drifting_gratings_orientation = []
    drifting_gratings_start_times = []
    drifting_gratings_end_times = []
    for i in range(60):
        start_times.append(time)
        stimulus_id.append(0)
        time += 8
        end_times.append(time)
        start_times.append(time)
        stimulus_id.append((i % 3) + 1)
        if (i % 3) + 1 == 1:
            drifting_gratings_start_times += [time + i * 4 for i in range(8)]
            drifting_gratings_end_times += [time + (i + 1) * 4 for i in range(8)]
            drifting_gratings_orientation += [
                int(Orientation_8_Classes.angle_0),
                int(Orientation_8_Classes.angle_315),
                int(Orientation_8_Classes.angle_270),
                int(Orientation_8_Classes.angle_225),
                int(Orientation_8_Classes.angle_180),
                int(Orientation_8_Classes.angle_135),
                int(Orientation_8_Classes.angle_90),
                int(Orientation_8_Classes.angle_45),
            ]

        time += 32
        end_times.append(time)

    stimulus_name = [stimulus_dict[index] for index in stimulus_id]

    movie_frame_times = np.array([])
    movie1_frames = np.array([2881 + i for i in range(60 * 32)])
    movie2_frames = np.array([5281 + i for i in range(60 * 32)])
    movie_frames = np.array([])
    for i in range(20):
        movie1_times = np.linspace(48 + i * 120, 80 + i * 120, 32 * 60, endpoint=False)
        movie2_times = np.linspace(88 + i * 120, 120 + i * 120, 32 * 60, endpoint=False)
        movie_frames = np.concatenate([movie_frames, movie1_frames, movie2_frames])
        movie_frame_times = np.concatenate(
            [movie_frame_times, movie1_times, movie2_times]
        )
    movie_frame_times = movie_frame_times.astype(np.float64)
    stimulus_intervals = Interval(
        start=np.array(start_times, dtype=np.float64),
        end=np.array(end_times, dtype=np.float64),
        stimulus_id=np.array(stimulus_id),
        stimulus_name=np.array(stimulus_name),
    )

    drifting_gratings_intervals = Interval(
        start=np.array(drifting_gratings_start_times, dtype=np.float64),
        end=np.array(drifting_gratings_end_times, dtype=np.float64),
        timestamps=np.array(drifting_gratings_start_times, dtype=np.float64) + 2.0,
        orientation=np.array(drifting_gratings_orientation),
    )

    movie_frames_series = IrregularTimeSeries(
        timestamps=movie_frame_times,
        frame_index=movie_frames,
        domain=Interval(start=0, end=2400),
    )

    return stimulus_intervals, drifting_gratings_intervals, movie_frames_series


def extract_spikes(AL_spikes, V1_spikes):
    """
    Extracts spike data from AL and V1 spike data. Approximates the timestamps of the
    spikes relative to the complete experiment, which lasts 2400 seconds

    The original experiment provides timestamps relative to the start of each stimulus,
    whose unit is the calcium frame number for that stimulus. We approximate the spike
    timestamp by adding the (calcium frame number) * (sampling rate)
    to the start time of the stimulus.

    Args:
        AL_spikes (dict): Dictionary containing AL spike data.
        V1_spikes (dict): Dictionary containing V1 spike data.

    Returns:
        IrregularTimeSeries: Irregular time series containing spike timestamps and
        unit indices.
    """
    V1_calcium_frame_rate = V1_spikes["imgPara"][0, 0][5][0][0]
    AL_calcium_frame_rate = AL_spikes["imgPara"][0, 0][5][0][0]

    V1_spike_times = []
    V1_spike_neurons = []

    for neuron, (stim_spikes, gray_spikes) in enumerate(V1_spikes["spiketrain"][0]):

        for stim_index in range(3):
            for repetition_index in range(20):

                spike_frames = stim_spikes[repetition_index, stim_index][0]
                initial_time = 8 + (40 * stim_index) + (120 * repetition_index)
                approx_spike_times = [
                    initial_time + (x * V1_calcium_frame_rate) for x in spike_frames
                ]
                V1_spike_times += approx_spike_times
                V1_spike_neurons += [neuron] * len(spike_frames)

                gray_spike_frames = gray_spikes[repetition_index, stim_index][0]
                gray_initial_time = (40 * stim_index) + (120 * repetition_index)
                gray_approx_spike_times = [
                    gray_initial_time + (x * V1_calcium_frame_rate)
                    for x in gray_spike_frames
                ]
                V1_spike_times += gray_approx_spike_times
                V1_spike_neurons += [neuron] * len(gray_spike_frames)

    AL_spike_times = []
    AL_spike_neurons = []

    for neuron, (stim_spikes, gray_spikes) in enumerate(AL_spikes["spiketrain"][0]):

        for stim_index in range(3):
            for repetition_index in range(20):

                spike_frames = stim_spikes[repetition_index, stim_index][0]
                initial_time = 8 + (40 * stim_index) + (120 * repetition_index)
                approx_spike_times = [
                    initial_time + (x * AL_calcium_frame_rate) for x in spike_frames
                ]
                AL_spike_times += approx_spike_times
                AL_spike_neurons += [352 + neuron] * len(spike_frames)

                gray_spike_frames = gray_spikes[repetition_index, stim_index][0]
                gray_initial_time = (40 * stim_index) + (120 * repetition_index)
                gray_approx_spike_times = [
                    gray_initial_time + (x * AL_calcium_frame_rate)
                    for x in gray_spike_frames
                ]
                AL_spike_times += gray_approx_spike_times
                AL_spike_neurons += [351 + neuron] * len(gray_spike_frames)

    V1_spike_times = np.array(V1_spike_times)
    V1_spike_neurons = np.array(V1_spike_neurons)

    AL_spike_times = np.array(AL_spike_times)
    AL_spike_neurons = np.array(AL_spike_neurons)

    V1_labels = np.array(["V1"] * len(V1_spike_times))
    AL_labels = np.array(["AL"] * len(AL_spike_times))

    all_spikes = np.concatenate([V1_spike_times, AL_spike_times])
    all_labels = np.concatenate([V1_labels, AL_labels])
    all_spike_neurons = np.concatenate([V1_spike_neurons, AL_spike_neurons])

    sorted_indices = np.argsort(all_spikes)

    all_spikes = all_spikes[sorted_indices]
    all_labels = all_labels[sorted_indices]
    all_spike_neurons = all_spike_neurons[sorted_indices]

    return IrregularTimeSeries(
        timestamps=all_spikes,
        unit_index=all_spike_neurons,
        domain=Interval(start=0, end=2400),
    )


def main():
    """
    Main function to extract data and register it in the dataset.

    Returns:
        None
    """
    experiment_name = "yu_smith_selective_2022"

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    db = DatasetBuilder(
        raw_folder_path=raw_folder_path,
        processed_folder_path=processed_folder_path,
        experiment_name=experiment_name,
        description="Calcium imaging of V1 and AL region in mice: \
        Selective representations of texture and motion in mouse higher visual areas",
        origin_version="1.0.0",
        derived_version="2.0.0",
        source="https://www.sciencedirect.com/science/article/pii/S0960982222007308",
    )

    logging.info("Extracting data from raw folder")
    with db.new_session() as session:

        # raw_folder_path = "/kirby/raw/smith_lab_calcium/Combo3_V1AL/"
        AL_spikes_path = os.path.join(raw_folder_path, "Combo3_AL.mat")
        V1_spikes_path = os.path.join(raw_folder_path, "Combo3_V1.mat")
        AL_trace_path = os.path.join(raw_folder_path, "Combo3_AL_trace.mat")
        V1_trace_path = os.path.join(raw_folder_path, "Combo3_V1_trace.mat")

        AL_spikes = scipy.io.loadmat(AL_spikes_path)
        V1_spikes = scipy.io.loadmat(V1_spikes_path)
        AL_trace = scipy.io.loadmat(AL_trace_path)
        V1_trace = scipy.io.loadmat(V1_trace_path)

        calcium_fr = V1_spikes["imgPara"][0, 0][5][0][0]

        subject = SubjectDescription(
            id="mouse_170",
            species=Species.MUS_MUSCULUS,
            sex=Sex.UNKNOWN,
        )

        units = extract_units(AL_spikes, V1_spikes)
        spikes = extract_spikes(AL_spikes, V1_spikes)

        stimulus_intervals, drifting_gratings_intervals, movie_frames_series = (
            extract_intervals()
        )

        calcium_regular_series = extract_calcium(AL_trace, V1_trace, calcium_fr)

        session.register_subject(subject)

        session.register_session(
            id="mouse_170_V1AL", recording_date="2022", task=Task.DISCRETE_VISUAL_CODING
        )

        session.register_sortset(
            id="mouse_170_V1AL_day_1",
            units=units,
        )

        session_data = Data(
            units=units,
            spikes=spikes,
            trials=stimulus_intervals,
            drifting_gratings=drifting_gratings_intervals,
            movie_frames_series=movie_frames_series,
            calcium_regular_series=calcium_regular_series,
            domain=Interval(start=0, end=2400),
        )

        logging.info("Registering session")

        session.register_data(session_data)

        logging.info("Creating splits")

        # split and register trials into train, validation and test
        train_trials, valid_trials, test_trials = stimulus_intervals.split(
            [0.7, 0.1, 0.2], shuffle=True, random_seed=42
        )

        logging.info("Registering splits")

        session.register_split("train", train_trials)
        session.register_split("valid", valid_trials)
        session.register_split("test", test_trials)

        logging.info("Saving to disk")
        session.save_to_disk()

    db.finish()

    logging.info("Data extraction complete")


if __name__ == "__main__":
    main()
