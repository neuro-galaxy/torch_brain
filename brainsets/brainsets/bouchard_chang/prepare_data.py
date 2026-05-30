import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from sklearn.model_selection import train_test_split
import copy
import collections
import datetime

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
from brainsets_utils import DatasetBuilder, find_files_by_extension, make_directory

from brainsets_utils.taxonomy.homosapiens import HomoSapiens
from brainsets_utils.taxonomy import speech

# logging.basicConfig(level=logging.DEBUG)
# logging.init_logger(FLAGS.logging)
# log = logging.getLogger()

SAMPLE_FREQUENCY = 3051.7578  # Hz


def extract_electrode_descriptions(nwb_file: NWBFile):
    # TODO do we need new data class other than Channel and Probe?
    # assert (list(nwb_file.devices.keys())[0] == 'L256Grid' or list(nwb_file.devices.keys())[0] == '256Grid') and len(list(nwb_file.devices.keys())) == 1, list(nwb_file.devices.keys())
    assert len(list(nwb_file.devices.keys())) == 1
    # print(list(nwb_file.devices.keys())[0])
    logging.debug(list(nwb_file.devices.keys())[0])
    electrodes_dynaTable = nwb_file.acquisition["ElectricalSeries"].electrodes.table
    probe_descriptions = []

    # need to check electrodes data from all the sessions filtering/imp/bad
    channels = []
    for i in range(256):
        if electrodes_dynaTable.bad.data[i]:
            continue
        channels.append(
            Channel(
                id=f"{nwb_file.session_id}_{i+1:03}",
                local_index=i,
                relative_x_um=electrodes_dynaTable.x.data[
                    i
                ],  # x, y, z coordinates are not always present in the dataset, e.g. missing for EC9 & GP33
                relative_y_um=electrodes_dynaTable.y.data[i],
                relative_z_um=electrodes_dynaTable.z.data[i],
                area=electrodes_dynaTable.location.data[i],
            )
        )

    probe_description = Probe(
        id=f"bouchard_chang_{nwb_file.subject.subject_id}_vSMC",
        type=RecordingTech.ECOG_ARRAY_ECOGS,
        wideband_sampling_rate=0,
        waveform_sampling_rate=0,
        lfp_sampling_rate=0,
        waveform_samples=0,
        channels=channels,
        ecog_sampling_rate=SAMPLE_FREQUENCY,
    )
    probe_descriptions.append(probe_description)

    # multi-electrode array
    # human vSMC: ventral sensorimotor cortex
    return probe_descriptions


def extract_ecog(ecog_signals: ElectricalSeries, channel_prefix="channel_"):
    data_points = ecog_signals.data.shape[0]
    num_channels = ecog_signals.data.shape[1]
    step = 1.0 / SAMPLE_FREQUENCY
    end = data_points / SAMPLE_FREQUENCY
    timestamps = torch.arange(ecog_signals.starting_time, end, step, dtype=torch.float)[
        :data_points
    ]

    # TODO should be regular?
    ecog = RegularTimeSeries(
        timestamps=timestamps,
        waveforms=torch.tensor(ecog_signals.data[:]),
    )

    extra = {
        "bad_channels": torch.tensor(
            ecog_signals.electrodes.table.bad.data[:], dtype=torch.bool
        ),
        "sampling_frequency": SAMPLE_FREQUENCY,
        "num_channels": num_channels,
        "timestamp_unit": str(ecog_signals.unit),
        "conversion": float(ecog_signals.conversion),
    }
    assert ecog.sorted

    channels = Data(
        channel_name=[f"{channel_prefix}{c:03}" for c in range(num_channels)],
        unit_name=[f"{channel_prefix}{c}" for c in range(num_channels)],
        channel_index=torch.arange(num_channels),
        type=torch.ones(num_channels) * int(RecordingTech.ECOG_ARRAY_ECOGS),
    )

    # name_to_index = {k: v for v, k in enumerate(channels.unit_name)}
    ecog.unit_index = torch.arange(num_channels).to(dtype=torch.long)

    return ecog, extra, channels


def check_sorted(arr: np.array):
    assert len(arr.shape) == 1
    return all(np.diff(arr) >= 0.0)


def extract_trials(
    trials: TimeIntervals, epochs: TimeIntervals, invalid_times: TimeIntervals
):
    # spikes = IrregularTimeSeries(
    #     timestamps=torch.tensor(spikes),
    #     waveforms=torch.tensor(waveforms),
    #     unit_index=torch.tensor(unit_index),
    #     types=torch.tensor(types),
    # )
    # print(trials.condition.data[:])
    syllable_indice = np.zeros_like(trials.condition.data[:], dtype=int)
    # print(syllable_indice.dtype)
    for i in range(trials.condition.data[:].shape[0]):
        if trials.condition.data[i] == "":
            syllable_indice[i] = int(speech.CVSyllable["empty"])
        else:
            syllable_indice[i] = int(speech.CVSyllable[trials.condition.data[i]])
    assert all(syllable_indice > -1)  # there shouldn't be an outlier
    if not (invalid_times == None):
        invalid_start_time = invalid_times.start_time.data[:]
        invalid_stop_time = invalid_times.stop_time.data[:]
    invalid_trial = torch.tensor(
        [False] * trials.start_time.data.shape[0], dtype=torch.bool
    )
    invalid_epoch = torch.tensor(
        [False] * epochs.start_time.data.shape[0], dtype=torch.bool
    )

    trial_start_time = trials.start_time.data[:]
    trial_stop_time = trials.stop_time.data[:]
    assert trial_start_time.shape[0] == trial_stop_time.shape[0]
    assert len(trial_start_time.shape) == 1 and len(trial_stop_time.shape) == 1
    check_sorted(trial_start_time)
    check_sorted(trial_stop_time)

    # check non-overlapping
    if all((trial_start_time[1:] - trial_stop_time[:-1]) >= 0):
        logging.info("there is no overlapping trial")
    else:
        logging.info(
            f"there is/are {np.sum((trial_start_time[1:] - trial_stop_time[:-1]) < 0)} trial overlapped"
        )

    for i in range(trial_start_time.shape[0]):
        if not (invalid_times == None):
            # flag the time intervals that overlap the invalid time
            if any(
                np.logical_and(
                    invalid_start_time <= trial_stop_time[i],
                    invalid_stop_time >= trial_start_time[i],
                )
            ):
                invalid_trial[i] = True
            # if syllable_indice[i] <= 0:
            #     invalid_trial[i] = True # for now skip null syllable
            if syllable_indice[i] < 0:
                invalid_trial[i] = True  # include null syllable

    epoch_start_time = epochs.start_time.data[:]
    epoch_stop_time = epochs.stop_time.data[:]
    assert epoch_start_time.shape[0] == epoch_stop_time.shape[0]
    assert len(epoch_start_time.shape) == 1 and len(epoch_stop_time.shape) == 1
    check_sorted(epoch_start_time)
    check_sorted(epoch_stop_time)
    for i in range(epoch_start_time.shape[0]):
        if not (invalid_times == None):
            if any(
                np.logical_and(
                    invalid_start_time <= epoch_stop_time[i],
                    invalid_stop_time >= epoch_start_time[i],
                )
            ):
                invalid_epoch[i] = True
    # print(syllable_indice)
    # print(type(syllable_indice), syllable_indice.dtype)

    # slicing will by default shift the start & end by start, so adding the timestamps as middle point afterwards
    cv_trials = Interval(
        start=torch.tensor(trial_start_time),
        end=torch.tensor(trial_stop_time),
        consonant_vowel_syllables=torch.tensor(syllable_indice),
        cv_transition_time=torch.tensor(trials.cv_transition_time.data[:]),
        speak=torch.tensor(trials.speak.data[:], dtype=torch.bool),
        invalid=invalid_trial,
    )
    rest_period = Interval(
        start=torch.tensor(epoch_start_time),
        end=torch.tensor(epoch_stop_time),
        invalid=invalid_epoch,
    )
    session_start = min(cv_trials.start[0], rest_period.start[0])
    session_end = max(cv_trials.end[-1], rest_period.end[-1])
    return cv_trials, rest_period, session_start, session_end


# adopt the interval.split function then still retrive each split segment ranges
def split_and_get_train_valid_test(
    trials: Interval,
    train_split=0.7,
    valid_split=0.1,
    test_split=0.2,
    random_seed=42,
    class_balance=False,
):
    r"""
    Lets conform with other datasets for now: 20% for test and 10% for valid.

    Need to properly skip the invalid time segments"""
    # # TODO: class balance
    # assert 0 < valid_split < 1, "valid_split must be positive, got {}".format(valid_split)
    # assert 0 < test_split < 1, "test_split must be positive, got {}".format(test_split)
    # assert 0 < (valid_split + test_split) < 1, "train_split must be positvie, got {}".format(1 - valid_split - test_split)
    # uninvalid_trial_id = [i for i in np.arange(trials.start.size(0)) if trials.invalid[i] == False]
    # train_valid_trial_ids, test_trial_ids = train_test_split(uninvalid_trial_id, test_size=test_split, random_state=random_state)
    # train_trial_ids, valid_trial_ids = train_test_split(train_valid_trial_ids, test_size=valid_split / (1 - test_split), random_state=random_state)

    # train_segments = [(trials.start[i], trials.end[i]) for i in train_trial_ids]
    # valid_segments = [(trials.start[i], trials.end[i]) for i in valid_trial_ids]
    # test_segments = [(trials.start[i], trials.end[i]) for i in test_trial_ids]

    if class_balance:
        raise NotImplementedError()
    else:
        # each result is splitted trials (one interval objects)
        # print(trials)
        # print(trials.invalid)
        train_trials, valid_trials, test_trials = trials[~trials.invalid].split(
            [train_split, valid_split, test_split],
            shuffle=True,
            random_seed=random_seed,
        )

    # get time segment ranges for slicing
    train_segments = [
        (train_trials.start[i], train_trials.end[i])
        for i in range(train_trials.start.size(0))
    ]  # TODO
    valid_segments = [
        (valid_trials.start[i], valid_trials.end[i])
        for i in range(valid_trials.start.size(0))
    ]
    test_segments = [
        (test_trials.start[i], test_trials.end[i])
        for i in range(test_trials.start.size(0))
    ]

    return train_segments, valid_segments, test_segments


def collect_slices_wChecks(data, segments):
    # include the functionality to handle overlapping time in trials, by picking the most time matched one
    slices = []
    slice_labels = []
    cnt_nospeak = 0
    for start, end in segments:
        data_slice = data.slice(start, end)
        # print(data_slice)
        assert (
            abs(
                (end - start)
                - (data_slice.ecog.timestamps[-1] - data_slice.ecog.timestamps[0])
            )
            <= 2 / SAMPLE_FREQUENCY
        ), "sliced time range too far from expected"
        # print(start, end, data_slice.speech.end - data_slice.speech.start)
        if (
            len(data_slice.speech) > 1
        ):  # rarely there is overlapping in the trial, when there is, take the longest one
            multiple_speech_trial = copy.deepcopy(data_slice.speech)
            idx_to_take = len(multiple_speech_trial)
            for i in range(len(multiple_speech_trial)):
                # recall that slice will shift timestamp start to 0.
                if (multiple_speech_trial[i].end - multiple_speech_trial[i].start) == (
                    end - start
                ):  # find the one trial with the exact interval range
                    idx_to_take = i
            assert not (
                idx_to_take == len(multiple_speech_trial)
            ), "there is something wrong during slicing"
            data_slice.speech = multiple_speech_trial[idx_to_take : (idx_to_take + 1)]
            print(start, end, data_slice.speech.end - data_slice.speech.start)

        # use the middle point as the timestamps
        data_slice.speech.timestamps = (
            data_slice.speech.end - data_slice.speech.start
        ) / 2
        assert torch.all(
            data_slice.speech.timestamps == (data_slice.end - data_slice.start) / 2
        )

        slices.append(data_slice)  # note that slice shift start to 0
        assert (
            len(slices[-1].speech) == 1
        )  # each segment now should contain only one consonant-voswel pair
        assert not torch.any(slices[-1].speech.invalid)
        cnt_nospeak += torch.sum(slices[-1].speech.speak == False)
        slice_labels.append(slices[-1].speech.consonant_vowel_syllables.item())
    logging.info(f"{cnt_nospeak} not speak")
    return slices, slice_labels


if __name__ == "__main__":
    experiment_name = "bouchard_chang_2020"

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    make_directory(processed_folder_path, prompt_if_exists=False)
    make_directory(os.path.join(processed_folder_path, "train"))
    make_directory(os.path.join(processed_folder_path, "valid"))
    make_directory(os.path.join(processed_folder_path, "test"))

    # # Here, we will have multiple trials in each session
    helper = DescriptionHelper()
    subjects = [
        SubjectDescription(id=subject_name, species=Species.HOMO_SAPIENS)
        for subject_name in ["EC2", "EC9", "GP31", "GP33"]
    ]
    # subjects = [SubjectDescription(id=subject_name, species=Species.HOMO_SAPIENS)
    #             for subject_name in ['GP33']]
    session_list = []
    sortsets = collections.defaultdict(list)  # do subject id sorted ?
    trials: list[TrialDescription] = []
    extension = ".nwb"
    cnt_bad_channels = 0

    for file_path in tqdm(sorted(find_files_by_extension(raw_folder_path, extension))):
        if os.path.basename(file_path) == "sub-EC2_ses-EC2-B1.nwb":
            continue
        # if 'sub-EC2' in os.path.basename(file_path):
        #     continue
        # if 'sub-EC9' in os.path.basename(file_path):
        #     continue
        # if 'sub-GP31' in os.path.basename(file_path):
        #     continue
        # if os.path.basename(file_path) == 'sub-EC9_ses-EC9-B46.nwb':
        #     continue
        logging.info(f"Processing file: {file_path}")

        with NWBHDF5IO(file_path, "r", load_namespaces=True) as io:
            nwb_file = io.read()

            session_id = nwb_file.session_id
            subject_id, subject_ses = session_id.split("_")
            assert subject_id == nwb_file.subject.subject_id

            ecog = nwb_file.acquisition["ElectricalSeries"]
            epochs = nwb_file.intervals["epochs"]
            trials = nwb_file.intervals["trials"]
            try:
                invalid_times = nwb_file.intervals["invalid_times"]
            except:
                invalid_times = None

            # extract channel information
            # print('extract probe')
            logging.debug("extract probe")
            probe = extract_electrode_descriptions(nwb_file)

            # TODO various extracting func
            # print('extract ecog signals')
            logging.debug("extract ecog signals")
            ecog_multichan, extra, channels = extract_ecog(
                ecog, channel_prefix=f"channel_{subject_id}_"
            )
            file_bad_channels = torch.sum(extra["bad_channels"] == True).item()
            if file_bad_channels > 0:
                logging.info(
                    f"{file_bad_channels} bad channels in session file {file_path}"
                )
                cnt_bad_channels += file_bad_channels

            # print('extract trials info')
            logging.debug("extract trials info")
            cv_trials, rest_period, session_start, session_end = extract_trials(
                trials, epochs, invalid_times
            )

            # print(ecog_multichan.timestamps[0], ecog_multichan.timestamps[-1], session_start, session_end)

            data = Data(
                ecog=ecog_multichan,
                units=channels,
                speech=cv_trials,
                rest_period=rest_period,
                start=session_start,
                end=session_end,
                probes=probe,
                # These are all the string metadata that we have. Later, we'll use this for
                # keying into EmbeddingWithVocab embeddings.
                session=f"{session_id}",
                sortset=f"{subject_id}",
                subject=f"{subject_id}",
                **extra,
            )

            # get successful trials, and keep 20% for test, 10% for valid
            # print('save splits')
            (
                train_segments,
                valid_segments,
                test_segments,
            ) = split_and_get_train_valid_test(
                cv_trials, train_split=0.7, valid_split=0.1, test_split=0.2
            )

            # collect data slices for validation and test segments
            train_slices, train_slice_labels = collect_slices_wChecks(
                data, train_segments
            )
            valid_slices, valid_slice_labels = collect_slices_wChecks(
                data, valid_segments
            )
            test_slices, test_slice_labels = collect_slices_wChecks(data, test_segments)

            # Double check that the frequencies are balanced. TODO how to make train / valid / test balanced here?
            char_counts = {}
            _, char_counts["train"] = np.unique(
                np.array(train_slice_labels), return_counts=True
            )
            _, char_counts["valid"] = np.unique(
                np.array(valid_slice_labels), return_counts=True
            )
            _, char_counts["test"] = np.unique(
                np.array(test_slice_labels), return_counts=True
            )

            for fold in ["train", "valid", "test"]:
                logging.info("checking class balance in the current splits")
                logging.info(char_counts[fold])

            # np.testing.assert_allclose(
            #     char_counts["train"] / char_counts["train"].sum(),
            #     char_counts["test"] / char_counts["test"].sum(),
            #     )

            # np.testing.assert_allclose(
            #     char_counts["train"] / char_counts["train"].sum(),
            #     char_counts["valid"] / char_counts["valid"].sum(),
            #     )

            # saving to disk
            trial_descriptions = []
            # footprints = collections.defaultdict(list)
            for buckets, fold in [
                (train_slices, "train"),
                (valid_slices, "valid"),
                (test_slices, "test"),
            ]:
                for i, sample in enumerate(buckets):
                    basename = f"{session_id}_{i:05}"
                    filename = f"{basename}.pt"
                    path = os.path.join(processed_folder_path, fold, filename)
                    torch.save(sample, path)

                    # footprints[fold].append(os.path.getsize(path))
                    chunk_description = ChunkDescription(
                        id=basename,
                        duration=(sample.end - sample.start).item(),
                        start_time=sample.start.item(),
                    )

                    trial_descriptions.append(
                        TrialDescription(
                            id=basename,
                            footprints={
                                fold: os.path.getsize(path),
                            },
                            chunks={fold: [chunk_description]},
                        )
                    )
                    # trial_descriptions.append(TrialDescription(
                    #     id=basename,
                    #     footprints={},
                    #     chunks={fold: [chunk_description]},)
                    #     )

            # footprints = {k: int(np.mean(v)) for k, v in footprints.items()}

            # create description.yml
            recording_date = nwb_file.session_start_time.strftime("%Y%m%d")
            session = SessionDescription(  # there is still start_time & end_time variable but as null
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.DISCRETE_SPEAKING_CVSYLLABLE,
                fields={
                    RecordingTech.ECOG_ARRAY_ECOGS: "ecog.waveforms",
                    Output.SPEAKING_CVSYLLABLE: "speech.consonant_vowel_syllables",
                },
                trials=trial_descriptions,
            )

            # each subject has a sortset / use channel names as the units variables,
            # TODO we might want more intuitive naming?
            if not subject_id in sortsets:
                sortset_description = SortsetDescription(  # sortset <=> experimental container : the same probe placement => same subject (same task?)
                    id=subject_id,
                    subject=f"{experiment_name}_{subject_id}",
                    areas=[HomoSapiens.ventral_sensorimotor_cortex],
                    recording_tech=[RecordingTech.ECOG_ARRAY_ECOGS],
                    sessions=[],
                    units=channels.unit_name,
                )
                sortsets[subject_id] = sortset_description

            sortsets[subject_id].sessions.append(session)

            # for the same subjects, channel names should be the same
            assert torch.all(
                torch.tensor(
                    [
                        channels.unit_name[i] == sortsets[subject_id].units[i]
                        for i in range(len(sortsets[subject_id].units))
                    ]
                )
            )

            # helper.register_session(sortset_name, session) # This in fact register session description to one sortset
            # helper.register_sortset(experiment_name, sortset_description)

    # sortsets = sorted(list(sortsets.values()), key=lambda x: x.id)
    sortsets = sorted(list(sortsets.values()), key=lambda x: x.id)
    for per_subject_sortset in sortsets:
        helper.register_sortset(experiment_name, per_subject_sortset)

    # Create a description file for ease of reference.
    helper.register_dandiset(
        DandisetDescription(
            id="bouchard_chang_2020",
            origin_version="0.220126.2148",  # dandi version id
            derived_version="0.0.0",  # This variant
            metadata_version="0.0.0",
            source="https://dandiarchive.org/dandiset/000019",
            description="Human ECoG speaking consonant-vowel syllables from Bouchard & Chang (2020).",
            folds=["train", "valid", "test"],
            subjects=subjects,
            sortsets=[],
        )
    )

    description = helper.finalize()
    helper.write_to_disk(processed_folder_path, description)
