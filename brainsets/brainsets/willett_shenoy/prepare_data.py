import argparse
import collections
from pathlib import Path
from typing import List

import dateutil
import numpy as np
import numpy.testing as npt
import torch
from scipy.io import loadmat

from brainsets_utils.processing import signal
from brainsets_utils.utils.data import (
    Channel,
    Data,
    Hemisphere,
    Interval,
    IrregularTimeSeries,
    Probe,
)
from brainsets_utils.taxonomy import Output, SessionDescription, Task, writing
from brainsets_utils.taxonomy.description_helper import DescriptionHelper
from brainsets_utils.taxonomy.macaque import Macaque
from brainsets_utils.taxonomy.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    RecordingTech,
    SortsetDescription,
    Species,
    SubjectDescription,
    TrialDescription,
)

experiment_name = "willett_shenoy"
subject_name = f"{experiment_name}_t5"


def generate_probe_description() -> List[Probe]:
    probes = []
    for location in ["lateral", "medial"]:
        channels = []
        for i in range(96):
            channels = [
                Channel(
                    f"{subject_name}_{location}_{i:03}",
                    i,
                    0,
                    0,
                    0,
                    Macaque.primary_motor_cortex,  # This is human but using Macaque nomenclature for now.
                    Hemisphere.LEFT,
                )
            ]

        probes.append(
            Probe(
                f"{subject_name}_{location}",
                RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
                0,
                0,
                0,
                0,
                channels=channels,
            )
        )
    return probes


def process_single_letters(
    session_path: Path,
    processed_folder_path: Path,
    straight_lines: bool = False,
):
    sortset_name = f"{subject_name}/{session_path.parent.parts[-1]}"
    probes = generate_probe_description()
    if straight_lines:
        session_name = f"{sortset_name}_straight_lines"
    else:
        session_name = f"{sortset_name}_single_letters"
    single_letters_data = loadmat(session_path)

    labels = []
    spike_cubes = []
    train_masks = []
    valid_masks = []
    test_masks = []

    nchars = 0
    for key in single_letters_data.keys():
        if not key.startswith("neuralActivityCube_"):
            continue

        letter = key[len("neuralActivityCube_") :]
        resolved = False
        found = False
        try:
            resolved = writing.Character[letter]
            found = True
        except:
            pass

        try:
            resolved = writing.Line[letter]
            found = True
        except:
            pass

        assert found

        data = single_letters_data[f"neuralActivityCube_{letter}"]

        spike_cubes.append(data)
        labels.append(np.array([int(resolved)] * len(data)))
        valid_mask = np.arange(len(data)) % 9 == 2
        test_mask = np.arange(len(data)) % 9 == 5
        train_mask = ~valid_mask & ~test_mask
        train_masks.append(train_mask)
        valid_masks.append(valid_mask)
        test_masks.append(test_mask)
        nchars += 1

    print(f"Number of distinct characters: {nchars}")
    print(f"Number of repeats per character: {len(train_mask)}")

    labels = np.stack(labels, axis=1).ravel()
    train_masks = np.stack(train_masks, axis=1).ravel()
    valid_masks = np.stack(valid_masks, axis=1).ravel()
    test_masks = np.stack(test_masks, axis=1).ravel()

    # Did we do the split the right way around? Double check.
    # If we did the split the wrong way around, we should switch rapidly between train
    # and test folds (fold is fast dim, character is slow dim).
    # This will make it easier to e.g. use exactly one trial per character for
    # calibration.
    assert abs(np.diff(1 * train_masks)).sum() < 20

    spike_cubes = np.stack(spike_cubes, axis=1)
    spike_cubes = spike_cubes.reshape(
        (spike_cubes.shape[0] * spike_cubes.shape[1], spike_cubes.shape[2], -1)
    )
    print(f"Number of threshold crossings: {spike_cubes.sum()}")

    assert len(test_masks) == len(labels)

    folds = np.where(train_masks, "train", np.where(valid_masks, "valid", "test"))

    assert spike_cubes.shape[0] == len(test_masks)

    # We only select 1.6 seconds, from -.3 to 1.3 seconds, where the go period is from
    # 0 to 1 seconds.
    spike_cubes = spike_cubes[:, 59:, :]
    ts = np.arange(0.5, 0.5 + spike_cubes.shape[1]) / 100.0  # 100 Hz sampling rate

    channel_prefix = f"{sortset_name}/channel_"
    trials, units = signal.cube_to_long(ts, spike_cubes, channel_prefix=channel_prefix)

    # TODO: use the geometry map.
    counters = collections.defaultdict(int)
    letters = collections.defaultdict(list)
    trial_descriptions = []

    for trial, label, fold in zip(trials, labels, folds):
        stimuli_segments = Interval(
            start=torch.Tensor([0]),
            end=torch.Tensor([1.6]),
            timestamps=torch.Tensor(
                [0.8]
            ),  # Assign the label to the center of the interval.
            letters=torch.tensor([[int(label)]]),
            behavior_type=torch.tensor([0]),
        )
        data = Data(
            spikes=trial,
            units=units,
            behavior=None,
            stimuli_segments=stimuli_segments,
            start=0,
            end=ts[-1],
            probes=probes,
            session=session_name,
            sortset=sortset_name,
            subject=subject_name,
        )
        i = counters[fold]
        basename = f"{session_name}_{i:05}"
        filename = f"{basename}.pt"
        path = processed_folder_path / fold / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(data, path)

        chunk_description = ChunkDescription(
            id=basename,
            duration=1,
            start_time=0,  # Not clear from the blocks here, unfortunately.
        )

        trial_descriptions.append(
            TrialDescription(
                id=basename,
                footprints={},
                chunks={fold.item(): [chunk_description]},
            )
        )

        counters[fold] += 1
        letters[fold].append(label)

    # Double check that the frequencies are balanced.
    char_counts = {}
    for fold in ["train", "valid", "test"]:
        _, char_counts[fold] = np.unique(np.array(letters[fold]), return_counts=True)

    npt.assert_allclose(
        char_counts["train"] / char_counts["train"].sum(),
        char_counts["test"] / char_counts["test"].sum(),
    )

    npt.assert_allclose(
        char_counts["train"] / char_counts["train"].sum(),
        char_counts["valid"] / char_counts["valid"].sum(),
    )

    if straight_lines:
        task = Task.DISCRETE_WRITING_LINE
        output = Output.WRITING_LINE
    else:
        task = Task.DISCRETE_WRITING_CHARACTER
        output = Output.WRITING_CHARACTER

    session = SessionDescription(
        id=session_name,
        recording_date=dateutil.parser.parse(
            single_letters_data.get("blockStartDates")[0][0].item()
        ),
        task=task,
        fields={
            RecordingTech.UTAH_ARRAY: "spikes",
            output: "stimuli_segments.letters",
        },
        trials=trial_descriptions,
    )

    return sortset_name, units.unit_name, session


# Load the straightLines.mat file
def load_straight_lines(session_path):
    straight_lines_data = loadmat(session_path + "straightLines.mat")

    # Spikes data structure to hold neuralActivityCube
    spikes = {}
    spikes["neural_activity_cubes"] = straight_lines_data.get("neuralActivityCube_{x}")
    spikes["neural_activity_time_series"] = straight_lines_data.get(
        "neuralActivityTimeSeries"
    )

    # Behaviour data structure to hold different behavioral aspects
    behaviour = {}
    behaviour["clock_time_series"] = straight_lines_data.get("clockTimeSeries")
    behaviour["block_nums_time_series"] = straight_lines_data.get("blockNumsTimeSeries")
    behaviour["go_cue_onset_time_bin"] = straight_lines_data.get("goCueOnsetTimeBin")
    behaviour["delay_cue_onset_time_bin"] = straight_lines_data.get(
        "delayCueOnsetTimeBin"
    )

    # Outputs data structure to hold output details
    outputs = {}
    outputs["means_per_block"] = straight_lines_data.get("meansPerBlock")
    outputs["std_across_all_data"] = straight_lines_data.get("stdAcrossAllData")
    outputs["array_geometry_map"] = straight_lines_data.get("arrayGeometryMap")

    return spikes, behaviour, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    partitions = loadmat(
        Path(raw_folder_path)
        / "RNNTrainingSteps"
        / "trainTestPartitions_HeldOutBlocks.mat"
    )
    helper = DescriptionHelper()

    files = sorted(
        list((Path(raw_folder_path) / "Datasets").glob("*/singleLetters.mat"))
        + list((Path(raw_folder_path) / "Datasets").glob("*/straightLines.mat"))
    )

    for file in files:
        sortset_name, channel_names, session = process_single_letters(
            file,
            Path(processed_folder_path),
            "straightLines.mat" in file.parts,
        )

        sortset_description = SortsetDescription(
            id=sortset_name,
            subject=subject_name,
            areas=[Macaque.primary_motor_cortex],
            recording_tech=[RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS],
            sessions=[],
            units=channel_names,
        )

        helper.register_session(sortset_name, session)
        helper.register_sortset(experiment_name, sortset_description)

    helper.register_dandiset(
        DandisetDescription(
            id=experiment_name,
            origin_version="0.0.0",
            derived_version="0.0.0",
            metadata_version="0.0.0",
            source="https://datadryad.org/stash/dataset/doi:10.5061/dryad.wh70rxwmv",
            description="Handwriting BCI data from Willett and Shenoy",
            folds=["train", "valid", "test"],
            subjects=[
                SubjectDescription(
                    id=subject_name,
                    species=Species.HOMO_SAPIENS,
                )
            ],
            sortsets=[],
        )
    )

    description = helper.finalize()
    helper.write_to_disk(processed_folder_path, description)
