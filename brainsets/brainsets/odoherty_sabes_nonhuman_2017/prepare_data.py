import argparse
import datetime
import logging
from pathlib import Path
from typing import List, Tuple

from brainsets_utils.processing import signal
import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import pandas as pd

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.taxonomy import (
    Macaque,
    RecordingTech,
    Species,
    SubjectDescription,
    Task,
    Channel,
    Probe,
)

logging.basicConfig(level=logging.INFO)


SAMPLE_FREQUENCY = 24414.0625

# We could read this from one of the LFP hdf5 file, but it's small enough that we can
# just hardcode it.
channel_map = np.array(
    [
        [0, 0, -1000],
        [0, 2000, -1000],
        [1200, 800, -1000],
        [800, 1600, -1000],
        [400, 400, -1000],
        [-400, 2000, -1000],
        [800, 1200, -1000],
        [400, 2000, -1000],
        [0, 400, -1000],
        [2400, -1200, -1000],
        [1200, 1200, -1000],
        [2400, -800, -1000],
        [0, 800, -1000],
        [2000, -1200, -1000],
        [400, 800, -1000],
        [2000, -800, -1000],
        [0, 1200, -1000],
        [1600, -1200, -1000],
        [400, 1200, -1000],
        [1600, -800, -1000],
        [0, 1600, -1000],
        [1200, -1200, -1000],
        [400, 1600, -1000],
        [1200, -800, -1000],
        [-800, 0, -1000],
        [800, -1200, -1000],
        [-400, 0, -1000],
        [800, -800, -1000],
        [-400, 400, -1000],
        [400, -1200, -1000],
        [-800, 400, -1000],
        [400, -800, -1000],
        [-400, 800, -1000],
        [0, -1200, -1000],
        [-800, 800, -1000],
        [0, -800, -1000],
        [-400, 1200, -1000],
        [-400, -1200, -1000],
        [-800, 1200, -1000],
        [-400, -800, -1000],
        [-800, 1600, -1000],
        [-800, -800, -1000],
        [-400, 1600, -1000],
        [0, -400, -1000],
        [-400, 2400, -1000],
        [-800, -400, -1000],
        [-800, 2000, -1000],
        [-400, -400, -1000],
        [2800, -400, -1000],
        [2000, 1200, -1000],
        [2800, -800, -1000],
        [2800, 2000, -1000],
        [2800, 0, -1000],
        [1600, 800, -1000],
        [2800, 400, -1000],
        [2400, 2000, -1000],
        [2000, 400, -1000],
        [2000, 800, -1000],
        [2800, 800, -1000],
        [2400, 2400, -1000],
        [2400, 400, -1000],
        [1600, 1200, -1000],
        [2800, 1200, -1000],
        [2000, 2000, -1000],
        [2400, 800, -1000],
        [1200, 1600, -1000],
        [2800, 1600, -1000],
        [2000, 2400, -1000],
        [2400, 1200, -1000],
        [2000, 1600, -1000],
        [2400, 1600, -1000],
        [1600, 2400, -1000],
        [1600, -400, -1000],
        [1600, 1600, -1000],
        [1600, 0, -1000],
        [1200, 2400, -1000],
        [1200, -400, -1000],
        [1600, 2000, -1000],
        [1200, 0, -1000],
        [800, 2400, -1000],
        [800, -400, -1000],
        [1200, 2000, -1000],
        [1200, 400, -1000],
        [400, 2400, -1000],
        [800, 0, -1000],
        [800, 2000, -1000],
        [800, 400, -1000],
        [0, 2400, -1000],
        [400, -400, -1000],
        [2400, -400, -1000],
        [800, 800, -1000],
        [2400, 0, -1000],
        [400, 0, -1000],
        [2000, -400, -1000],
        [1600, 400, -1000],
        [2000, 0, -1000],
    ]
)


def extract_relevant_probes(probes, chan_names, animal, sortset_id):
    """
    Assemble probe information.

    ..note::
        It's a bit awkward because we have the info in the
        case of local field potentials, but not in the case of spikes.
    """
    relevant_probe_names = set(
        [f"odoherty_sabes_{animal}_{x[:2]}".lower() for x in chan_names]
    )
    relevant_probes = []
    for probe in probes:
        if probe.id in relevant_probe_names:
            relevant_probes.append(probe)
    if len(relevant_probes) == 0:
        raise ValueError(f"No probes found for {sortset_id}")


def extract_probe_description() -> list[Probe]:
    # In this case, there are exactly 4 probes, 2 for each animal. We have the exact
    # locations for the probes that have local field potential info, (indy m1), but
    # not for the ones that don't.
    infos = [
        ("indy_m1", Macaque.primary_motor_cortex, "M1"),
        ("indy_s1", Macaque.primary_somatosensory_cortex, "S1"),
        ("loco_m1", Macaque.primary_motor_cortex, "M1"),
        ("loco_s1", Macaque.primary_somatosensory_cortex, "S1"),
    ]

    descriptions = []
    for suffix, area, name in infos:
        channels = [
            Channel(
                id=f"{name} {i+1:03}",
                local_index=i,
                relative_x_um=channel_map[i, 0] if "suffix" == "indy_m1" else 0,
                relative_y_um=channel_map[i, 1] if "suffix" == "indy_m1" else 0,
                relative_z_um=channel_map[i, 2] if "suffix" == "indy_m1" else 0,
                area=area,
            )
            for i in range(96)
        ]

        description = Probe(
            id=f"odoherty_sabes_{suffix}",
            type=RecordingTech.UTAH_ARRAY,
            wideband_sampling_rate=SAMPLE_FREQUENCY,
            waveform_sampling_rate=SAMPLE_FREQUENCY,
            lfp_sampling_rate=500,
            waveform_samples=48,
            channels=channels,
        )
        descriptions.append(description)

    return descriptions


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """

    def _identify_outliers(cursor, threshold=6000):
        """
        Helper to identify outliers in the behavior data.
        Outliers are defined as points where the hand acceleration is greater than a
        threshold. This is a simple heuristic to identify when the monkey is moving
        the hand quickly when angry or frustrated.
        An additional step is to dilate the binary mask to flag the surrounding points.
        """
        hand_acc_norm = np.linalg.norm(cursor.acc, axis=1)
        mask = hand_acc_norm > threshold
        structure = np.ones(100, dtype=bool)
        # Dilate the binary mask
        dilated = binary_dilation(mask, structure)
        return dilated

    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    subtask_index = np.ones(len(timestamps), dtype=np.int64) * int(Task.REACHING.RANDOM)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel / 200.0,  # TODO: this is used to match the other datasets
        acc=cursor_acc,
        subtask_index=subtask_index,
        domain="auto",
    )

    # The position of the working fingertip in Cartesian coordinates (z, -x, -y), as
    # reported by the hand tracker in cm. Thus the cursor position is an affine
    # transformation of fingertip position.
    finger = IrregularTimeSeries(
        timestamps=timestamps,
        pos=finger_pos[:, :3],
        vel=finger_vel[:, :3],
        subtask_index=subtask_index,
        domain="auto",
    )
    if finger_pos.shape[1] == 6:
        finger.orientation = finger_pos[:, 3:]
        finger.angular_vel = finger_vel[:, 3:]

    # find outliers
    mask = _identify_outliers(cursor)
    cursor.subtask_index[mask] = int(Task.REACHING.OUTLIER)
    finger.subtask_index[mask] = int(Task.REACHING.OUTLIER)

    # TODO move this extracted data to trials
    # TODO: Refactor this for reusability.
    # Extract two traces that capture the target and movement onsets.
    # Similar to https://www.biorxiv.org/content/10.1101/2021.11.21.469441v3.full.pdf
    cursor_vel_abs = np.sqrt(cursor_vel[:, 0] ** 2 + cursor_vel[:, 1] ** 2)

    # Dirac delta whenever the target changes.
    delta_time = np.sqrt((np.diff(target_pos, axis=0) ** 2).sum(axis=1)) >= 1e-9
    delta_time = np.concatenate([delta_time, [0]])

    tics = np.where(delta_time)[0]

    thresh = 0.2

    # Find the maximum for each integer value of period.
    max_times = np.zeros(len(tics) - 1, dtype=int)
    reaction_times = np.zeros(len(tics) - 1, dtype=int)
    for i in range(len(tics) - 1):
        max_vel = cursor_vel_abs[tics[i] : tics[i + 1]].max()
        reaction_times[i] = np.where(
            cursor_vel_abs[tics[i] : tics[i + 1]] >= thresh * max_vel
        )[0][0]
        max_times[i] = reaction_times[i] + tics[i]

    # Transform it back to a Dirac delta.
    start_times = np.zeros_like(delta_time)
    start_times[max_times] = 1

    # behavior = IrregularTimeSeries(
    #     target_pos=torch.tensor(target_pos),
    #     trial_onset_offset=torch.stack(
    #         [torch.tensor(start_times), torch.tensor(delta_time)], dim=1
    #     ),
    # )

    return cursor, finger


def extract_lfp(
    h5file: h5py.File, channels: List[str]
) -> Tuple[RegularTimeSeries, Data]:
    """Extract the LFP from the h5 file."""
    logging.info("Broadband data attached. Computing LFP.")
    timestamps = h5file.get("/acquisition/timeseries/broadband/timestamps")[:].squeeze()

    # unfortunately, we have to chunk this because it's too big to fit in memory.
    n_samples_per_chunk = int(SAMPLE_FREQUENCY * 128)
    assert n_samples_per_chunk % 1000 == 0
    n_chunks = int(
        np.ceil(
            h5file.get("/acquisition/timeseries/broadband/data").shape[0]
            / n_samples_per_chunk
        )
    )

    lfps = []
    t_lfps = []
    for i in tqdm(range(n_chunks)):
        # Slow, iterative algorithm to prevent OOM issues.
        # Easiest would be to do this by channel but the memory layout in hdf5 doesn't
        # permit doing this efficiently.
        rg = slice(i * n_samples_per_chunk, (i + 1) * n_samples_per_chunk)
        broadband = h5file.get("/acquisition/timeseries/broadband/data")[rg, :]
        lfp, t_lfp = signal.downsample_wideband(
            broadband, timestamps[rg], SAMPLE_FREQUENCY
        )

        lfps.append(lfp.squeeze())
        t_lfps.append(t_lfp)

    lfp = np.concatenate(lfps, axis=0)
    t_lfp = np.concatenate(t_lfps)

    assert lfp.shape[0] == t_lfp.shape[0]
    assert lfp.ndim == 2
    assert t_lfp.ndim == 1

    lfp_bands, t_lfp_bands, names = signal.extract_bands(lfp, t_lfp)

    assert np.std(np.diff(t_lfp_bands)) < 1e-5  # less than 10 ns
    sampling_rate = 1 / np.mean(np.diff(t_lfp_bands))

    lfp = RegularTimeSeries(
        sampling_rate=sampling_rate,
        lfp=lfp_bands,
        domain=Interval(
            t_lfp_bands[0], t_lfp_bands[0] + (len(t_lfp_bands) - 1) / sampling_rate
        ),
    )

    return lfp, np.array(channels).astype("S"), np.array(names).astype("S")


def extract_spikes(h5file: h5py.File):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings)
    units.
    """

    # helpers specific to spike extraction
    def _to_ascii(vector):
        return ["".join(chr(char) for char in row) for row in vector]

    def _load_references_2d(h5file, ref_name):
        return [[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]]

    spikesvec = _load_references_2d(h5file, "spikes")
    waveformsvec = _load_references_2d(h5file, "wf")

    # this is slightly silly but we can convert channel names back to an ascii token
    # this way.
    chan_names = _to_ascii(
        np.array(_load_references_2d(h5file, "chan_names")).squeeze()
    )

    spikes = []
    unit_index = []
    types = []
    waveforms = []
    unit_meta = []

    # The 0'th spikesvec corresponds to unsorted thresholded units, the rest are sorted.
    suffixes = ["unsorted"] + [f"sorted_{i:02}" for i in range(1, 11)]
    type_map = [int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS)] + [
        int(RecordingTech.UTAH_ARRAY_SPIKES)
    ] * 10

    # Map from common names to brodmann areas
    bas = {"s1": 3, "m1": 4}

    encountered = set()

    unit_index_delta = 0
    for j in range(len(spikesvec)):
        crossings = spikesvec[j]
        for i in range(len(crossings)):
            spiketimes = crossings[i][:][0]
            if spiketimes.ndim == 0:
                continue

            spikes.append(spiketimes)
            area, channel_number = chan_names[i].split(" ")

            unit_name = f"{chan_names[i]}/{suffixes[j]}"

            unit_index.append([unit_index_delta] * len(spiketimes))
            types.append(np.ones_like(spiketimes, dtype=np.int64) * type_map[j])

            if unit_name in encountered:
                raise ValueError(f"Duplicate unit name: {unit_name}")
            encountered.add(unit_name)

            wf = np.array(waveformsvec[j][i][:])
            unit_meta.append(
                {
                    "count": len(spiketimes),
                    "channel_name": chan_names[i],
                    "id": unit_name,
                    "area_name": area,
                    "channel_number": channel_number,
                    "unit_number": j,
                    "ba": bas[area.lower()],
                    "type": type_map[j],
                    "average_waveform": wf.mean(axis=1)[:48],
                    # Based on https://zenodo.org/record/1488441
                    "waveform_sampling_rate": SAMPLE_FREQUENCY,
                }
            )
            waveforms.append(wf.T)
            unit_index_delta += 1

    spikes = np.concatenate(spikes)
    waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)

    spikes = IrregularTimeSeries(
        timestamps=spikes,
        unit_index=unit_index,
        waveforms=waveforms,
        domain="auto",
    )
    spikes.sort()

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))
    return spikes, units, chan_names


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
        experiment_name="odoherty_sabes_nonhuman_2017",
        origin_version="583331",  # Zenodo version
        derived_version="1.0.0",  # This variant
        source="https://zenodo.org/record/583331",
        description="The behavioral task was to make self-paced reaches to targets "
        "arranged in a grid (e.g. 8x8) without gaps or pre-movement delay intervals. "
        "One monkey reached with the right arm (recordings made in the left hemisphere)"
        "The other reached with the left arm (right hemisphere). In some sessions "
        "recordings were made from both M1 and S1 arrays (192 channels); "
        "in most sessions M1 recordings were made alone (96 channels).",
    )

    # common to all sessions
    probes = extract_probe_description()

    # find all files with extension .mat in folder_path
    for file_path in sorted(find_files_by_extension(db.raw_folder_path, ".mat")):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # load file
            h5file = h5py.File(file_path, "r")

            # extract experiment metadata
            # determine session_id and sortset_id
            session_id = Path(file_path).stem  # type: ignore

            # TODO do recording from the same day share the same sortset
            sortset_id = session_id[:-3]
            assert sortset_id.count("_") == 1, f"Unexpected file name: {sortset_id}"

            # get subject and recording date from sortset_id
            animal, recording_date = sortset_id.split("_")
            # TODO we don't have any info about age or sex for these subjects
            subject = SubjectDescription(
                id=animal,
                species=Species.MACACA_MULATTA,
            )
            session.register_subject(subject)

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity, unit metadata and channel names info
            spikes, units, chan_names = extract_spikes(h5file)

            # register sortset
            session.register_sortset(
                id=session_id,  # sortset_id,
                units=units,
            )

            # extract behavior
            cursor, finger = extract_behavior(h5file)

            # since this dataset has both LFP and spike data, we first check if
            # broadband datasets exist.
            # check if the broadband data file exists.
            broadband_path = (
                Path(db.raw_folder_path) / "broadband" / f"{session_id}.nwb"
            )
            broadband = broadband_path.exists()

            # extract Local Field Potential (LFP) data if exists
            extras = dict()
            if broadband:
                # Load the associated broadband data.
                broadband_file = h5py.File(broadband_path, "r")
                extras["lfps"], extras["lfp_channels"], extras["lfp_band_names"] = (
                    extract_lfp(broadband_file, chan_names)
                )

            # extract probes relevant to our particular session using channel info
            relevant_probes = extract_relevant_probes(
                probes, chan_names, animal, sortset_id
            )

            data = Data(
                probes=relevant_probes,
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                cursor=cursor,
                finger=finger,
                # other info like LFP
                **extras,
                domain=cursor.domain,
            )

            session.register_data(data)

            # slice the session into 10 blocks then randomly split them into train,
            # validation and test sets, using a 8/1/1 ratio.
            intervals = Interval.linspace(data.domain.start[0], data.domain.end[-1], 10)
            [
                train_sampling_intervals,
                valid_sampling_intervals,
                test_sampling_intervals,
            ] = intervals.split([8, 1, 1], shuffle=True, random_seed=42)

            # save samples
            session.register_split("train", train_sampling_intervals)
            session.register_split("valid", valid_sampling_intervals)
            session.register_split("test", test_sampling_intervals)

            # save data to disk
            session.save_to_disk()

            # close the file
            h5file.close()

    db.finish()


if __name__ == "__main__":
    main()
