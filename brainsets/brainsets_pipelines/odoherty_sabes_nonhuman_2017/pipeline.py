# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["zenodo_get==2.0.0"]
# ///

import os
import datetime
import hashlib
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Species, Task

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "odoherty_sabes_nonhuman_2017"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        raw_dir.mkdir(exist_ok=True, parents=True)

        # list files to download
        r_fd = os.popen(f"zenodo_get -w - 3854034")
        manifest_out = r_fd.read()
        exit_code = r_fd.close()
        if exit_code is not None and exit_code != 0:
            raise RuntimeError("zenodo_get (list) failed with exit code {exit_code}")

        # fetch md5sums
        exit_code = os.system(f'cd "{raw_dir}" && zenodo_get -m 3854034')
        if exit_code != 0:
            raise RuntimeError("zenodo_get (md5) failed")

        # parse md5sums
        with open(raw_dir / "md5sums.txt", "r") as f:
            md5sums = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                md5, filename = parts
                if filename.startswith("*"):
                    filename = filename[1:]
                md5sums[filename] = md5.lower()

        # create manifest
        manifest_list = []
        for line in manifest_out.split("\n"):
            line = line.strip()
            if line:
                file_name = line.split("/")[-2]
                if not file_name.endswith(".mat"):
                    continue
                manifest_list.append(
                    {
                        "session_id": file_name.split(".")[0],
                        "file": file_name,
                        "url": line,
                        "md5": md5sums[file_name],
                    }
                )

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = self.raw_dir / Path(manifest_item.file)

        if (
            not self.args.redownload
            and fpath.exists()
            and hashlib.md5(fpath.read_bytes()).hexdigest() == manifest_item.md5
        ):
            logging.info(f"Skipping download for {fpath} because it already exists")
            self.update_status("Skipped Download")
            return fpath

        # download file
        exit_code = os.system(f'wget -O "{fpath}" "{manifest_item.url}"')
        if exit_code != 0:
            logging.error(
                f"Failed to download {manifest_item.url} with exit code {exit_code}"
            )
            self.update_status("Failed Download")
            raise RuntimeError(f"Failed to download {manifest_item.url}")

        # verify md5sum
        md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
        if md5 != manifest_item.md5:
            logging.error(f"MD5 mismatch for {fpath}: {md5} != {manifest_item.md5}")
            self.update_status("Failed Download Verification")
            raise ValueError(f"MD5 mismatch for {fpath}: {md5} != {manifest_item.md5}")
        return fpath

    def process(self, fpath):
        processed_dir = self.processed_dir / self.brainset_id
        processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id="odoherty_sabes_nonhuman_2017",
            origin_version="583331",  # Zenodo version
            derived_version="1.0.0",
            source="https://zenodo.org/record/583331",
            description="The behavioral task was to make self-paced reaches to targets "
            "arranged in a grid (e.g. 8x8) without gaps or pre-movement delay intervals. "
            "One monkey reached with the right arm (recordings made in the left hemisphere)"
            "The other reached with the left arm (right hemisphere). In some sessions "
            "recordings were made from both M1 and S1 arrays (192 channels); "
            "in most sessions M1 recordings were made alone (96 channels).",
        )

        logging.info(f"Processing file: {fpath}")

        # open file
        self.update_status("Loading mat")
        h5file = h5py.File(fpath, "r")

        # extract experiment metadata
        # determine session_id and sortset_id
        session_id = Path(fpath).stem  # type: ignore
        if not self.args.reprocess and (processed_dir / f"{session_id}.h5").exists():
            logging.info(
                f"Skipping processing for {session_id} because it already exists"
            )
            self.update_status("Skipped Processing")
            return
        device_id = session_id[:-3]
        assert device_id.count("_") == 1, f"Unexpected file name: {device_id}"

        animal, recording_date = device_id.split("_")
        subject = SubjectDescription(
            id=animal,
            species=Species.MACACA_MULATTA,
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.REACHING,
        )

        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # extract spiking activity, unit metadata and channel names info
        self.update_status("Extracting spikes")
        spikes, units = extract_spikes(h5file)

        # extract behavior
        self.update_status("Extracting behavior")
        cursor, finger = extract_behavior(h5file)
        no_movement_segments = detect_no_movement(cursor)

        # close file
        h5file.close()

        # register session
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            # stimuli and behavior
            cursor=cursor,
            finger=finger,
            no_movement_segments=no_movement_segments,
            domain=cursor.domain,
        )

        self.update_status("Creating splits")
        train_sampling_intervals, valid_sampling_intervals, test_sampling_intervals = (
            split_intervals(data)
        )
        # set the domains
        data.set_train_domain(train_sampling_intervals)
        data.set_valid_domain(valid_sampling_intervals)
        data.set_test_domain(test_sampling_intervals)

        # save data to disk
        self.update_status("Storing")
        path = processed_dir / f"{session_id}.h5"
        with h5py.File(str(path), "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence.
    """

    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    expected_period = 0.004
    assert np.all(np.abs(np.diff(timestamps) - expected_period) < 1e-4)

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        domain="auto",
    )

    # The position of the working fingertip in Cartesian coordinates (z, -x, -y), as
    # reported by the hand tracker in cm. Thus the cursor position is an affine
    # transformation of fingertip position.
    finger = IrregularTimeSeries(
        timestamps=timestamps,
        pos_3d=finger_pos[:, :3],
        vel_3d=finger_vel[:, :3],
        domain="auto",
    )
    if finger_pos.shape[1] == 6:
        finger.orientation = finger_pos[:, 3:]
        finger.angular_vel = finger_vel[:, 3:]

    assert cursor.is_sorted()
    assert finger.is_sorted()

    return cursor, finger


def detect_no_movement(cursor):
    """Detect segments where the cursor is not moving"""
    mask = np.linalg.norm(cursor.vel, axis=1) < 10.0

    # convert to interval, you need to find the start and end of the outlier segments
    start = cursor.timestamps[np.where(np.diff(mask.astype(int)) == 1)[0]]
    if mask[0]:
        start = np.insert(start, 0, cursor.timestamps[0])

    end = cursor.timestamps[np.where(np.diff(mask.astype(int)) == -1)[0]]
    if mask[-1]:
        end = np.append(end, cursor.timestamps[-1])

    no_movement_segments = Interval(start=start, end=end)
    assert no_movement_segments.is_disjoint()

    no_movement_segments = no_movement_segments.select_by_mask(
        (no_movement_segments.end - no_movement_segments.start) > 10.0
    )
    return no_movement_segments


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
                    "type": type_map[j],
                    "average_waveform": wf.mean(axis=1)[:48],
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
    return spikes, units


def split_intervals(data):
    # slice the session into 10 blocks then randomly split them into train,
    # validation and test sets, using a 8/1/1 ratio.
    task_domain = data.domain
    if len(data.no_movement_segments) > 0:
        task_domain = task_domain.difference(data.no_movement_segments)
        task_domain = task_domain.select_by_mask(
            task_domain.end - task_domain.start > 10.0
        )

    intervals = Interval.linspace(task_domain.start[0], task_domain.end[-1], 10)
    if len(data.no_movement_segments) > 0:
        task_ratio = []
        for start, end in zip(intervals.start, intervals.end):
            intersection = task_domain & Interval(start, end)
            duration = np.sum(intersection.end - intersection.start)
            task_ratio.append(duration / (end - start))

        task_ratio = np.array(task_ratio)
        task_ratio_with_index = np.column_stack((np.arange(len(intervals)), task_ratio))
        rng = np.random.default_rng(42)
        rng.shuffle(task_ratio_with_index)
        sorted_index = task_ratio_with_index[task_ratio_with_index[:, 1].argsort()][
            :, 0
        ].astype(int)

        train_sampling_intervals = Interval(
            start=intervals.start[sorted_index[:8]], end=intervals.end[sorted_index[:8]]
        )
        valid_sampling_intervals = Interval(
            start=intervals.start[sorted_index[8:9]],
            end=intervals.end[sorted_index[8:9]],
        )
        test_sampling_intervals = Interval(
            start=intervals.start[sorted_index[9:]], end=intervals.end[sorted_index[9:]]
        )

        train_sampling_intervals.sort()
        valid_sampling_intervals.sort()
        test_sampling_intervals.sort()

        train_sampling_intervals = train_sampling_intervals.difference(
            data.no_movement_segments
        )
        valid_sampling_intervals = valid_sampling_intervals.difference(
            data.no_movement_segments
        )
        test_sampling_intervals = test_sampling_intervals.difference(
            data.no_movement_segments
        )

        train_sampling_intervals = train_sampling_intervals.select_by_mask(
            train_sampling_intervals.end - train_sampling_intervals.start > 10.0
        )
        valid_sampling_intervals = valid_sampling_intervals.select_by_mask(
            valid_sampling_intervals.end - valid_sampling_intervals.start > 10.0
        )
        test_sampling_intervals = test_sampling_intervals.select_by_mask(
            test_sampling_intervals.end - test_sampling_intervals.start > 10.0
        )
    else:
        [
            train_sampling_intervals,
            valid_sampling_intervals,
            test_sampling_intervals,
        ] = intervals.split([8, 1, 1], shuffle=True, random_seed=42)
    return train_sampling_intervals, valid_sampling_intervals, test_sampling_intervals
