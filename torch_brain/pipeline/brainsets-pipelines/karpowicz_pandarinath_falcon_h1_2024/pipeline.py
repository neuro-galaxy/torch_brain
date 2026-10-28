# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///
"""FALCON H1 (Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks): a human participant
with tetraplegia (Pitt BCI) attempting 7-DoF reach and grasp in open loop; 176
threshold-crossing channels from motor cortex, 7-D kinematic velocity as decoding target.
Session dates in the files are shifted for de-identification and are kept as given.
"""

import datetime
import re
from argparse import ArgumentParser

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.dandi import download_file, get_nwb_asset_list

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

FALCON_SPLITS = ("held_in_calib", "held_in_minival", "held_out_calib")
KINEMATIC_DIMS = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")


class Pipeline(BrainsetPipeline):
    brainset_id = "karpowicz_pandarinath_falcon_h1_2024"
    dandiset_id = "DANDI:000954/draft"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        manifest = []
        for asset in get_nwb_asset_list(cls.dandiset_id):
            split, stamp = parse_filename(asset.path)
            manifest.append(
                {
                    "id": f"{split}_{stamp}",
                    "path": asset.path,
                    "url": asset.download_url,
                }
            )
        return pd.DataFrame(manifest).set_index("id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        return download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )

    def process(self, fpath):
        split, stamp = parse_filename(str(fpath))
        session_id = f"human_pitt_{stamp}_{split}"
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000954/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000954",
            description="FALCON H1: threshold crossings from 176 motor-cortex channels "
            "of a human participant with tetraplegia attempting 7-DoF reach and grasp in "
            "open loop, with 7-D kinematic velocity as the decoding target. Held-in "
            "sessions are for training; held-out sessions are later recording days, "
            "released as small calibration sets for few-shot adaptation. Dates are shifted "
            "for de-identification.",
        )

        self.update_status("Loading NWB")
        with NWBHDF5IO(fpath, "r") as io:
            nwbfile = io.read()

            subject = SubjectDescription(id="human_pitt", species="HOMO_SAPIENS")
            session_description = SessionDescription(
                id=session_id,
                recording_date=datetime.datetime.strptime(stamp, "%Y%m%dT%H%M%S"),
                task="REACHING",
            )
            device_description = DeviceDescription(
                id=f"human_pitt_{stamp[:8]}",
                recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS",
            )

            self.update_status("Extracting Spikes")
            spikes, units = extract_spikes(nwbfile)

            self.update_status("Extracting Kinematics")
            vel = nwbfile.acquisition["OpenLoopKinematicsVelocity"]
            pos = nwbfile.acquisition["OpenLoopKinematics"]
            # FALCON stores the sampling period (0.02 s) in the `rate` field
            period = float(vel.rate)
            timestamps = (
                float(vel.starting_time or 0.0) + np.arange(vel.data.shape[0]) * period
            )
            kinematics = RegularTimeSeries(
                sampling_rate=1.0 / period,
                pos=pos.data[:].astype(np.float32),
                vel=vel.data[:].astype(np.float32),
                domain_start=timestamps[0],
            )
            eval_mask = nwbfile.acquisition["eval_mask"].data[:].astype(bool)
            eval_intervals = mask_to_interval(eval_mask, timestamps)
            trials = trials_from_numbers(
                nwbfile.acquisition["TrialNum"].data[:], timestamps
            )

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            kinematics=kinematics,
            kinematic_dims=ArrayDict(name=np.array(KINEMATIC_DIMS, dtype="S")),
            trials=trials,
            falcon_eval_intervals=eval_intervals,
            domain=kinematics.domain,
        )
        data.falcon_split = split
        assign_domains(data, split)

        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file)


def parse_filename(path: str) -> tuple[str, str]:
    r"""``sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb`` ->
    ``("held_in_calib", "19250108T110520")``."""
    name = path.split("/")[-1]
    m = re.search(
        r"sub-[^-_]+-(held-(?:in|out))-(calib|minival)_ses-(\d{8}T\d{6})", name
    )
    if m is None:
        raise ValueError(f"Unexpected FALCON file name: {name}")
    split = f"{m.group(1).replace('-', '_')}_{m.group(2)}"
    assert split in FALCON_SPLITS, split
    return split, m.group(3)


def extract_spikes(nwbfile) -> tuple[IrregularTimeSeries, ArrayDict]:
    r"""H1 units are threshold-crossing channels without an electrodes table."""
    spike_times = nwbfile.units.spike_times_index
    timestamps, unit_index = [], []
    for i in range(len(nwbfile.units)):
        t = np.asarray(spike_times[i], dtype=np.float64)
        timestamps.append(t)
        unit_index.append(np.full(len(t), i, dtype=np.int64))
    units = ArrayDict(
        id=np.array([f"chan{i:03d}" for i in range(len(nwbfile.units))], dtype="S"),
        count=np.array([len(t) for t in timestamps], dtype=np.int64),
    )
    spikes = IrregularTimeSeries(
        timestamps=np.concatenate(timestamps),
        unit_index=np.concatenate(unit_index),
        domain="auto",
    )
    spikes.sort()
    return spikes, units


def mask_to_interval(mask: np.ndarray, timestamps: np.ndarray) -> Interval:
    r"""Convert a per-sample boolean mask to an :obj:`Interval`. Each run of ``True``
    samples ``i..j`` becomes ``[timestamps[i] - dt/2, timestamps[j] + dt/2)``, so masked
    samples fall inside and unmasked samples outside, with half a sample of margin
    against floating-point error."""
    if len(mask) != len(timestamps):
        raise ValueError(
            f"mask has {len(mask)} samples but timestamps has {len(timestamps)}"
        )
    if not mask.any():
        return Interval(start=np.array([]), end=np.array([]))
    dt = np.median(np.diff(timestamps))
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return Interval(start=timestamps[first] - dt / 2, end=timestamps[last] + dt / 2)


def trials_from_numbers(trial_num: np.ndarray, timestamps: np.ndarray) -> Interval:
    r"""H1 has no trials table; trial boundaries come from the per-sample trial number."""
    dt = np.median(np.diff(timestamps))
    change = np.flatnonzero(np.diff(trial_num) != 0) + 1
    first = np.concatenate([[0], change])
    last = np.concatenate([change - 1, [len(trial_num) - 1]])
    return Interval(
        start=timestamps[first] - dt / 2,
        end=timestamps[last] + dt / 2,
        number=trial_num[first].astype(np.int64),
    )


def assign_domains(data: Data, split: str) -> None:
    r"""Held-in calibration -> train, held-in minival -> valid. Held-out calibration
    files are few-shot calibration data for later sessions (FALCON's evaluation data
    for them is private), so they get no train/valid/test domain; select them by
    ``falcon_split``."""
    empty = Interval(start=np.array([]), end=np.array([]))
    data.train_domain = data.domain if split == "held_in_calib" else empty
    data.valid_domain = data.domain if split == "held_in_minival" else empty
    data.test_domain = empty
