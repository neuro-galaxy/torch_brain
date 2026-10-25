# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///
"""FALCON M1 (Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks): a monkey performing
reach-to-grasp, 64-channel Utah array in motor cortex, 16 EMG channels as decoding target.

Ported from neuro-galaxy/brainsets#81 by Felipe Parodi.
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
    RegularTimeSeries,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.dandi import (
    download_file,
    extract_spikes_from_nwbfile,
    get_nwb_asset_list,
)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

# FALCON file groups, from the file names
FALCON_SPLITS = ("held_in_calib", "held_in_minival", "held_out_calib")


class Pipeline(BrainsetPipeline):
    brainset_id = "karpowicz_pandarinath_falcon_m1_2024"
    dandiset_id = "DANDI:000941/draft"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        manifest = []
        for asset in get_nwb_asset_list(cls.dandiset_id):
            split, date = parse_filename(asset.path)
            manifest.append(
                {"id": f"{split}_{date}", "path": asset.path, "url": asset.download_url}
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
        split, date = parse_filename(str(fpath))
        session_id = f"monkey_l_{date}_{split}"
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000941/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000941",
            description="FALCON M1: threshold crossings from a 64-channel Utah array in "
            "motor cortex of a monkey performing reach-to-grasp, with 16 EMG channels as "
            "the decoding target. Held-in sessions are for training; held-out sessions are "
            "later recording days, released as small calibration sets for few-shot "
            "adaptation.",
        )

        self.update_status("Loading NWB")
        with NWBHDF5IO(fpath, "r") as io:
            nwbfile = io.read()

            subject = SubjectDescription(id="monkey_l", species="MACACA_MULATTA")
            session_description = SessionDescription(
                id=session_id,
                recording_date=datetime.datetime.strptime(date, "%Y%m%d"),
                task="REACHING",
            )
            device_description = DeviceDescription(
                id=f"monkey_l_{date}", recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS"
            )

            self.update_status("Extracting Spikes")
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS"
            )

            self.update_status("Extracting EMG")
            emg, timestamps, muscles = extract_emg(nwbfile)
            eval_mask = nwbfile.acquisition["eval_mask"].data[:].astype(bool)
            eval_intervals = mask_to_interval(eval_mask, timestamps)

            self.update_status("Extracting Trials")
            trials = extract_trials(nwbfile)

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            emg=emg,
            muscles=muscles,
            trials=trials,
            falcon_eval_intervals=eval_intervals,
            domain=emg.domain,
        )
        data.falcon_split = split
        assign_domains(data, split)

        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file)


def parse_filename(path: str) -> tuple[str, str]:
    r"""``sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb`` ->
    ``("held_in_calib", "20120924")``."""
    name = path.split("/")[-1]
    m = re.search(r"sub-[^-_]+-(held-(?:in|out))-(calib|minival)_ses-(\d{8})", name)
    if m is None:
        raise ValueError(f"Unexpected FALCON file name: {name}")
    split = f"{m.group(1).replace('-', '_')}_{m.group(2)}"
    assert split in FALCON_SPLITS, split
    return split, m.group(3)


def extract_emg(nwbfile) -> tuple[RegularTimeSeries, np.ndarray, ArrayDict]:
    r"""The 16 preprocessed EMG channels share one 20 ms time base."""
    container = nwbfile.acquisition["preprocessed_emg"]
    names = list(container.time_series)
    timestamps = container.get_timeseries(names[0]).timestamps[:]
    for name in names[1:]:
        if not np.array_equal(container.get_timeseries(name).timestamps[:], timestamps):
            raise ValueError(f"EMG channel {name} has its own timestamps")
    dt = np.diff(timestamps)
    if not np.allclose(dt, dt[0]):
        raise ValueError("EMG timestamps are not regularly sampled")
    data = np.stack([container.get_timeseries(n).data[:] for n in names], axis=1)
    emg = RegularTimeSeries(
        sampling_rate=1.0 / dt[0],
        data=data.astype(np.float32),
        domain_start=timestamps[0],
        domain="auto",
    )
    return emg, timestamps, ArrayDict(name=np.array(names, dtype="S"))


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


def extract_trials(nwbfile) -> Interval:
    table = nwbfile.trials.to_dataframe().reset_index()
    table = table.rename(columns={"start_time": "start", "stop_time": "end"})
    # free-text columns (e.g. trial result codes) do not serialize to HDF5
    table = table.drop(
        columns=[c for c in table.columns if table[c].dtype == object], errors="ignore"
    )
    return Interval.from_dataframe(table)


def assign_domains(data: Data, split: str) -> None:
    r"""Held-in calibration -> train, held-in minival -> valid. Held-out calibration
    files are few-shot calibration data for later sessions (FALCON's evaluation data
    for them is private), so they get no train/valid/test domain; select them by
    ``falcon_split``."""
    empty = Interval(start=np.array([]), end=np.array([]))
    data.train_domain = data.domain if split == "held_in_calib" else empty
    data.valid_domain = data.domain if split == "held_in_minival" else empty
    data.test_domain = empty
