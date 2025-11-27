from argparse import ArgumentParser
import datetime

import h5py
from pynwb import NWBHDF5IO
from temporaldata import Data, IrregularTimeSeries, Interval
import pandas as pd

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
    get_nwb_asset_list,
    download_file,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "pei_pandarinath_nlb_2021"
    dandiset_id = "DANDI:000140/0.220113.0408"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]

        for m in manifest_list:
            path = m["path"]
            m["id"] = "jenkins_maze_test" if "test" in path else "jenkins_maze_train"

        manifest = pd.DataFrame(manifest_list).set_index("id")

        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )
        return fpath

    def process(self, fpath):
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        # intiantiate a DatasetBuilder which provides utilities for processing data
        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000140/0.220113.0408",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000140",
            description="This dataset contains sorted unit spiking times and behavioral"
            " data from a macaque performing a delayed reaching task. The experimental task"
            " was a center-out reaching task with obstructing barriers forming a maze,"
            " resulting in a variety of straight and curved reaches.",
        )

        # open file
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.update_status("Extracting Metadata")
        # extract subject metadata
        # this dataset is from dandi, which has structured subject metadata, so we
        # can use the helper function extract_subject_from_nwb
        subject = extract_subject_from_nwb(nwbfile)

        # extract experiment metadata
        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        session_id = f"{subject.id}_maze"
        if "test" in str(fpath):
            session_id += "_test"
        else:
            session_id += "_train"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # register session
        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.REACHING,
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # extract spiking activity
        # this data is from dandi, we can use our helper function
        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes_from_nwbfile(
            nwbfile,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # extract data about trial structure
        self.update_status("Extracting Trials")
        trials = extract_trials(nwbfile)

        data = Data(
            brainset=brainset_description,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            # stimuli and behavior
            trials=trials,
            # domain
            domain="auto",
        )

        if not "test" in str(fpath):
            self.update_status("Creating Splits")
            # extract behavior
            data.hand, data.eye = extract_behavior(nwbfile, trials)

            # report accuracy only on the evaluation intervals
            data.nlb_eval_intervals = Interval(
                start=trials.move_onset_time - 0.05,
                end=trials.move_onset_time + 0.65,
            )

            # split and register trials into train, validation and test
            train_trials, valid_trials = trials.select_by_mask(
                trials.train_mask_nwb
            ).split([0.8, 0.2], shuffle=True, random_seed=42)
            test_trials = trials.select_by_mask(trials.test_mask_nwb)

            data.set_train_domain(train_trials)
            data.set_valid_domain(valid_trials)
            data.set_test_domain(test_trials)

        # close file
        io.close()

        # save data to disk
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "split": "split_indicator",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    # the dataset has pre-defined train/valid splits, we will use the valid split
    # as our test
    train_mask_nwb = trial_table.split_indicator.to_numpy() == "train"
    test_mask_nwb = trial_table.split_indicator.to_numpy() == "val"

    trials.train_mask_nwb = (
        train_mask_nwb  # Naming with "_" since train_mask is reserved
    )
    trials.test_mask_nwb = test_mask_nwb  # Naming with "_" since test_mask is reserved

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["hand_vel"].timestamps[:]
    hand_pos = nwbfile.processing["behavior"]["hand_pos"].data[:]
    hand_vel = nwbfile.processing["behavior"]["hand_vel"].data[:]
    eye_pos = nwbfile.processing["behavior"]["eye_pos"].data[:]

    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos=hand_pos,
        vel=hand_vel,
        domain="auto",
    )

    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos=eye_pos,
        domain="auto",
    )

    return hand, eye
